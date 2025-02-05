import os
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import torchvision.models as models  # For ResNet encoder

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torchmetrics
import random

# ------------------------------
# 0. Reproducibility (Optional)
# ------------------------------

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# 1. Dataset and Transforms
# ------------------------------

class ResizeAndToTensor:
    """
    Custom transform to resize images and masks to a fixed size and convert them to tensors.
    Removes data augmentations like flipping and rotating.
    """
    def __init__(self, size=(512, 512)):
        self.size = size  # (height, width)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    def __call__(self, image, mask):
        """
        Args:
            image (PIL Image): Input image.
            mask (np.array): Corresponding mask as a NumPy array.
        
        Returns:
            image_tensor (torch.Tensor): Resized and normalized image tensor.
            mask_tensor (torch.Tensor): Resized mask tensor (Long type).
        """
        # Resize image using bilinear interpolation
        image = resize(
            image,
            self.size,
            interpolation=InterpolationMode.BILINEAR
        )
        # Resize mask using nearest-neighbor interpolation to preserve label integrity
        mask_pil = Image.fromarray(mask)
        mask_pil = resize(
            mask_pil,
            self.size,
            interpolation=InterpolationMode.NEAREST
        )
        # Convert back to NumPy array
        mask = np.array(mask_pil, dtype=np.uint8)

        # Convert image and mask to tensors
        image_tensor = to_tensor(image)             # [C, H, W], float32 in [0,1]
        image_tensor = self.normalize(image_tensor)
        mask_tensor  = torch.from_numpy(mask).long()# [H, W], dtype=torch.int64

        return image_tensor, mask_tensor

class BinarySegDataset(Dataset):
    """
    Custom Dataset for binary image segmentation.
    Assumes that mask files are stored as .npy files.
    """
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        """
        Args:
            images_dir (str): Directory with input images.
            masks_dir (str): Directory with corresponding mask files (.npy).
            file_list (list): List of image filenames.
            transform (callable, optional): Transform to be applied on a sample.
        """
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.file_list  = file_list
        self.transform  = transform

        # Debug: Print how many files in this split
        print(f"[Dataset] Initialized with {len(file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask pair at the specified index.
        """
        img_filename = self.file_list[idx]
        base_name = os.path.splitext(img_filename)[0]

        # Load image
        img_path = os.path.join(self.images_dir, img_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"[Error] Image file not found: {img_path}")
        image = Image.open(img_path).convert("RGB")

        # Load mask
        mask_path = os.path.join(self.masks_dir, base_name + ".npy")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"[Error] Mask file not found: {mask_path}")
        mask = np.load(mask_path)

        # Remap label >1 to 1 to ensure binary segmentation
        mask = np.where(mask > 1, 1, mask).astype(np.uint8)

        # Apply transform
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

# ------------------------------
# 2. Transformer Module Implementation
# ------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512*512):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x):
        # x shape: [B, N, D]
        return x + self.position_embeddings[:, :x.size(1), :]

class TransformerBottleneck(nn.Module):
    def __init__(self, embed_dim=2048, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerBottleneck, self).__init__()
        self.embed_dim = embed_dim
        self.positional_encoding = PositionalEncoding(d_model=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            dim_feedforward=embed_dim*4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.size()
        N = H * W
        # Flatten spatial dimensions
        x = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        x = self.positional_encoding(x)        # [B, N, C]
        x = self.transformer_encoder(x)        # [B, N, C]
        x = self.layer_norm(x)                 # [B, N, C]
        # Reshape back to [B, C, H, W]
        x = x.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        return x

# ------------------------------
# 3. U-Net Model Definition with Transformer
# ------------------------------

class DoubleConv(nn.Module):
    """
    A block consisting of two convolutional layers each followed by BatchNorm, ReLU, and Dropout.
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.5):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x):
        return self.conv(x)

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, true):
        probs = nn.functional.softmax(logits, dim=1)
        true_one_hot = nn.functional.one_hot(true, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class ResNetUNetWithTransformer(nn.Module):
    """
    U-Net architecture with ResNet encoder and Transformer bottleneck.
    """
    def __init__(self, n_classes=2, encoder_name='resnet50', pretrained=True, transformer_layers=6, transformer_heads=8, dropout_p=0.5):
        """
        Args:
            n_classes (int): Number of output classes.
            encoder_name (str): Name of the ResNet encoder to use ('resnet18', 'resnet34', 'resnet50', etc.).
            pretrained (bool): Whether to use pretrained ResNet weights.
            transformer_layers (int): Number of Transformer encoder layers.
            transformer_heads (int): Number of attention heads in Transformer.
            dropout_p (float): Dropout probability.
        """
        super(ResNetUNetWithTransformer, self).__init__()
        
        # Initialize ResNet encoder
        if encoder_name == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet34':
            self.encoder = models.resnet34(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif encoder_name == 'resnet101':
            self.encoder = models.resnet101(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif encoder_name == 'resnet152':
            self.encoder = models.resnet152(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError("Unsupported ResNet variant")

        # Encoder layers
        self.initial = nn.Sequential(
            self.encoder.conv1,  # [B, 64, H/2, W/2]
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool  # [B, 64, H/4, W/4]
        )
        self.encoder_layer1 = self.encoder.layer1  # [B, 256, H/4, W/4] for ResNet101
        self.encoder_layer2 = self.encoder.layer2  # [B, 512, H/8, W/8]
        self.encoder_layer3 = self.encoder.layer3  # [B, 1024, H/16, W/16]
        self.encoder_layer4 = self.encoder.layer4  # [B, 2048, H/32, W/32]

        # Transformer Bottleneck
        self.transformer = TransformerBottleneck(
            embed_dim=encoder_channels[4],
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=0.1
        )
        
        # Decoder layers
        # up4: [B, 2048, H/32, W/32] -> [B,1024,H/16,W/16]
        self.up4 = nn.ConvTranspose2d(encoder_channels[4], encoder_channels[3], kernel_size=2, stride=2)
        self.conv4 = DoubleConv(encoder_channels[3] + encoder_channels[3], encoder_channels[3], dropout_p=dropout_p)

        # up3: [B,1024,H/16,W/16] -> [B,512,H/8,W/8]
        self.up3 = nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], kernel_size=2, stride=2)
        self.conv3 = DoubleConv(encoder_channels[2] + encoder_channels[2], encoder_channels[2], dropout_p=dropout_p)

        # up2: [B,512,H/8,W/8] -> [B,256,H/4,W/4]
        self.up2 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], kernel_size=2, stride=2)
        self.conv2 = DoubleConv(encoder_channels[1] + encoder_channels[1], encoder_channels[1], dropout_p=dropout_p)

        # up1: [B,256,H/4,W/4] -> [B,64,H/4,W/4] (no upsampling)
        self.up1 = nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], kernel_size=1, stride=1)
        self.conv1 = DoubleConv(encoder_channels[0] + encoder_channels[0], encoder_channels[0], dropout_p=dropout_p)

        # Output convolution
        self.out_conv = nn.Conv2d(encoder_channels[0], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.initial(x)           # [B,64,H/4,W/4]
        x1 = self.encoder_layer1(x0)   # [B,256,H/4,W/4] for ResNet101
        x2 = self.encoder_layer2(x1)   # [B,512,H/8,W/8]
        x3 = self.encoder_layer3(x2)   # [B,1024,H/16,W/16]
        x4 = self.encoder_layer4(x3)   # [B,2048,H/32,W/32]

        # Transformer Bottleneck
        x4 = self.transformer(x4)      # [B,2048,H/32,W/32]

        # Decoder
        up4 = self.up4(x4)             # [B,1024,H/16,W/16]
        merge4 = torch.cat([up4, x3], dim=1)  # [B,2048,H/16,W/16]
        conv4 = self.conv4(merge4)     # [B,1024,H/16,W/16]

        up3 = self.up3(conv4)          # [B,512,H/8,W/8]
        merge3 = torch.cat([up3, x2], dim=1)  # [B,1024,H/8,W/8]
        conv3 = self.conv3(merge3)     # [B,512,H/8,W/8]

        up2 = self.up2(conv3)          # [B,256,H/4,W/4]
        merge2 = torch.cat([up2, x1], dim=1)  # [B,512,H/4,W/4]
        conv2 = self.conv2(merge2)     # [B,256,H/4,W/4]

        up1 = self.up1(conv2)          # [B,64,H/4,W/4]
        merge1 = torch.cat([up1, x0], dim=1)  # [B,128,H/4,W/4]
        conv1 = self.conv1(merge1)     # [B,64,H/4,W/4]

        # Output convolution
        out = self.out_conv(conv1)     # [B,2,H/4,W/4]

        # Upsample to original size
        out = nn.functional.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)  # [B,2,H,W]

        return out

# ------------------------------
# 4. Early Stopping Implementation
# ------------------------------

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0.0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"[EarlyStopping] Initial best loss: {self.best_loss:.4f}")
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("[EarlyStopping] Early stopping triggered.")
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Validation loss improved to: {self.best_loss:.4f}")

# ------------------------------
# 5. Training and Validation Functions
# ------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx, scaler):
    """
    Trains the model for one epoch using Mixed Precision.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        device (torch.device): Device to run on.
        epoch_idx (int): Current epoch index.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    print(f"--- [Train] Starting epoch {epoch_idx+1} ---")
    # Use tqdm to show progress
    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        if batch_idx % 10 == 0:
            print(f"[Train] Batch {batch_idx}, Loss: {loss.item():.4f}")
    average_loss = total_loss / len(loader.dataset)
    print(f"--- [Train] Epoch {epoch_idx+1} Average Loss: {average_loss:.4f} ---")
    return average_loss

def validate_one_epoch(model, loader, criterion, device, epoch_idx, metrics=None):
    """
    Validates the model for one epoch using Mixed Precision.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the validation set.
        criterion (Loss): Loss function.
        device (torch.device): Device to run on.
        epoch_idx (int): Current epoch index.
        metrics (dict, optional): Dictionary of TorchMetrics to compute.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0
    print(f"--- [Val] Starting epoch {epoch_idx+1} ---")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Validation", leave=False)):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)
            if batch_idx % 10 == 0:
                print(f"[Val] Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Compute metrics
            if metrics:
                preds = torch.argmax(outputs, dim=1)
                for metric in metrics.values():
                    metric(preds, masks)

    average_loss = total_loss / len(loader.dataset)
    print(f"--- [Val] Epoch {epoch_idx+1} Average Loss: {average_loss:.4f} ---")
    
    # Compute metric values
    if metrics:
        metric_values = {k: v.compute().item() for k, v in metrics.items()}
        # Reset metrics
        for metric in metrics.values():
            metric.reset()
    else:
        metric_values = {}
    
    # Optionally, print metrics
    if metrics:
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metric_values.items()])
        print(f"--- [Val] Epoch {epoch_idx+1} Metrics: {metric_str} ---")
    
    return average_loss

# ------------------------------
# 6. Testing and Metrics
# ------------------------------

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Computes the Dice coefficient for binary masks.

    Args:
        pred (torch.Tensor): Predicted mask [H, W].
        target (torch.Tensor): Ground truth mask [H, W].
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()

def test_segmentation(model, loader, device, metrics=None):
    """
    Evaluates the model on the test set and computes the average Dice coefficient and other metrics.
    
    Args:
        model (nn.Module): The trained segmentation model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run on.
        metrics (dict, optional): Dictionary of TorchMetrics.
    
    Returns:
        dict: Dictionary of average metrics on the test set.
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Testing", leave=False)):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)  # [B, H, W]

            with torch.cuda.amp.autocast():
                outputs = model(images)    # [B, n_classes, H, W]
                preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # Compute metrics
            if metrics:
                for metric in metrics.values():
                    metric(preds, masks)

    # Compute metric values
    if metrics:
        metric_values = {k: v.compute().item() for k, v in metrics.items()}
        # Reset metrics
        for metric in metrics.values():
            metric.reset()
    else:
        # If metrics are not provided, compute only Dice
        dice_scores = []
        for pred, mask in zip(preds, masks):
            dice = dice_coefficient(pred, mask)
            dice_scores.append(dice)
        metric_values = {
            'dice': np.mean(dice_scores) if dice_scores else 0
        }

    # Print metrics
    if metrics:
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metric_values.items()])
        print(f"--- [Test] Metrics: {metric_str} ---")
    else:
        print(f"--- [Test] Dice Coefficient: {metric_values['dice']:.4f} ---")

    return metric_values

# ------------------------------
# 7. Visualization of Predictions
# ------------------------------

def visualize_predictions(model, dataset, device, num_samples=3, post_process_flag=False):
    """
    Visualizes predictions alongside the original images and ground truth masks.

    Args:
        model (nn.Module): The trained segmentation model.
        dataset (Dataset): The test dataset.
        device (torch.device): Device to run on.
        num_samples (int): Number of samples to visualize.
        post_process_flag (bool): Whether to apply post-processing to predictions.
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0).to(device, non_blocking=True)  # [1, C, H, W]

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(image_batch)  # [1, n_classes, H, W]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]

        # No post-processing as cv2 is removed
        # If needed, you can implement torch-based post-processing here

        image_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        mask_np = mask.cpu().numpy()  # [H, W]

        # Handle image normalization for visualization
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image_np)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

# ------------------------------
# 8. Main Function
# ------------------------------

def main():
    # ------------------------------
    # Define Your Directories Here
    # ------------------------------
    # Training and Validation directories (split from the same folder)
    images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/images/"
    masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_train/masks/"

    # Testing directories (completely separate)
    test_images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/images/"
    test_masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_test/masks/"

    # Hyperparameters
    batch_size    = 8  # Adjust based on GPU memory
    num_epochs    = 30
    learning_rate = 1e-4
    val_split     = 0.2
    save_path     = "best_model_resnet101.pth"
    patience      = 10
    post_process_flag = False  # Set to True to apply post-processing if implemented

    # ------------------------------
    # Device Configuration
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # ------------------------------
    # Collect Training File Lists
    # ------------------------------
    # List all training image files
    all_train_images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_train_images)} training image files in {images_dir}")

    if len(all_train_images) == 0:
        print("[Error] No training image files found. Check your training path!")
        return

    # Ensure corresponding mask files exist
    all_train_images = sorted([
        f for f in all_train_images
        if os.path.isfile(os.path.join(masks_dir, os.path.splitext(f)[0] + ".npy"))
    ])
    print(f"[Main] {len(all_train_images)} training images have corresponding masks.")

    if len(all_train_images) == 0:
        print("[Error] No training mask files found or mismatched filenames. Check your training mask path!")
        return

    # List all test image files
    all_test_images = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_test_images)} test image files in {test_images_dir}")

    if len(all_test_images) == 0:
        print("[Error] No test image files found. Check your test path!")
        return

    # Ensure corresponding test mask files exist
    all_test_images = sorted([
        f for f in all_test_images
        if os.path.isfile(os.path.join(test_masks_dir, os.path.splitext(f)[0] + ".npy"))
    ])
    print(f"[Main] {len(all_test_images)} test images have corresponding masks.")

    if len(all_test_images) == 0:
        print("[Error] No test mask files found or mismatched filenames. Check your test mask path!")
        return

    # ------------------------------
    # Train/Validation Split
    # ------------------------------
    train_files, val_files = train_test_split(
        all_train_images,
        test_size=val_split,
        random_state=42
    )
    print(f"[Main] Training samples: {len(train_files)}")
    print(f"[Main] Validation samples: {len(val_files)}")

    # ------------------------------
    # Create Datasets with Transforms
    # ------------------------------
    train_transform = ResizeAndToTensor(size=(512, 512))
    val_transform = ResizeAndToTensor(size=(512, 512))
    test_transform = ResizeAndToTensor(size=(512, 512))

    train_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        file_list=train_files,
        transform=train_transform
    )
    val_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        file_list=val_files,
        transform=val_transform
    )
    test_dataset = BinarySegDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        file_list=all_test_images,
        transform=test_transform
    )

    # ------------------------------
    # Create DataLoaders
    # ------------------------------
    num_workers = 8  # Adjust based on CPU cores

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if torch.cuda.is_available() else False
    )

    # ------------------------------
    # Initialize Model, Loss, Optimizer, Scheduler
    # ------------------------------
    model = ResNetUNetWithTransformer(
        n_classes=2, 
        encoder_name='resnet34', 
        pretrained=True, 
        transformer_layers=16, 
        transformer_heads=16,
        dropout_p=0.5
    ).to(device)
    
    # Initialize Dice Loss
    criterion = DiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize Metrics
    metrics_val = {
        'dice': torchmetrics.Dice(num_classes=2, average='macro').to(device),
        'iou': torchmetrics.JaccardIndex(task='binary').to(device),
        'accuracy': torchmetrics.Accuracy(task='binary').to(device)
    }

    metrics_test = {
        'dice': torchmetrics.Dice(num_classes=2, average='macro').to(device),
        'iou': torchmetrics.JaccardIndex(task='binary').to(device),
        'accuracy': torchmetrics.Accuracy(task='binary').to(device)
    }

    # Initialize Gradient Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    # ------------------------------
    # Training Loop
    # ------------------------------
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, metrics_val)
        val_losses.append(val_loss)

        # Scheduler step
        scheduler.step(epoch + val_split)  # Using epoch count including validation split

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f">> Saved best model with Val Loss: {best_val_loss:.4f}")
            early_stopping.counter = 0  # Reset early stopping counter
        else:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("[Main] Early stopping triggered. Stopping training.")
                break

    print(f"\n[Main] Training complete. Best Validation Loss: {best_val_loss:.4f}")
    print(f"[Main] Best model saved at: {save_path}")

    # Plot Training and Validation Loss
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # ------------------------------
    # Testing
    # ------------------------------
    print("\n>>> Loading best model for testing...")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_metrics = test_segmentation(model, test_loader, device, metrics=metrics_test)
    print(f"Test Metrics: {test_metrics}")

    # ------------------------------
    # Visualization of Test Samples
    # ------------------------------
    if len(test_dataset) > 0:
        print("\n>>> Visualizing predictions on test samples...")
        visualize_predictions(model, test_dataset, device, num_samples=20, post_process_flag=post_process_flag)
    else:
        print("[Warning] No test samples available for visualization.")

# ------------------------------
# 9. Run the Main Function
# ------------------------------

if __name__ == "__main__":
    main()
