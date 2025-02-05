import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchmetrics
import random

from einops import rearrange  # For reshaping tensors

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
# 1. Combined Loss Function
# ------------------------------

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        """
        Args:
            logits (torch.Tensor): Predicted logits with shape [B, C, H, W].
            true (torch.Tensor): Ground truth masks with shape [B, H, W].

        Returns:
            torch.Tensor: Dice loss.
        """
        probs = nn.functional.softmax(logits, dim=1)  # Convert logits to probabilities
        true_one_hot = nn.functional.one_hot(true, num_classes=probs.shape[1])  # [B, H, W, C]
        true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)  # Dimensions to sum over: batch, height, width
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy Loss and Dice Loss.
    """
    def __init__(self, weight=None, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(smooth)

    def forward(self, logits, true):
        """
        Args:
            logits (torch.Tensor): Predicted logits with shape [B, C, H, W].
            true (torch.Tensor): Ground truth masks with shape [B, H, W].

        Returns:
            torch.Tensor: Combined loss.
        """
        ce_loss = self.ce(logits, true)
        dice_loss = self.dice(logits, true)
        return ce_loss + dice_loss

# ------------------------------
# 2. Dataset and Transforms
# ------------------------------

class DoubleConv(nn.Module):
    """
    A block consisting of two convolutional layers each followed by BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ResizeAndToTensor:
    """
    Custom transform to resize images and masks to a fixed size and convert them to tensors.
    """
    def __init__(self, size=(512, 512), augment=False):
        self.size = size  # (height, width)
        self.augment = augment
        self.normalize = transforms.Normalize(mean=[0.5],
                                             std=[0.5])

    def __call__(self, image, mask):
        """
        Args:
            image (PIL Image): Input image.
            mask (np.array or PIL Image): Corresponding mask as a NumPy array or PIL Image.

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
        if isinstance(mask, np.ndarray):
            mask_pil = Image.fromarray(mask)
        elif isinstance(mask, Image.Image):
            mask_pil = mask
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        mask_pil = resize(
            mask_pil,
            self.size,
            interpolation=InterpolationMode.NEAREST
        )
        # Convert back to NumPy array if it was initially
        if isinstance(mask, np.ndarray):
            mask = np.array(mask_pil, dtype=np.uint8)
        else:
            mask = np.array(mask_pil, dtype=np.uint8)

        # Data Augmentation (if enabled)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = Image.fromarray(mask).transpose(Image.FLIP_LEFT_RIGHT)
                mask = np.array(mask, dtype=np.uint8)

            # Random vertical flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = Image.fromarray(mask).transpose(Image.FLIP_TOP_BOTTOM)
                mask = np.array(mask, dtype=np.uint8)

            # Random rotation by 0°, 90°, 180°, or 270°
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            if angle != 0:
                image = image.rotate(angle)
                mask = Image.fromarray(mask).rotate(angle)
                mask = np.array(mask, dtype=np.uint8)

            # Add more augmentations as needed

        # Convert image and mask to tensors
        image_tensor = to_tensor(image)             # [1, H, W], float32 in [0,1]
        image_tensor = self.normalize(image_tensor)

        mask_tensor  = torch.from_numpy(mask).long()# [H, W], dtype=torch.int64

        return image_tensor, mask_tensor

class BinarySegDataset(Dataset):
    """
    Custom Dataset for binary image segmentation.
    """
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        """
        Args:
            images_dir (str): Directory with input images.
            masks_dir (str): Directory with corresponding mask files (.npy or image files).
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
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        # Load mask
        # Attempt to find mask with the same base name and supported extensions
        mask_extensions = ['.npy', '.png', '.jpg', '.jpeg']
        mask_path = None
        for ext in mask_extensions:
            potential_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.isfile(potential_path):
                mask_path = potential_path
                break
        if mask_path is None:
            raise FileNotFoundError(f"[Error] Mask file not found for image: {img_filename}")

        # Load mask
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
            mask = Image.fromarray(mask)
        else:
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        # Remap any label >1 to 1 to ensure binary segmentation
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 1, 1, mask_np).astype(np.uint8)
        mask = Image.fromarray(mask_np)

        # Apply transformations if any
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

# ------------------------------
# 3. SegViT Model Definition
# ------------------------------

class SegViT(nn.Module):
    """
    SegViT model combining CNN and Transformer Encoder for image segmentation.
    """
    def __init__(self, img_size=512, num_classes=2, in_channels=1, 
                 vit_model='vit_base_patch16_384', pretrained=True):
        super(SegViT, self).__init__()
        
        # CNN Encoder (e.g., ResNet34)
        self.cnn_encoder = models.resnet34(pretrained=pretrained)
        # Modify first convolutional layer for single-channel input
        self.cnn_encoder.conv1 = nn.Conv2d(in_channels, 
                                           self.cnn_encoder.conv1.out_channels, 
                                           kernel_size=self.cnn_encoder.conv1.kernel_size, 
                                           stride=self.cnn_encoder.conv1.stride, 
                                           padding=self.cnn_encoder.conv1.padding, 
                                           bias=self.cnn_encoder.conv1.bias is not None)
        with torch.no_grad():
            if in_channels == 1:
                # Average the weights across the channel dimension to simulate grayscale input
                self.cnn_encoder.conv1.weight = nn.Parameter(self.cnn_encoder.conv1.weight.mean(dim=1, keepdim=True))
            else:
                raise ValueError("Only single-channel modification is supported.")
        
        self.cnn_initial = nn.Sequential(
            self.cnn_encoder.conv1,  # [B, 64, H/2, W/2]
            self.cnn_encoder.bn1,
            self.cnn_encoder.relu,
            self.cnn_encoder.maxpool  # [B, 64, H/4, W/4]
        )
        self.cnn_layer1 = self.cnn_encoder.layer1  # [B, 64, H/4, W/4]
        self.cnn_layer2 = self.cnn_encoder.layer2  # [B, 128, H/8, W/8]
        self.cnn_layer3 = self.cnn_encoder.layer3  # [B, 256, H/16, W/16]
        self.cnn_layer4 = self.cnn_encoder.layer4  # [B, 512, H/32, W/32]
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=24, dim_feedforward=3072, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=24)
        # Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, 768))  # [1, 256, 768]
        
        # Projection Layers to match embedding dimensions
        self.proj_in = nn.Linear(512, 768)   # CNN output (512) -> Transformer input (768)
        self.proj_out = nn.Linear(768, 512)  # Transformer output (768) -> CNN output (512)
        
        # Positional Encoding for Transformer output
        self.pos_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64 + 64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # CNN Encoder
        x0 = self.cnn_initial(x)          # [B, 64, H/4, W/4]
        x1 = self.cnn_layer1(x0)          # [B, 64, H/4, W/4]
        x2 = self.cnn_layer2(x1)          # [B, 128, H/8, W/8]
        x3 = self.cnn_layer3(x2)          # [B, 256, H/16, W/16]
        x4 = self.cnn_layer4(x3)          # [B, 512, H/32, W/32]
        
        # Transformer Encoder
        B, C, H, W = x4.shape
        x4_flat = x4.flatten(2).transpose(1, 2)  # [B,256,512]
        x4_flat = self.proj_in(x4_flat)          # [B,256,768]
        x4_trans = x4_flat + self.pos_embedding  # [B,256,768]
        x4_trans = x4_trans.transpose(0,1)      # [256,B,768]
        x4_trans = self.transformer_encoder(x4_trans)  # [256,B,768]
        x4_trans = x4_trans.transpose(0,1)      # [B,256,768]
        x4_trans = self.proj_out(x4_trans)      # [B,256,512]
        
        # Correctly apply pos_encoder
        # Reshape to [B,16,16,512] to apply pos_encoder on last dimension
        x4_trans = x4_trans.view(B, 16, 16, 512)    # [B,16,16,512]
        x4_trans = self.pos_encoder(x4_trans)       # [B,16,16,512]
        x4_trans = x4_trans.permute(0, 3, 1, 2)    # [B,512,16,16]
        x4 = x4_trans + x4                        # Residual connection [B,512,16,16]
        
        # Decoder
        up4 = self.up4(x4)                  # [B,256,32,32]
        merge4 = torch.cat([up4, x3], dim=1)  # [B,512,32,32]
        conv4 = self.conv4(merge4)           # [B,256,32,32]
        
        up3 = self.up3(conv4)                # [B,128,64,64]
        merge3 = torch.cat([up3, x2], dim=1)  # [B,256,64,64]
        conv3 = self.conv3(merge3)           # [B,128,64,64]
        
        up2 = self.up2(conv3)                # [B,64,128,128]
        merge2 = torch.cat([up2, x1], dim=1)  # [B,128,128,128]
        conv2 = self.conv2(merge2)           # [B,64,128,128]
        
        out = self.out_conv(conv2)           # [B,num_classes,128,128]
        out = nn.functional.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)  # [B,num_classes,512,512]
        
        return out

# ------------------------------
# 4. Early Stopping Class
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

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx, scaler=None, max_grad_norm=1.0):
    """
    Trains the model for one epoch using mixed precision.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Combined loss function.
        device (torch.device): Device to run on.
        epoch_idx (int): Current epoch index.
        scaler (GradScaler, optional): Gradient scaler for mixed precision.
        max_grad_norm (float): Maximum norm for gradient clipping.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    print(f"--- [Train] Starting epoch {epoch_idx+1} ---")
    # Use tqdm to show progress
    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        if batch_idx % 10 == 0:
            print(f"[Train] Batch {batch_idx}, Loss: {loss.item():.4f}")
    average_loss = total_loss / len(loader.dataset)
    print(f"--- [Train] Epoch {epoch_idx+1} Average Loss: {average_loss:.4f} ---")
    return average_loss

def validate_one_epoch(model, loader, criterion, device, epoch_idx, metrics=None):
    """
    Validates the model for one epoch.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the validation set.
        criterion (Loss): Combined loss function.
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
            images = images.to(device)
            masks = masks.to(device)
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
    dice_scores = []
    iou_scores = []
    accuracy_scores = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Testing", leave=False)):
            images = images.to(device)
            masks = masks.to(device)  # [B, H, W]

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
        image_batch = image.unsqueeze(0).to(device)  # [1, C, H, W]

        with torch.no_grad():
            output = model(image_batch)  # [1, n_classes, H, W]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]

        image_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        mask_np = mask.cpu().numpy()  # [H, W]

        # Handle image normalization for visualization
        image_np = image_np * 0.5 + 0.5  # Unnormalize
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        if image_np.ndim == 2:
                # If single-channel, convert to 3-channel for visualization
            image_np = np.stack([image_np]*3, axis=-1)

            # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image_np, cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title("Predicted Mask" + (" (Post-Processed)" if post_process_flag else ""))
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

# ------------------------------
# 7. Main Function
# ------------------------------

def main():
    # ------------------------------
    # Define Your Directories Here
    # ------------------------------
    # Replace these paths with your actual dataset paths
    dataset_dir = "/home/yanghehao/tracklearning/segmentation/"  # Change this to your dataset directory
    train_images_dir = os.path.join(dataset_dir, "phantom_train", "images")
    train_masks_dir  = os.path.join(dataset_dir, "phantom_train", "masks")
    test_images_dir  = os.path.join(dataset_dir, "phantom_test", "images")
    test_masks_dir   = os.path.join(dataset_dir, "phantom_test", "masks")

    # Hyperparameters
    batch_size    = 8
    num_epochs    = 50
    learning_rate = 1e-4
    val_split     = 0.2
    save_path     = "best_segvit_model.pth"
    patience      = 7
    post_process_flag = False  # Set to True to apply post-processing

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
        f for f in os.listdir(train_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_train_images)} training image files in {train_images_dir}")

    if len(all_train_images) == 0:
        print("[Error] No training image files found. Check your training path!")
        return

    # Ensure corresponding mask files exist
    all_train_images = sorted([
        f for f in all_train_images
        if any(
            os.path.isfile(os.path.join(train_masks_dir, os.path.splitext(f)[0] + ext))
            for ext in ['.npy', '.png', '.jpg', '.jpeg']
        )
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
        if any(
            os.path.isfile(os.path.join(test_masks_dir, os.path.splitext(f)[0] + ext))
            for ext in ['.npy', '.png', '.jpg', '.jpeg']
        )
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
    train_transform = ResizeAndToTensor(size=(512, 512), augment=True)
    val_transform = ResizeAndToTensor(size=(512, 512), augment=False)
    test_transform = ResizeAndToTensor(size=(512, 512), augment=False)

    train_dataset = BinarySegDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        file_list=train_files,
        transform=train_transform
    )
    val_dataset = BinarySegDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your CPU
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Changed to False for consistency
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # ------------------------------
    # Initialize Model, Loss, Optimizer, Scheduler
    # ------------------------------
    model = SegViT(img_size=512, num_classes=2, in_channels=1, 
                 vit_model='vit_base_patch16_384', pretrained=True).to(device)
    
    # Initialize Combined Loss (Cross-Entropy + Dice)
    criterion = CombinedLoss(weight=None, smooth=1e-6)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

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

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # ------------------------------
    # Training Loop
    # ------------------------------
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler=scaler)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, metrics_val)
        val_losses.append(val_loss)

        # Scheduler step
        scheduler.step()

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
        visualize_predictions(model, test_dataset, device, num_samples=10, post_process_flag=post_process_flag)
    else:
        print("[Warning] No test samples available for visualization.")

# ------------------------------
# 8. Run the Main Function
# ------------------------------

if __name__ == "__main__":
    main()
