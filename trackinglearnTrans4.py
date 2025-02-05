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

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

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
# 1. Combined Loss Function
# ------------------------------

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

class CombinedLoss(nn.Module):
    """
    Combined Dice Loss and Cross-Entropy Loss.
    """
    def __init__(self, weight=None, smooth=1):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss(smooth)
        self.ce = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, logits, true):
        return self.ce(logits, true) + self.dice(logits, true)

# ------------------------------
# 2. Dataset and Transforms
# ------------------------------

class ResizeAndToTensor:
    """
    Custom transform to resize images and masks to a fixed size, apply controlled augmentations,
    and convert them to tensors.
    """
    def __init__(self, size=(512, 512), augment=False):
        self.size = size  # (height, width)
        self.augment = augment
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
        
        # Controlled Augmentations
        if self.augment:
            # Random brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            brightness_factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(brightness_factor)
            
            # Random contrast adjustment
            enhancer = ImageEnhance.Contrast(image)
            contrast_factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(contrast_factor)
            
            # Additional augmentations can be added here

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
# 3. Attention Gates Implementation
# ------------------------------

class AttentionGate(nn.Module):
    """
    Attention Gate as described in "Attention U-Net: Learning Where to Look for the Pancreas"
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        # g: gating signal (from decoder)
        # x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ------------------------------
# 4. Custom Transformer Bottleneck Implementation
# ------------------------------

class TransformerBottleneck(nn.Module):
    """
    Custom Transformer Bottleneck for U-Net++ architecture.
    Utilizes standard TransformerEncoder layers to capture global dependencies.
    """
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerBottleneck, self).__init__()
        self.embed_dim = embed_dim
        self.flatten = nn.Flatten(2)  # [B, C, H, W] -> [B, C, H*W]
        self.pos_embedding = nn.Parameter(torch.zeros(1, embed_dim, 32*32))  # [1, 512, 1024]
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.unflatten = nn.Unflatten(2, (32, 32))  # [B, 512, 1024] -> [B, 512, 32, 32]

    def forward(self, x):
        # x: [B, C, H, W] where H=32, W=32
        B, C, H, W = x.size()
        x = self.flatten(x)  # [B, C, H*W] = [B, 512, 1024]
        x = x + self.pos_embedding  # [B, 512, 1024]
        x = x.permute(0, 2, 1)  # [B, 1024, 512]
        x = self.transformer_encoder(x)  # [B, 1024, 512]
        x = x.permute(0, 2, 1)  # [B, 512, 1024]
        x = self.unflatten(x)    # [B, 512, 32, 32]
        return x

# ------------------------------
# 5. U-Net++ Model Definition with Transformer and Attention Gates
# ------------------------------

class DoubleConv(nn.Module):
    """
    A block consisting of two convolutional layers each followed by BatchNorm, ReLU, and Dropout.
    Includes a residual connection.
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
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)

class UNetPlusPlus(nn.Module):
    """
    U-Net++ architecture with nested skip connections and a Transformer bottleneck.
    Includes Attention Gates and residual connections.
    """
    def __init__(self, in_channels=3, out_channels=2, dropout_p=0.5):
        super(UNetPlusPlus, self).__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64, dropout_p)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(64, 128, dropout_p)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(128, 256, dropout_p)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(256, 512, dropout_p)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Transformer Bottleneck
        self.transformer = TransformerBottleneck(
            embed_dim=512,
            num_heads=16,
            num_layers=16,
            dropout=0.1
        )

        # Decoder
        # Level 4
        self.up4_0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4_0 = AttentionGate(F_g=256, F_l=512, F_int=128)  # F_l corrected to 512
        self.dec4_0 = DoubleConv(256 + 512, 256, dropout_p)      # Merge up4_0 and att4_0

        # Level 3
        self.up3_0 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3_0 = AttentionGate(F_g=128, F_l=256, F_int=64)   # F_l corrected to 256
        self.dec3_0 = DoubleConv(128 + 256, 128, dropout_p)      # Merge up3_0 and att3_0

        # Level 2
        self.up2_0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2_0 = AttentionGate(F_g=64, F_l=128, F_int=32)    # F_l corrected to 128
        self.dec2_0 = DoubleConv(64 + 128, 64, dropout_p)        # Merge up2_0 and att2_0

        # Final Output
        self.final_conv = nn.Conv2d(64 + 64, out_channels, kernel_size=1)  # Merge up1_0 and enc1

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)          # [B,64,H,W]
        pool1 = self.pool1(enc1)     # [B,64,H/2,W/2]

        enc2 = self.enc2(pool1)      # [B,128,H/2,W/2]
        pool2 = self.pool2(enc2)     # [B,128,H/4,W/4]

        enc3 = self.enc3(pool2)      # [B,256,H/4,W/4]
        pool3 = self.pool3(enc3)     # [B,256,H/8,W/8]

        enc4 = self.enc4(pool3)      # [B,512,H/8,W/8]
        pool4 = self.pool4(enc4)     # [B,512,H/16,W/16]

        # Transformer Bottleneck
        bottleneck = self.transformer(pool4)  # [B,512,H/16,W/16]

        # Decoder
        up4_0 = self.up4_0(bottleneck)        # [B,256,H/8,W/8]
        att4_0 = self.att4_0(up4_0, enc4)      # [B,256,H/8,W/8]
        merge4_0 = torch.cat([up4_0, att4_0], dim=1)  # [B,256+256=512,H/8,W/8]
        dec4_0 = self.dec4_0(merge4_0)         # [B,256,H/8,W/8]

        up3_0 = self.up3_0(dec4_0)            # [B,128,H/4,W/4]
        att3_0 = self.att3_0(up3_0, enc3)      # [B,128,H/4,W/4]
        merge3_0 = torch.cat([up3_0, att3_0], dim=1)  # [B,128+128=256,H/4,W/4]
        dec3_0 = self.dec3_0(merge3_0)         # [B,128,H/4,W/4]

        up2_0 = self.up2_0(dec3_0)            # [B,64,H/2,W/2]
        att2_0 = self.att2_0(up2_0, enc2)      # [B,64,H/2,W/2]
        merge2_0 = torch.cat([up2_0, att2_0], dim=1)  # [B,64+64=128,H/2,W/2]
        dec2_0 = self.dec2_0(merge2_0)         # [B,64,H/2,W/2]

        # Final upsampling
        up1_0 = nn.functional.interpolate(dec2_0, scale_factor=2, mode='bilinear', align_corners=True)  # [B,64,H,W]
        merge1_0 = torch.cat([up1_0, enc1], dim=1)  # [B,64+64=128,H,W]
        final = self.final_conv(merge1_0)            # [B,2,H,W]

        return final

# ------------------------------
# 6. Early Stopping Implementation
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
# 7. Training and Validation Functions
# ------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx, scaler, clip_grad_norm=1.0):
    """
    Trains the model for one epoch using Mixed Precision.
    
    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Combined loss function.
        device (torch.device): Device to run on.
        epoch_idx (int): Current epoch index.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision.
        clip_grad_norm (float): Maximum norm for gradient clipping.
    
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
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        
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
# 8. Testing and Metrics
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
# 9. Visualization of Predictions
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

        # Optional Post-Processing
        if post_process_flag:
            # Example: Simple morphological operations can be implemented here
            # For demonstration, this is left as a placeholder
            pass

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
# 10. Main Function
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
    num_epochs    = 50
    learning_rate = 1e-4
    val_split     = 0.2
    save_path     = "best_model_unetpp_transformer.pth"
    patience      = 2
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
    train_transform = ResizeAndToTensor(size=(512, 512), augment=True)
    val_transform = ResizeAndToTensor(size=(512, 512), augment=False)
    test_transform = ResizeAndToTensor(size=(512, 512), augment=False)

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
    model = UNetPlusPlus(
        in_channels=3, 
        out_channels=2, 
        dropout_p=0.1
    ).to(device)
    
    # Initialize Combined Loss (Dice + Cross-Entropy)
    # Without class weights
    criterion = CombinedLoss(weight=None)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler: Cosine Annealing
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

        # Scheduler step based on epoch
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

    # Plot Training and Validation Loss
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, train_losses, 'b', label='Training loss')
    plt.plot(epochs_range, val_losses, 'r', label='Validation loss')
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
# 11. Run the Main Function
# ------------------------------

if __name__ == "__main__":
    main()
