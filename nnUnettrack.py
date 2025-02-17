import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchmetrics

# ------------------------------
# 0. Reproducibility
# ------------------------------

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# 1. Loss Function: Dice Loss Only
# ------------------------------

class DiceLoss(nn.Module):
    """Standard Dice Loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        # logits: (B, num_classes, H, W); true: (B, H, W)
        probs = F.softmax(logits, dim=1)
        true_one_hot = F.one_hot(true, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

# ------------------------------
# 2. Dataset and Transforms (for images and masks)
# ------------------------------

def get_augmentation_pipeline(size=(512, 512), augment=True):
    if augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.CoarseDropout(max_holes=1, max_height=int(size[1] * 0.1), max_width=int(size[0] * 0.1), p=0.2),
            A.Resize(width=size[0], height=size[1]),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(width=size[0], height=size[1]),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2()
        ])

class ResizeAndToTensorAlbumentations:
    def __init__(self, size=(512, 512), augment=False):
        self.transform = get_augmentation_pipeline(size=size, augment=augment)
    def __call__(self, image, mask):
        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 1, 1, mask_np).astype(np.uint8)
        augmented = self.transform(image=image_np, mask=mask_np)
        return augmented['image'], augmented['mask'].long()

class BinarySegDatasetAlbumentations(Dataset):
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_list = file_list
        self.transform = transform
        print(f"[Dataset] Initialized with {len(file_list)} files.")
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        img_filename = self.file_list[idx]
        base_name = os.path.splitext(img_filename)[0]
        img_path = os.path.join(self.images_dir, img_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"[Error] Image file not found: {img_path}")
        image = Image.open(img_path).convert("L")
        mask_extensions = ['.npy', '.png', '.jpg', '.jpeg']
        mask_path = None
        for ext in mask_extensions:
            potential_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.isfile(potential_path):
                mask_path = potential_path
                break
        if mask_path is None:
            raise FileNotFoundError(f"[Error] Mask file not found for image: {img_filename}")
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
            mask = Image.fromarray(mask)
        else:
            mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 1, 1, mask_np).astype(np.uint8)
        mask = Image.fromarray(mask_np)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask

# ------------------------------
# 3. Advanced Model Definition: nnU-Net (Simplified Version)
# ------------------------------

class ConvBlock(nn.Module):
    """
    A basic convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class nnUNet(nn.Module):
    """
    A simplified nnU-Net–style architecture for 2D segmentation.
    
    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale).
        out_channels (int): Number of output channels/classes.
        base_channels (int): Number of feature channels in the first level.
    """
    def __init__(self, in_channels=1, out_channels=2, base_channels=32):
        super(nnUNet, self).__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # Output layer
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        x1 = self.enc1(x)      # (B, base_channels, H, W)
        x2 = self.pool1(x1)    # (B, base_channels, H/2, W/2)
        x2 = self.enc2(x2)     # (B, base_channels*2, H/2, W/2)
        x3 = self.pool2(x2)    # (B, base_channels*2, H/4, W/4)
        x3 = self.enc3(x3)     # (B, base_channels*4, H/4, W/4)
        x4 = self.pool3(x3)    # (B, base_channels*4, H/8, W/8)
        x4 = self.enc4(x4)     # (B, base_channels*8, H/8, W/8)
        x5 = self.pool4(x4)    # (B, base_channels*8, H/16, W/16)
        
        # Bottleneck
        b = self.bottleneck(x5)  # (B, base_channels*16, H/16, W/16)
        
        # Decoder path with skip connections
        d4 = self.up4(b)               # (B, base_channels*8, H/8, W/8)
        d4 = torch.cat([d4, x4], dim=1)  # (B, base_channels*16, H/8, W/8)
        d4 = self.dec4(d4)             # (B, base_channels*8, H/8, W/8)
        
        d3 = self.up3(d4)              # (B, base_channels*4, H/4, W/4)
        d3 = torch.cat([d3, x3], dim=1) # (B, base_channels*8, H/4, W/4)
        d3 = self.dec3(d3)             # (B, base_channels*4, H/4, W/4)
        
        d2 = self.up2(d3)              # (B, base_channels*2, H/2, W/2)
        d2 = torch.cat([d2, x2], dim=1) # (B, base_channels*4, H/2, W/2)
        d2 = self.dec2(d2)             # (B, base_channels*2, H/2, W/2)
        
        d1 = self.up1(d2)              # (B, base_channels, H, W)
        d1 = torch.cat([d1, x1], dim=1) # (B, base_channels*2, H, W)
        d1 = self.dec1(d1)             # (B, base_channels, H, W)
        
        out = self.out_conv(d1)        # (B, out_channels, H, W)
        # Optionally, upsample to original resolution if needed.
        return out

# ------------------------------
# 4. EarlyStopping, Training, Validation, Testing, and Visualization Functions
# (These remain largely unchanged from your previous code.)
# ------------------------------

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
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

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx, scaler):
    model.train()
    total_loss = 0.0
    print(f"--- [Train] Epoch {epoch_idx+1} ---")
    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(loader.dataset)
    print(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(model, loader, criterion, device, epoch_idx, metrics=None):
    model.eval()
    total_loss = 0.0
    print(f"--- [Val] Epoch {epoch_idx+1} ---")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Validation", leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)
            if batch_idx % 10 == 0:
                print(f"[Val] Batch {batch_idx}, Loss: {loss.item():.4f}")
            if metrics:
                preds = torch.argmax(outputs, dim=1)
                for metric in metrics.values():
                    metric(preds, masks)
    avg_loss = total_loss / len(loader.dataset)
    print(f"--- [Val] Epoch {epoch_idx+1} Average Loss: {avg_loss:.4f} ---")
    if metrics:
        metric_values = {k: v.compute().item() for k, v in metrics.items()}
        for metric in metrics.values():
            metric.reset()
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metric_values.items()])
        print(f"--- [Val] Epoch {epoch_idx+1} Metrics: {metric_str} ---")
    return avg_loss

def test_segmentation(model, loader, device, metrics=None):
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Testing", leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            if metrics:
                for metric in metrics.values():
                    metric(preds, masks)
    if metrics:
        metric_values = {k: v.compute().item() for k, v in metrics.items()}
        for metric in metrics.values():
            metric.reset()
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metric_values.items()])
        print(f"--- [Test] Metrics: {metric_str} ---")
    else:
        print("Testing complete.")
        metric_values = {}
    return metric_values

def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# ------------------------------
# 7. Visualization and Post-Processing Functions
# ------------------------------

def thin_mask(mask, kernel_size=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)

def connected_component_postprocessing(pred, min_size=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
    new_mask = np.zeros_like(pred)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 1
    return new_mask

def visualize_predictions(model, dataset, device, num_samples=3, post_process_flag=False, apply_erosion=False):
    model.eval()
    rng = np.random.default_rng()
    indices = rng.choice(len(dataset), num_samples, replace=False)
    for idx in indices:
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_batch)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        if post_process_flag:
            pred = connected_component_postprocessing(pred)
        if apply_erosion:
            pred = thin_mask(pred, kernel_size=3, iterations=1)
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.cpu().numpy()
        # Denormalize (assuming normalization mean=0.485, std=0.229)
        image_np = image_np * 0.229 + 0.485
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 1:
            image_np = np.concatenate([image_np] * 3, axis=-1)
        overlay_image = image_np.copy()
        binary_mask = (pred == 1).astype(np.uint8)
        color = np.array([0, 255, 0], dtype=np.uint8)
        color_mask = np.zeros_like(overlay_image)
        color_mask[binary_mask == 1] = color
        overlay_image = cv2.addWeighted(overlay_image, 0.5, color_mask, 0.5, 0)
        fig, axs = plt.subplots(1, 4, figsize=(25, 5))
        axs[0].imshow(image_np)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")
        title = "Predicted Mask"
        if post_process_flag:
            title += " (Post-Processed)"
        if apply_erosion:
            title += " + Erosion"
        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title(title)
        axs[2].axis("off")
        axs[3].imshow(overlay_image)
        axs[3].set_title("Overlayed Mask on Image")
        axs[3].axis("off")
        plt.tight_layout()
        plt.show()

# ------------------------------
# 8. Main Function: Setup, Training, and Testing (Using nnU-Net for Segmentation)
# ------------------------------

def main():
    # Update these paths accordingly.
    images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/images/"
    masks_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/masks/"
    test_images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/images"
    test_masks_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/masks"

    batch_size = 16
    num_epochs = 150
    learning_rate = 1e-4
    save_path = "best_nnunet_model2.pth"
    patience = 10
    post_process_flag = False
    apply_erosion = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # Labeled training data.
    all_train_images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"[Main] Found {len(all_train_images)} training images in {images_dir}")
    if len(all_train_images) == 0:
        print("[Error] No training images found.")
        return
    all_train_images = sorted([f for f in all_train_images if any(os.path.isfile(os.path.join(masks_dir, os.path.splitext(f)[0] + ext))
                                                                    for ext in ['.npy', '.png', '.jpg', '.jpeg'])])
    print(f"[Main] {len(all_train_images)} training images have corresponding masks.")
    if len(all_train_images) == 0:
        print("[Error] No training masks found or mismatched filenames.")
        return

    # Labeled test data.
    all_test_images = sorted([f for f in os.listdir(test_images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"[Main] Found {len(all_test_images)} test images in {test_images_dir}")
    if len(all_test_images) == 0:
        print("[Error] No test images found.")
        return
    all_test_images = sorted([f for f in all_test_images if any(os.path.isfile(os.path.join(test_masks_dir, os.path.splitext(f)[0] + ext))
                                                                for ext in ['.npy', '.png', '.jpg', '.jpeg'])])
    print(f"[Main] {len(all_test_images)} test images have corresponding masks.")
    if len(all_test_images) == 0:
        print("[Error] No test masks found or mismatched filenames.")
        return

    train_files = all_train_images
    print(f"[Main] Training samples: {len(train_files)}; Validation samples: {len(all_test_images)}")

    train_transform = ResizeAndToTensorAlbumentations(size=(512, 512), augment=True)
    test_transform = ResizeAndToTensorAlbumentations(size=(512, 512), augment=False)

    train_dataset = BinarySegDatasetAlbumentations(images_dir, masks_dir, train_files, transform=train_transform)
    val_dataset = BinarySegDatasetAlbumentations(test_images_dir, test_masks_dir, all_test_images, transform=test_transform)
    test_dataset = BinarySegDatasetAlbumentations(test_images_dir, test_masks_dir, all_test_images, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("[Main] Starting training with advanced model: nnU-Net (Simplified Version for Catheter Segmentation)")

    # Initialize the nnU-Net model.
    model = nnUNet(in_channels=1, out_channels=2, base_channels=8).to(device)
    # Use Dice loss only.
    criterion = DiceLoss(smooth=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    scaler = torch.amp.GradScaler('cuda')

    # Define evaluation metrics using torchmetrics.
    metrics_val = {
        'dice': torchmetrics.Dice(num_classes=2, average='macro').to(device),
        'iou': torchmetrics.JaccardIndex(task='multiclass', num_classes=2).to(device),
        'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device),
        'precision': torchmetrics.Precision(task='multiclass', num_classes=2, average='macro').to(device),
        'recall': torchmetrics.Recall(task='multiclass', num_classes=2, average='macro').to(device),
        'f1': torchmetrics.F1Score(task='multiclass', num_classes=2, average='macro').to(device)
    }
    metrics_test = {
        'dice': torchmetrics.Dice(num_classes=2, average='macro').to(device),
        'iou': torchmetrics.JaccardIndex(task='multiclass', num_classes=2).to(device),
        'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device),
        'precision': torchmetrics.Precision(task='multiclass', num_classes=2, average='macro').to(device),
        'recall': torchmetrics.Recall(task='multiclass', num_classes=2, average='macro').to(device),
        'f1': torchmetrics.F1Score(task='multiclass', num_classes=2, average='macro').to(device)
    }

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        train_losses.append(train_loss)
        val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, metrics_val)
        val_losses.append(val_loss)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f">> Saved best model with Val Loss: {best_val_loss:.4f}")
            early_stopping.counter = 0
        else:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("[Main] Early stopping triggered. Stopping training.")
                break

    print(f"\n[Main] Training complete. Best Validation Loss: {best_val_loss:.4f}")
    print(f"[Main] Best model saved at: {save_path}")

    print("\n>>> Loading best model for testing...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_metrics_results = test_segmentation(model, test_loader, device, metrics=metrics_test)
    print(f"Test Metrics: {test_metrics_results}")

    if len(test_dataset) > 0:
        print("\n>>> Visualizing predictions on test samples...")
        visualize_predictions(model, test_dataset, device, num_samples=10, post_process_flag=post_process_flag, apply_erosion=apply_erosion)
    else:
        print("[Warning] No test samples available for visualization.")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
