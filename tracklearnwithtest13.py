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
    """
    Standard Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        # logits: (B, num_classes, H, W)
        # true: (B, H, W) with class indices
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
# 3. Positional Encoding (Corrected)
# ------------------------------

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(PositionalEncoding2D, self).__init__()
        assert embed_dim % 2 == 0, "Embed dimension must be even"
        self.row_embed = nn.Parameter(torch.randn(1, embed_dim // 2, height))
        self.col_embed = nn.Parameter(torch.randn(1, embed_dim // 2, width))

    def forward(self, x):
        # x: (B, embed_dim, H, W)
        B, C, H, W = x.shape
        col_embed = self.col_embed.unsqueeze(2).expand(B, -1, H, -1)
        row_embed = self.row_embed.unsqueeze(3).expand(B, -1, -1, W)
        pos = torch.cat([col_embed, row_embed], dim=1)  # (B, embed_dim, H, W)
        return pos

# ------------------------------
# 4. Advanced Model Definition: TransUNet with Flexible ResNet Encoder
# ------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                    dim_feedforward=ff_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        # x: (S, B, embed_dim)
        return self.transformer(x)

class TransUNet(nn.Module):
    def __init__(self, num_classes=2, img_size=512, encoder_name='resnet34', pretrained=True):
        super(TransUNet, self).__init__()
        self.img_size = img_size

        # Encoder: Choose different ResNet backbones.
        encoder_fn = getattr(models, encoder_name)
        self.encoder = encoder_fn(pretrained=pretrained)
        # Adjust first conv layer for single-channel input.
        self.encoder.conv1 = nn.Conv2d(1, self.encoder.conv1.out_channels,
                                       kernel_size=self.encoder.conv1.kernel_size,
                                       stride=self.encoder.conv1.stride,
                                       padding=self.encoder.conv1.padding,
                                       bias=False)
        if pretrained:
            with torch.no_grad():
                self.encoder.conv1.weight = nn.Parameter(self.encoder.conv1.weight.mean(dim=1, keepdim=True))
        
        # Use layers up to layer4.
        self.encoder_layers = nn.Sequential(
            self.encoder.conv1,    # (B,64,H/2,W/2)
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,  # (B,64,H/4,W/4)
            self.encoder.layer1,   # (B,64,H/4,W/4)
            self.encoder.layer2,   # (B,128,H/8,W/8)
            self.encoder.layer3,   # (B,256,H/16,W/16)
            self.encoder.layer4    # (B,512,H/32,W/32)
        )
        
        # Transformer Encoder on the deepest feature map.
        self.transformer_embed_dim = 512  # Adjust if needed based on backbone.
        self.feat_h = img_size // 32
        self.feat_w = img_size // 32

        self.flatten_conv = nn.Conv2d(512, self.transformer_embed_dim, kernel_size=1)
        self.pos_encoding = PositionalEncoding2D(self.transformer_embed_dim, self.feat_h, self.feat_w)
        self.transformer_encoder = TransformerEncoder(embed_dim=self.transformer_embed_dim, num_layers=4, num_heads=8, ff_dim=2048)

        # Decoder: U-Net style upsampling with skip connections.
        # Using skip connections from encoder.layer3, encoder.layer2, and encoder.layer1.
        self.up4 = nn.ConvTranspose2d(self.transformer_embed_dim, 256, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B,1,H,W) assumed grayscale.
        x1 = self.encoder.conv1(x)       # (B,64,H/2,W/2)
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x2 = self.encoder.maxpool(x1)      # (B,64,H/4,W/4)
        x3 = self.encoder.layer1(x2)       # (B,64,H/4,W/4)
        x4 = self.encoder.layer2(x3)       # (B,128,H/8,W/8)
        x5 = self.encoder.layer3(x4)       # (B,256,H/16,W/16)
        x6 = self.encoder.layer4(x5)       # (B,512,H/32,W/32)

        # Transformer branch.
        feat = self.flatten_conv(x6)       # (B,512,H/32,W/32)
        pos = self.pos_encoding(feat)        # (B,512,H/32,W/32)
        feat = feat + pos
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1).permute(2, 0, 1)  # (S, B, C) with S = H*W
        feat_trans = self.transformer_encoder(feat_flat)
        feat_trans = feat_trans.permute(1, 2, 0).view(B, C, H, W)

        # Decoder with skip connections.
        d4 = self.up4(feat_trans)      # (B,256,H/16,W/16)
        d4 = torch.cat([d4, x5], dim=1)  # Skip connection from layer3.
        d4 = self.conv4(d4)

        d3 = self.up3(d4)              # (B,128,H/8,W/8)
        d3 = torch.cat([d3, x4], dim=1)  # Skip connection from layer2.
        d3 = self.conv3(d3)

        d2 = self.up2(d3)              # (B,64,H/4,W/4)
        d2 = torch.cat([d2, x3], dim=1)  # Skip connection from layer1.
        d2 = self.conv2(d2)

        d1 = self.up1(d2)              # (B,64,H/2,W/2)
        d1 = self.conv1(d1)
        out = self.out_conv(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

# ------------------------------
# 5. EarlyStopping Class
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

# ------------------------------
# 6. Training, Validation, and Testing Functions (Without Mixed Precision)
# ------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx):
    model.train()
    total_loss = 0.0
    print(f"--- [Train] Epoch {epoch_idx+1} ---")
    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
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
# 7. Visualization and Post-Processing
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
        image_np = image_np * 0.229 + 0.485  # Denormalize (assuming mean=0.485, std=0.229)
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
# 8. Main Function: Setup, Training, and Testing (Without Mixed Precision)
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
    save_path = "best_transunet_model2.pth"
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

    print("[Main] Starting training with advanced model: TransUNet (Without Mixed Precision)")

    # Initialize the TransUNet model with flexible ResNet encoder.
    # You can change encoder_name to 'resnet18', 'resnet34', 'resnet50', etc.
    model = TransUNet(num_classes=2, img_size=512, encoder_name='resnet34', pretrained=True).to(device)
    # Use only the Dice loss.
    criterion = DiceLoss(smooth=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

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
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
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
