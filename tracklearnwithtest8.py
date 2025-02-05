import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.models as models  # For ResNet encoder

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchmetrics
import random
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize
from collections import Counter, deque
import torch.nn.functional as F

# ------------------------------
# 0. Reproducibility
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
# 1. Loss Functions
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

class ClDiceLoss(nn.Module):
    """
    clDice Loss for binary segmentation.
    Reference: https://arxiv.org/abs/1811.00888
    """
    def __init__(self, smooth=1e-6):
        super(ClDiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, true):
        """
        Args:
            logits (torch.Tensor): Predicted logits with shape [B, C, H, W].
            true (torch.Tensor): Ground truth masks with shape [B, H, W].
        
        Returns:
            torch.Tensor: clDice loss.
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        
        # Assuming class 1 is catheter; convert predictions to binary masks
        pred = probs[:,1,:,:]  # [B, H, W]
        # Detach tensors before converting to NumPy
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        
        clDice_scores = []
        for i in range(pred.shape[0]):
            pred_bin = (pred[i] > 0.5).astype(np.uint8)
            true_bin = (true[i] > 0).astype(np.uint8)
            
            # Skeletonize
            pred_skel = skeletonize(pred_bin).astype(np.uint8)
            true_skel = skeletonize(true_bin).astype(np.uint8)
            
            # Compute intersection
            intersection = np.logical_and(pred_skel, true_skel).sum()
            
            # Compute clDice
            pred_skel_sum = pred_skel.sum()
            true_skel_sum = true_skel.sum()
            clDice = (2. * intersection + self.smooth) / (pred_skel_sum + true_skel_sum + self.smooth)
            clDice_scores.append(clDice)
        
        # Compute average clDice
        avg_clDice = np.mean(clDice_scores)
        clDice_loss = 1 - avg_clDice
        # Return as a tensor (Note: This tensor does not carry gradients)
        return torch.tensor(clDice_loss, requires_grad=True).to(logits.device)

class FocalLoss(nn.Module):
    """
    Focal Loss to focus training on hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
      
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedClDiceDiceFocalLoss(nn.Module):
    """
    Combined clDice, Dice, and Focal Loss with configurable weights.
    """
    def __init__(self, dice_weight=0.1, cldice_weight=0.7, focal_weight=0.2, smooth=1e-6):
        super(CombinedClDiceDiceFocalLoss, self).__init__()
        self.cldice = ClDiceLoss(smooth)
        self.dice = DiceLoss(smooth)
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.focal_weight = focal_weight
    
    def forward(self, logits, true):
        cldice_loss = self.cldice(logits, true)
        dice_loss = self.dice(logits, true)
        focal_loss = self.focal(logits, true)
        combined_loss = (self.cldice_weight * cldice_loss +
                         self.dice_weight * dice_loss +
                         self.focal_weight * focal_loss)
        return combined_loss

# ------------------------------
# 2. Dataset and Transforms
# ------------------------------

class ResizeAndToTensorAlbumentations:
    """
    Custom transform using Albumentations to resize images and masks, apply augmentations, and convert them to tensors.
    """
    def __init__(self, size=(512, 512), augment=False):
        self.size = size  # (width, height)
        self.augment = augment
        
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                A.GridDistortion(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.CoarseDropout(max_holes=1, max_height=int(self.size[1]*0.1), max_width=int(self.size[0]*0.1), p=0.2),
                A.Resize(width=self.size[0], height=self.size[1]),
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(width=self.size[0], height=self.size[1]),
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2()
            ])

    def __call__(self, image, mask):
        """
        Args:
            image (PIL Image): Input image.
            mask (np.array or PIL Image): Corresponding mask as a NumPy array or PIL Image.

        Returns:
            image_tensor (torch.Tensor): Resized and normalized image tensor.
            mask_tensor (torch.Tensor): Resized mask tensor (Long type).
        """
        # Convert PIL Images to NumPy arrays
        image_np = np.array(image)
        if isinstance(mask, np.ndarray):
            mask_np = mask
        else:
            mask_np = np.array(mask)

        # Ensure mask is binary
        mask_np = np.where(mask_np > 1, 1, mask_np).astype(np.uint8)

        # Apply Albumentations transformations
        augmented = self.transform(image=image_np, mask=mask_np)
        image_aug = augmented['image']  # Tensor [C, H, W]
        mask_aug = augmented['mask']    # Tensor [H, W]

        return image_aug, mask_aug.long()

class BinarySegDatasetAlbumentations(Dataset):
    """
    Custom Dataset for binary image segmentation using Albumentations.
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

        print(f"[Dataset] Initialized with {len(file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_filename = self.file_list[idx]
        base_name = os.path.splitext(img_filename)[0]

        # Load image
        img_path = os.path.join(self.images_dir, img_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"[Error] Image file not found: {img_path}")
        image = Image.open(img_path).convert("L")  # Grayscale

        # Load mask
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
# 3. Advanced Model Definition (SegViT with Attention)
# ------------------------------

class TransformerBridge(nn.Module):
    def __init__(self, in_channels, embed_dim=768, num_heads=12, num_layers=12, ff_dim=3072, dropout=0.1):
        """
        Transformer bridge module.
        """
        super(TransformerBridge, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.InstanceNorm2d(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)  # [B, embed_dim, H, W]
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, embed_dim]
        x = self.transformer_encoder(x)     
        x = x.permute(1, 2, 0).view(B, -1, H, W)  # [B, embed_dim, H, W]
        x = self.norm(x)
        return x

class DoubleConv(nn.Module):
    """
    A block with two convolutional layers each followed by BatchNorm and ReLU.
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

class AttentionGate(nn.Module):
    """
    Attention gate module to refine skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in the gating signal (from decoder).
            F_l: Number of channels in the skip connection (from encoder).
            F_int: Intermediate number of channels.
        """
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
        """
        Args:
            g: Gating signal (decoder feature).
            x: Skip connection feature (encoder feature).
        Returns:
            Refined feature map.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SegViT(nn.Module):
    """
    SegViT architecture integrating a ResNet encoder with a Transformer bridge and U-Net-like decoder enhanced with attention gates.
    """
    def __init__(self, encoder_name='resnet50', pretrained=True, num_classes=2):
        """
        Args:
            encoder_name (str): Name of the encoder.
            pretrained (bool): Use pretrained weights.
            num_classes (int): Number of segmentation classes.
        """
        super(SegViT, self).__init__()
        
        # Initialize encoder and get channel sizes
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
        else:
            raise ValueError("Unsupported encoder. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101'.")
        
        # Modify first conv layer for single-channel input
        self.encoder.conv1 = nn.Conv2d(
            1, 
            self.encoder.conv1.out_channels, 
            kernel_size=self.encoder.conv1.kernel_size, 
            stride=self.encoder.conv1.stride, 
            padding=self.encoder.conv1.padding, 
            bias=self.encoder.conv1.bias is not None
        )
        if pretrained:
            with torch.no_grad():
                self.encoder.conv1.weight = nn.Parameter(
                    self.encoder.conv1.weight.mean(dim=1, keepdim=True)
                )
        
        # Remove FC and pooling layers; define encoder layers
        self.encoder_layers = nn.Sequential(
            self.encoder.conv1,   # [B, 64, H/2, W/2]
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool, # [B, 64, H/4, W/4]
            self.encoder.layer1,  # [B, 64, H/4, W/4]
            self.encoder.layer2,  # [B, 128, H/8, W/8]
            self.encoder.layer3,  # [B, 256, H/16, W/16]
            self.encoder.layer4   # [B, 512 or 2048, H/32, W/32]
        )
        
        # Transformer Bridge
        self.transformer = TransformerBridge(
            in_channels=encoder_channels[-1],
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ff_dim=3072,
            dropout=0.1
        )
        
        # Define decoder channels based on encoder type
        if encoder_name in ['resnet50', 'resnet101']:
            decoder_channels = [512, 256, 128, 64]
        else:
            decoder_channels = [256, 128, 64, 64]
        
        # Decoder layers with attention gates
        self.up1 = nn.ConvTranspose2d(768, decoder_channels[0], kernel_size=2, stride=2)
        self.attn1 = AttentionGate(F_g=decoder_channels[0], F_l=encoder_channels[3], F_int=encoder_channels[3]//2)
        self.conv1 = DoubleConv(decoder_channels[0] + encoder_channels[3], decoder_channels[0])
        
        self.up2 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2)
        self.attn2 = AttentionGate(F_g=decoder_channels[1], F_l=encoder_channels[2], F_int=encoder_channels[2]//2)
        self.conv2 = DoubleConv(decoder_channels[1] + encoder_channels[2], decoder_channels[1])
        
        self.up3 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        self.attn3 = AttentionGate(F_g=decoder_channels[2], F_l=encoder_channels[1], F_int=encoder_channels[1]//2)
        self.conv3 = DoubleConv(decoder_channels[2] + encoder_channels[1], decoder_channels[2])
        
        self.up4 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        self.attn4 = AttentionGate(F_g=decoder_channels[3], F_l=encoder_channels[0], F_int=encoder_channels[0]//2)
        self.conv4 = DoubleConv(decoder_channels[3] + encoder_channels[0], decoder_channels[3])
        
        # Final segmentation head
        self.out_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        skip_features = []
        out = x
        for i, layer in enumerate(self.encoder_layers):
            out = layer(out)
            if i == 0:
                skip_features.append(out)  # conv1
            elif i == 4:
                skip_features.append(out)  # layer1
            elif i == 5:
                skip_features.append(out)  # layer2
            elif i == 6:
                skip_features.append(out)  # layer3
        # out is from layer4
        
        # Transformer Bridge
        transformed = self.transformer(out)  # [B, 768, H/32, W/32]
        
        # Decoder with attention on skip connections
        up1 = self.up1(transformed)    # [B, decoder_channels[0], H/16, W/16]
        skip1 = skip_features[3]       # from layer3
        skip1_att = self.attn1(g=up1, x=skip1)
        conv1 = self.conv1(torch.cat([up1, skip1_att], dim=1))
        
        up2 = self.up2(conv1)          # [B, decoder_channels[1], H/8, W/8]
        skip2 = skip_features[2]       # from layer2
        skip2_att = self.attn2(g=up2, x=skip2)
        conv2 = self.conv2(torch.cat([up2, skip2_att], dim=1))
        
        up3 = self.up3(conv2)          # [B, decoder_channels[2], H/4, W/4]
        skip3 = skip_features[1]       # from layer1
        skip3_att = self.attn3(g=up3, x=skip3)
        conv3 = self.conv3(torch.cat([up3, skip3_att], dim=1))
        
        up4 = self.up4(conv3)          # [B, decoder_channels[3], H/2, W/2]
        skip4 = skip_features[0]       # from conv1 output
        skip4_att = self.attn4(g=up4, x=skip4)
        conv4 = self.conv4(torch.cat([up4, skip4_att], dim=1))
        
        out_seg = self.out_conv(conv4) # [B, num_classes, H/2, W/2]
        # Upsample to original image size
        out_seg = F.interpolate(out_seg, scale_factor=2, mode='bilinear', align_corners=True)
        return out_seg

# ------------------------------
# 4. Early Stopping Class
# ------------------------------

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve.
    """
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
# 5. Training and Validation Functions
# ------------------------------

def calculate_class_weights(dataloader, num_classes=2, device='cuda'):
    """
    Calculates class weights based on frequency in the dataset.
    """
    class_counts = Counter()
    for _, masks in tqdm(dataloader, desc="Calculating class weights"):
        masks = masks.view(-1).cpu().numpy()
        class_counts.update(masks)
    
    total = sum(class_counts.values())
    class_weights = []
    for cls in range(num_classes):
        count = class_counts.get(cls, 0)
        weight = total / (num_classes * (count + 1e-6))
        class_weights.append(weight)
    
    return torch.tensor(class_weights, dtype=torch.float32).to(device)

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx):
    model.train()
    total_loss = 0.0
    print(f"--- [Train] Starting epoch {epoch_idx+1} ---")
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
            print(f"[Train] Batch {batch_idx}, Loss: {loss.item():.4f}")
    average_loss = total_loss / len(loader.dataset)
    print(f"--- [Train] Epoch {epoch_idx+1} Average Loss: {average_loss:.4f} ---")
    return average_loss

def validate_one_epoch(model, loader, criterion, device, epoch_idx, metrics=None):
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
            
            if metrics:
                preds = torch.argmax(outputs, dim=1)
                for metric in metrics.values():
                    metric(preds, masks)

    average_loss = total_loss / len(loader.dataset)
    print(f"--- [Val] Epoch {epoch_idx+1} Average Loss: {average_loss:.4f} ---")
    
    if metrics:
        metric_values = {k: v.compute().item() for k, v in metrics.items()}
        for metric in metrics.values():
            metric.reset()
    else:
        metric_values = {}
    
    if metrics:
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metric_values.items()])
        print(f"--- [Val] Epoch {epoch_idx+1} Metrics: {metric_str} ---")
    
    return average_loss

# ------------------------------
# 6. Testing and Metrics
# ------------------------------

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
    else:
        dice_scores = []
        for pred, mask in zip(preds, masks):
            dice = dice_coefficient(pred, mask)
            dice_scores.append(dice)
        metric_values = {
            'dice': np.mean(dice_scores) if dice_scores else 0
        }
    if metrics:
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metric_values.items()])
        print(f"--- [Test] Metrics: {metric_str} ---")
    else:
        print(f"--- [Test] Dice Coefficient: {metric_values['dice']:.4f} ---")
    return metric_values

def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()

# ------------------------------
# 7. Visualization and Post-Processing
# ------------------------------

def connected_component_postprocessing(pred, min_size=100):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=8)
    new_mask = np.zeros_like(pred)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 1
    return new_mask

class TemporalSmoothing:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.queue = deque(maxlen=window_size)

    def smooth(self, pred):
        self.queue.append(pred)
        if len(self.queue) == 0:
            return pred
        sum_pred = np.sum(self.queue, axis=0)
        avg_pred = (sum_pred / len(self.queue)) > 0.5
        return avg_pred.astype(np.uint8)

def visualize_predictions(model, dataset, device, num_samples=3, post_process_flag=False):
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

        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.cpu().numpy()
        image_np = image_np * 0.485 + 0.229  # Unnormalize assuming ImageNet stats
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        
        if image_np.ndim == 2:
            image_np = np.stack([image_np]*3, axis=-1)
        elif image_np.shape[2] == 1:
            image_np = np.concatenate([image_np]*3, axis=-1)
        
        print(f"Sample Index: {idx}")
        print(f"Image Shape: {image_np.shape}")
        print(f"Mask Shape: {mask_np.shape}")
        
        overlay_image = image_np.copy()
        binary_mask = (pred == 1).astype(np.uint8)
        if binary_mask.ndim == 3 and binary_mask.shape[2] == 1:
            binary_mask = binary_mask.squeeze(2)
        color = np.array([0, 255, 0], dtype=np.uint8)
        color_mask = np.zeros_like(overlay_image)
        color_mask[binary_mask == 1] = color
        alpha = 0.5
        overlay_image = cv2.addWeighted(overlay_image, 1 - alpha, color_mask, alpha, 0)

        fig, axs = plt.subplots(1, 4, figsize=(25, 5))
        axs[0].imshow(image_np)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")
        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title("Predicted Mask" + (" (Post-Processed)" if post_process_flag else ""))
        axs[2].axis("off")
        axs[3].imshow(overlay_image)
        axs[3].set_title("Overlayed Mask on Image")
        axs[3].axis("off")
        plt.tight_layout()
        plt.show()

# ------------------------------
# 8. Main Function
# ------------------------------

def main():
    # Update these directories as needed.
    images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/images/"
    masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_train/masks/"
    test_images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/images/"
    test_masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_test/masks/"

    batch_size    = 16
    num_epochs    = 200
    learning_rate = 1e-4
    val_split     = 0.2
    save_path     = "best_segvit_model9.pth"
    patience      = 10
    post_process_flag = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # Collect training file lists
    all_train_images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_train_images)} training image files in {images_dir}")

    if len(all_train_images) == 0:
        print("[Error] No training image files found. Check your training path!")
        return

    all_train_images = sorted([
        f for f in all_train_images
        if any(
            os.path.isfile(os.path.join(masks_dir, os.path.splitext(f)[0] + ext))
            for ext in ['.npy', '.png', '.jpg', '.jpeg']
        )
    ])
    print(f"[Main] {len(all_train_images)} training images have corresponding masks.")

    if len(all_train_images) == 0:
        print("[Error] No training mask files found or mismatched filenames. Check your training mask path!")
        return

    all_test_images = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_test_images)} test image files in {test_images_dir}")

    if len(all_test_images) == 0:
        print("[Error] No test image files found. Check your test path!")
        return

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

    # Train/Validation split
    train_files, val_files = train_test_split(
        all_train_images,
        test_size=val_split,
        random_state=42
    )
    print(f"[Main] Training samples: {len(train_files)}")
    print(f"[Main] Validation samples: {len(val_files)}")

    # Create Datasets
    train_transform = ResizeAndToTensorAlbumentations(size=(512, 512), augment=True)
    val_transform = ResizeAndToTensorAlbumentations(size=(512, 512), augment=False)
    test_transform = ResizeAndToTensorAlbumentations(size=(512, 512), augment=False)

    train_dataset = BinarySegDatasetAlbumentations(
        images_dir=images_dir,
        masks_dir=masks_dir,
        file_list=train_files,
        transform=train_transform
    )
    val_dataset = BinarySegDatasetAlbumentations(
        images_dir=images_dir,
        masks_dir=masks_dir,
        file_list=val_files,
        transform=val_transform
    )
    test_dataset = BinarySegDatasetAlbumentations(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        file_list=all_test_images,
        transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    print("[Main] Calculating class weights...")
    class_weights = calculate_class_weights(train_loader, num_classes=2, device=device)
    print(f"[Main] Class Weights: {class_weights}")

    # Initialize Model, Loss, Optimizer, Scheduler
    model = SegViT(encoder_name='resnet50', pretrained=True, num_classes=2).to(device)
    criterion = CombinedClDiceDiceFocalLoss(dice_weight=0.1, cldice_weight=0.7, focal_weight=0.2, smooth=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

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

        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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

    test_metrics = test_segmentation(model, test_loader, device, metrics=metrics_test)
    print(f"Test Metrics: {test_metrics}")

    if len(test_dataset) > 0:
        print("\n>>> Visualizing predictions on test samples...")
        visualize_predictions(model, test_dataset, device, num_samples=10, post_process_flag=post_process_flag)
    else:
        print("[Warning] No test samples available for visualization.")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

# ------------------------------
# 9. Run the Main Function
# ------------------------------

if __name__ == "__main__":
    main()
