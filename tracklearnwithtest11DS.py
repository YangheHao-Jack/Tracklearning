import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import itertools

# Import torchmetrics for evaluation
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
# 1. Loss Functions: Dice, Shape-Sensitive, and Combined Loss
# ------------------------------

class DiceLoss(nn.Module):
    """
    Standard Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        probs = F.softmax(logits, dim=1)
        true_one_hot = F.one_hot(true, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class ViTFeatureExtractor(nn.Module):
    """
    Uses a pretrained ViT (vit_base_patch16_384) to extract features.
    """
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_384', pretrained=True)
        self.vit.reset_classifier(0)
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.vit(x)
        return features

def compute_SDM(mask_np):
    dt_out = distance_transform_edt(1 - mask_np)
    dt_in = distance_transform_edt(mask_np)
    sdm = dt_out - dt_in
    return sdm

def compute_SDM_batch(mask_tensor):
    mask_np = mask_tensor.cpu().numpy()
    sdm_list = []
    for i in range(mask_np.shape[0]):
        sdm = compute_SDM(mask_np[i,0])
        sdm_list.append(sdm)
    sdm_np = np.stack(sdm_list, axis=0)
    sdm_tensor = torch.tensor(sdm_np, dtype=torch.float32, device=mask_tensor.device).unsqueeze(1)
    return sdm_tensor

class ShapeSensitiveLoss(nn.Module):
    """
    Computes shape-sensitive loss as 1 - cosine similarity between features
    extracted from the SDMs of the predicted and ground-truth masks.
    """
    def __init__(self, smooth=1e-6):
        super(ShapeSensitiveLoss, self).__init__()
        self.smooth = smooth
        self.vit_extractor = ViTFeatureExtractor().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for param in self.vit_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred_logits, true):
        pred_probs = F.softmax(pred_logits, dim=1)[:, 1:2, :, :]  # (B,1,H,W)
        pred_mask = (pred_probs > 0.5).float()
        true_mask = true.unsqueeze(1).float()  # (B,1,H,W)
        pred_SDM = compute_SDM_batch(pred_mask)
        true_SDM = compute_SDM_batch(true_mask)
        # Resize SDMs to the ViT expected input resolution (384x384)
        pred_SDM_resized = F.interpolate(pred_SDM, size=(384,384), mode='bilinear', align_corners=False)
        true_SDM_resized = F.interpolate(true_SDM, size=(384,384), mode='bilinear', align_corners=False)
        features_pred = self.vit_extractor(pred_SDM_resized)
        features_true = self.vit_extractor(true_SDM_resized)
        cos_sim = F.cosine_similarity(features_pred, features_true, dim=1)
        loss = 1 - cos_sim.mean()
        return loss

class CombinedLoss(nn.Module):
    """
    Total loss is a combination of Dice Loss and Shape-Sensitive Loss.
    """
    def __init__(self, dice_weight=0.5, shape_weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.shape_loss = ShapeSensitiveLoss(smooth)
        self.dice_weight = dice_weight
        self.shape_weight = shape_weight

    def forward(self, logits, true):
        loss_dice = self.dice_loss(logits, true)
        loss_shape = self.shape_loss(logits, true)
        return self.dice_weight * loss_dice + self.shape_weight * loss_shape

def entropy_loss_fn(logits):
    """
    Unsupervised entropy loss for domain adaptation.
    Encourages confident predictions.
    """
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()

# ------------------------------
# 2. Dataset and Transforms
# ------------------------------

def get_augmentation_pipeline(size=(512,512), augment=True):
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
            A.CoarseDropout(max_holes=1, max_height=int(size[1]*0.1), max_width=int(size[0]*0.1), p=0.2),
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
    def __init__(self, size=(512,512), augment=False):
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
# 3. Video Frame Dataset for Domain Adaptation
# ------------------------------

class VideoFrameDataset(Dataset):
    """
    Extracts frames from a video file for unsupervised domain adaptation.
    Loads all frames into memory.
    """
    def __init__(self, video_path, transform=None, every_n_frame=1):
        self.transform = transform
        self.frames = []
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % every_n_frame == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frames.append(Image.fromarray(frame))
            count += 1
        cap.release()
        print(f"[VideoFrameDataset] Loaded {len(self.frames)} frames from {video_path}")
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        image = self.frames[idx]
        if self.transform is not None:
            image = self.transform(image)['image']  # image-only transform
        return image

class ResizeAndToTensorAlbumentationsImageOnly:
    def __init__(self, size=(512,512), augment=False):
        self.transform = A.Compose([
            A.Resize(width=size[0], height=size[1]),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2()
        ])
    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        return augmented

# ------------------------------
# 4. SegViT Model Definition (ResNet Encoder + Transformer Bridge + U-Net Decoder)
# ------------------------------

class TransformerBridge(nn.Module):
    def __init__(self, in_channels, embed_dim=768, num_heads=12, num_layers=12, ff_dim=3072, dropout=0.1):
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
        x = self.conv(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0).view(B, -1, H, W)
        x = self.norm(x)
        return x

class DoubleConv(nn.Module):
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

class SegViT(nn.Module):
    def __init__(self, encoder_name='resnet34', pretrained=True, num_classes=2):
        super(SegViT, self).__init__()
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
            raise ValueError("Unsupported encoder.")

        self.encoder.conv1 = nn.Conv2d(1, self.encoder.conv1.out_channels,
                                       kernel_size=self.encoder.conv1.kernel_size,
                                       stride=self.encoder.conv1.stride,
                                       padding=self.encoder.conv1.padding,
                                       bias=self.encoder.conv1.bias is not None)
        if pretrained:
            with torch.no_grad():
                self.encoder.conv1.weight = nn.Parameter(
                    self.encoder.conv1.weight.mean(dim=1, keepdim=True)
                )
        self.encoder_layers = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        )
        self.transformer = TransformerBridge(
            in_channels=encoder_channels[-1],
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ff_dim=3072,
            dropout=0.1
        )
        if encoder_name in ['resnet50', 'resnet101']:
            decoder_channels = [512, 256, 128, 64]
        else:
            decoder_channels = [256, 128, 64, 64]
        self.decoder = nn.ModuleDict({
            'up1': nn.ConvTranspose2d(768, decoder_channels[0], kernel_size=2, stride=2),
            'conv1': DoubleConv(decoder_channels[0] + encoder_channels[3], decoder_channels[0]),
            'up2': nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2),
            'conv2': DoubleConv(decoder_channels[1] + encoder_channels[2], decoder_channels[1]),
            'up3': nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2),
            'conv3': DoubleConv(decoder_channels[2] + encoder_channels[1], decoder_channels[2]),
            'up4': nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2),
            'conv4': DoubleConv(decoder_channels[3] + encoder_channels[0], decoder_channels[3])
        })
        self.out_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, x):
        skip_features = []
        out = x
        for i, layer in enumerate(self.encoder_layers):
            out = layer(out)
            if i == 0:
                skip_features.append(out)
            elif i == 4:
                skip_features.append(out)
            elif i == 5:
                skip_features.append(out)
            elif i == 6:
                skip_features.append(out)
        transformed = self.transformer(out)
        up1 = self.decoder['up1'](transformed)
        skip1 = skip_features[3]
        conv1 = self.decoder['conv1'](torch.cat([up1, skip1], dim=1))
        up2 = self.decoder['up2'](conv1)
        skip2 = skip_features[2]
        conv2 = self.decoder['conv2'](torch.cat([up2, skip2], dim=1))
        up3 = self.decoder['up3'](conv2)
        skip3 = skip_features[1]
        conv3 = self.decoder['conv3'](torch.cat([up3, skip3], dim=1))
        up4 = self.decoder['up4'](conv3)
        skip4 = skip_features[0]
        conv4 = self.decoder['conv4'](torch.cat([up4, skip4], dim=1))
        out = self.out_conv(conv4)
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
# 6. Training and Evaluation Functions (with Mixed Precision and Domain Adaptation)
# ------------------------------

def train_one_epoch(model, loader, video_loader, optimizer, criterion, device, epoch_idx, scaler, unsup_weight=0.1):
    model.train()
    total_loss = 0.0
    print(f"--- [Train] Epoch {epoch_idx+1} ---")
    # Create an infinite iterator for video frames.
    video_iter = itertools.cycle(video_loader)
    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss_sup = criterion(outputs, masks)
            # Domain adaptation: get a batch of unlabeled video frames.
            video_images = next(video_iter)
            video_images = video_images.to(device)
            video_outputs = model(video_images)
            # Unsupervised entropy loss: encourage confident predictions.
            video_probs = F.softmax(video_outputs, dim=1)
            unsup_loss = -torch.mean(torch.sum(video_probs * torch.log(video_probs + 1e-8), dim=1))
            loss = loss_sup + unsup_weight * unsup_loss
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
    else:
        metric_values = {}
    if metrics:
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
    else:
        dice_scores = []
        for pred, mask in zip(preds, masks):
            dice = dice_coefficient(pred, mask)
            dice_scores.append(dice)
        metric_values = {'dice': np.mean(dice_scores) if dice_scores else 0}
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
        image_np = image.permute(1,2,0).cpu().numpy()
        mask_np = mask.cpu().numpy()
        image_np = image_np * 0.485 + 0.229
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        if image_np.ndim == 2:
            image_np = np.stack([image_np]*3, axis=-1)
        elif image_np.shape[2] == 1:
            image_np = np.concatenate([image_np]*3, axis=-1)
        overlay_image = image_np.copy()
        binary_mask = (pred == 1).astype(np.uint8)
        color = np.array([0,255,0], dtype=np.uint8)
        color_mask = np.zeros_like(overlay_image)
        color_mask[binary_mask == 1] = color
        overlay_image = cv2.addWeighted(overlay_image, 0.5, color_mask, 0.5, 0)
        fig, axs = plt.subplots(1,4, figsize=(25,5))
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
# 8. EarlyStopping Class
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
# 9. Main Function
# ------------------------------

def main():
    # Update these paths accordingly.
    images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/images/"
    masks_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/masks/"
    test_images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/images/"
    test_masks_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/masks/"
    video_path = "/home/yanghehao/tracklearning/DATA/Tom.mp4"  # Video for domain adaptation

    batch_size = 16
    num_epochs = 150
    learning_rate = 1e-4
    val_split = 0.2
    save_path = "best_segvit_model_Domain adaptation.pth"
    patience = 10
    unsup_weight = 0.1  # Weight for unsupervised (entropy) loss on video frames
    post_process_flag = False
    apply_erosion = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # Labeled source domain.
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

    # Test domain.
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

    train_files= all_train_images
    print(f"[Main] Training samples: {len(train_files)}; Validation samples: {len(all_test_images)}")

    train_transform = ResizeAndToTensorAlbumentations(size=(512,512), augment=True)
    val_transform = ResizeAndToTensorAlbumentations(size=(512,512), augment=False)
    test_transform = ResizeAndToTensorAlbumentations(size=(512,512), augment=False)

    train_dataset = BinarySegDatasetAlbumentations(images_dir, masks_dir, train_files, transform=train_transform)
    val_dataset = BinarySegDatasetAlbumentations(test_images_dir, test_masks_dir, all_test_images, transform=test_transform)
    test_dataset = BinarySegDatasetAlbumentations(test_images_dir, test_masks_dir, all_test_images, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Unlabeled target domain: video frames for domain adaptation.
    video_transform = ResizeAndToTensorAlbumentationsImageOnly(size=(512,512), augment=False)
    video_dataset = VideoFrameDataset(video_path, transform=video_transform, every_n_frame=5)
    video_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("[Main] Class weights not used for shape-sensitive loss.")

    # Model: Use SegViT (ResNet encoder + Transformer Bridge + U-Net decoder).
    model = SegViT(encoder_name='resnet34', pretrained=True, num_classes=2).to(device)
    # Supervised loss: Combined Dice + Shape-Sensitive Loss.
    criterion = CombinedLoss(dice_weight=0.5, shape_weight=0.5, smooth=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Set up AMP scaler.
    scaler = torch.amp.GradScaler(device='cuda')

    # Define metrics using torchmetrics.
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
        train_loss = train_one_epoch(model, train_loader, video_loader, optimizer, criterion, device, epoch, scaler, unsup_weight)
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

    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
