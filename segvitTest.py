#!/usr/bin/env python3
# test.py

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.models as models  # For ResNet encoder

from tqdm import tqdm
import torchmetrics
import matplotlib.pyplot as plt
import cv2
import random
from PIL import Image

##################################
# 1. Data Transforms & Dataset
##################################

class ResizeAndToTensor:
    """
    Custom transform to resize images and masks to a fixed size and convert them to tensors.
    """
    def __init__(self, size=(512, 512)):
        self.size = size  # (height, width)
        # Example normalization for single-channel
        self.normalize = transforms.Normalize(mean=[0.485],
                                              std=[0.229])

    def __call__(self, image, mask):
        # Resize image
        image = TF.resize(
            image,
            self.size,
            interpolation=InterpolationMode.BILINEAR
        )
        # Resize mask
        mask = TF.resize(
            mask,
            self.size,
            interpolation=InterpolationMode.NEAREST
        )
        # Convert to tensor
        image_tensor = TF.to_tensor(image)  # [1, H, W]
        image_tensor = self.normalize(image_tensor)
        mask_tensor  = torch.from_numpy(np.array(mask)).long()  # [H, W]
        return image_tensor, mask_tensor

class BinarySegDataset(Dataset):
    """
    Custom Dataset for binary image segmentation.
    """
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.file_list  = file_list
        self.transform  = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_filename = self.file_list[idx]
        base_name = os.path.splitext(img_filename)[0]

        # Load image
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert("L")  # grayscale

        # Find mask
        mask_extensions = ['.npy', '.png', '.jpg', '.jpeg']
        mask_path = None
        for ext in mask_extensions:
            potential_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.isfile(potential_path):
                mask_path = potential_path
                break
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {img_filename}")

        # Load mask
        if mask_path.endswith('.npy'):
            mask_data = np.load(mask_path)
            mask = Image.fromarray(mask_data)
        else:
            mask = Image.open(mask_path).convert("L")

        # Convert any label >1 into 1 (binary segmentation)
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 1, 1, mask_np).astype(np.uint8)
        mask = Image.fromarray(mask_np)

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

##################################
# 2. SegViT Model
##################################

class DiceLoss(nn.Module):
    """ Dice Loss for binary segmentation. """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        probs = F.softmax(logits, dim=1)
        true_one_hot = F.one_hot(true, num_classes=probs.shape[1]).permute(0,3,1,2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """ Combined CrossEntropy + Dice Loss. """
    def __init__(self, weight=None, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(smooth)

    def forward(self, logits, true):
        ce_loss = self.ce(logits, true)
        dice_loss = self.dice(logits, true)
        return ce_loss + dice_loss

class TransformerBridge(nn.Module):
    """ Transformer module bridging the encoder and decoder. """
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
        x = self.conv(x)              # [B, embed_dim, H, W]
        x = x.flatten(2).permute(2,0,1)   # [H*W, B, embed_dim]
        x = self.transformer_encoder(x)   # [H*W, B, embed_dim]
        x = x.permute(1,2,0).view(B, -1, H, W)  # [B, embed_dim, H, W]
        x = self.norm(x)
        return x

class DoubleConv(nn.Module):
    """ 2 conv layers each with BatchNorm + ReLU. """
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
    """
    ResNet encoder -> Transformer -> U-Net-like decoder for segmentation.
    """
    def __init__(self, encoder_name='resnet34', pretrained=True, num_classes=2):
        super(SegViT, self).__init__()
        
        # Encoder
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
            raise ValueError("Unsupported encoder. Choose 'resnet18', 'resnet34', 'resnet50', or 'resnet101'.")

        # Modify first layer for single-channel
        self.encoder.conv1 = nn.Conv2d(
            1,
            self.encoder.conv1.out_channels,
            kernel_size=self.encoder.conv1.kernel_size,
            stride=self.encoder.conv1.stride,
            padding=self.encoder.conv1.padding,
            bias=(self.encoder.conv1.bias is not None)
        )
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
            self.encoder.layer1,  # i=4
            self.encoder.layer2,  # i=5
            self.encoder.layer3,  # i=6
            self.encoder.layer4   # i=7
        )
        
        # Transformer
        self.transformer = TransformerBridge(
            in_channels=encoder_channels[-1],
            embed_dim=768,
            num_heads=32,
            num_layers=32,
            ff_dim=3072,
            dropout=0.1
        )

        # Decoder setup
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
            if i in [0,4,5,6]:
                skip_features.append(out)

        transformed = self.transformer(out)

        up1 = self.decoder['up1'](transformed)
        skip1 = skip_features[3]  # layer3 output
        conv1 = self.decoder['conv1'](torch.cat([up1, skip1], dim=1))

        up2 = self.decoder['up2'](conv1)
        skip2 = skip_features[2]  # layer2
        conv2 = self.decoder['conv2'](torch.cat([up2, skip2], dim=1))

        up3 = self.decoder['up3'](conv2)
        skip3 = skip_features[1]  # layer1
        conv3 = self.decoder['conv3'](torch.cat([up3, skip3], dim=1))

        up4 = self.decoder['up4'](conv3)
        skip4 = skip_features[0]  # conv1 output
        conv4 = self.decoder['conv4'](torch.cat([up4, skip4], dim=1))

        out = self.out_conv(conv4)
        # Upsample back to original size (2x)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

##################################
# 3. Testing & Visualization
##################################

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Computes the Dice coefficient for binary masks.
    """
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()

def test_segmentation(model, loader, device, metrics=None):
    """
    Run inference on a test set and compute metrics.
    """
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Testing", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # If using TorchMetrics
            if metrics:
                for metric in metrics.values():
                    metric(preds, masks)

            preds_all.append(preds.cpu())
            targets_all.append(masks.cpu())

    if metrics:
        test_results = {k: v.compute().item() for k,v in metrics.items()}
        for metric in metrics.values():
            metric.reset()
        print("--- [Test] Metrics:", test_results)
    else:
        # Fallback: compute dice manually
        preds_concat = torch.cat(preds_all, dim=0)
        targets_concat = torch.cat(targets_all, dim=0)
        dice_scores = []
        for p, t in zip(preds_concat, targets_concat):
            dice_scores.append(dice_coefficient(p, t))
        test_results = {'dice': float(np.mean(dice_scores))}
        print(f"--- [Test] Dice Coefficient: {test_results['dice']:.4f}")

    return test_results

def visualize_predictions(model, dataset, device, num_samples=3, post_process_flag=False):
    """
    Display original image, ground truth, prediction, and an overlayed image.
    """
    model.eval()

    # pick random unique indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_batch)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Convert to numpy for visualization
        image_np = image.permute(1,2,0).cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Example: invert your normalization for display
        image_np = image_np * 0.485 + 0.229
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        if image_np.ndim == 2:
            image_np = np.stack([image_np]*3, axis=-1)
        elif image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)

        # Overlay
        overlay_img = image_np.copy()
        binary_mask = (pred == 1).astype(np.uint8)
        color_mask = np.zeros_like(overlay_img)
        color_mask[binary_mask == 1] = [0, 255, 0]  # green overlay
        alpha = 0.5
        overlay_img = cv2.addWeighted(overlay_img, 1-alpha, color_mask, alpha, 0)

        # Plot
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        axs[0].imshow(image_np)
        axs[0].set_title("Original")
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title("Prediction" + (" (Post-Proc)" if post_process_flag else ""))
        axs[3].imshow(overlay_img)
        axs[3].set_title("Overlay")
        for ax in axs:
            ax.axis("off")
        plt.show()


##################################
# 4. Main Testing Routine
##################################

def main():
    # ------------------------------
    # A. Paths & Hyperparameters
    # ------------------------------
    test_images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/images"
    test_masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_test/masks"
    model_path      = "/home/yanghehao/tracklearning/best_segvit_model.pth"
    batch_size      = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Using device: {device}")

    # ------------------------------
    # B. Collect test files
    # ------------------------------
    all_test_images = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Test] Found {len(all_test_images)} test images in {test_images_dir}")

    # Ensure corresponding masks exist
    filtered_test_images = []
    for fn in all_test_images:
        base = os.path.splitext(fn)[0]
        # Check for mask
        mask_extensions = [".npy", ".png", ".jpg", ".jpeg"]
        has_mask = any(os.path.isfile(os.path.join(test_masks_dir, base + ext))
                       for ext in mask_extensions)
        if has_mask:
            filtered_test_images.append(fn)
    print(f"[Test] {len(filtered_test_images)} test images have matching masks.")

    # ------------------------------
    # C. Create Test Dataset/Loader
    # ------------------------------
    test_transform = ResizeAndToTensor(size=(512, 512))
    test_dataset = BinarySegDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        file_list=filtered_test_images,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # ------------------------------
    # D. Initialize & Load Model
    # ------------------------------
    model = SegViT(encoder_name='resnet34', pretrained=False, num_classes=2).to(device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Test] Loaded model from {model_path}")

    # If you want TorchMetrics (e.g., Dice, IoU, Accuracy):
    metrics_test = {
        'dice': torchmetrics.Dice(num_classes=2, average='macro').to(device),
        'iou': torchmetrics.JaccardIndex(task='multiclass', num_classes=2).to(device),
        'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)
    }

    # ------------------------------
    # E. Run Testing
    # ------------------------------
    test_results = test_segmentation(model, test_loader, device, metrics=metrics_test)
    print("[Final Test Results]", test_results)

    # ------------------------------
    # F. Visualization (Optional)
    # ------------------------------
    if len(test_dataset) > 0:
        print("\n>>> Visualizing some predictions...")
        visualize_predictions(model, test_dataset, device, num_samples=20, post_process_flag=False)
    else:
        print("[Test] No test samples available for visualization.")

if __name__ == "__main__":
    main()
