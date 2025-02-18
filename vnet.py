#!/usr/bin/env python
"""
vnet_with_visualization_full.py

- Patch-based training using V-Net
- Before training, visualize random slices from 10 training images/labels
- After training, run inference on up to 10 validation volumes (no patch extraction)
  and for each volume, display up to 10 random slices. Each slice is shown in a separate
  figure with two subplots:
    (left) image + ground truth overlay
    (right) image + predicted mask overlay
"""

import os
import json
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff, cdist

##############################
# 1) V-Net Model Definitions
##############################

class LUConv(nn.Module):
    def __init__(self, n_channels):
        super(LUConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size=5, padding=2),
            nn.InstanceNorm3d(n_channels),
            nn.PReLU(n_channels)
        )
    def forward(self, x):
        return self.conv(x)

def make_nConv(n_channels, depth):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(n_channels))
    return nn.Sequential(*layers)

class InputTransition(nn.Module):
    def __init__(self, out_channels):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # Expand input to match channels (residual connection)
        x_expanded = x.repeat(1, out.size(1), 1, 1, 1)
        out = self.prelu(out + x_expanded)
        return out

class DownTransition(nn.Module):
    def __init__(self, in_channels, n_convs):
        super(DownTransition, self).__init__()
        out_channels = 2 * in_channels
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.ops = make_nConv(out_channels, n_convs)

    def forward(self, x):
        down = self.prelu(self.bn1(self.down_conv(x)))
        out = self.ops(down)
        out = self.prelu(out + down)
        return out

class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.ops = make_nConv(out_channels * 2, n_convs)
        self.reduce_conv = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)
        self.prelu_out = nn.PReLU(out_channels)

    def forward(self, x, skipx):
        out = self.prelu(self.bn1(self.up_conv(x)))
        if skipx.shape[2:] != out.shape[2:]:
            skipx = self.center_crop(skipx, out.shape[2:])
        out = torch.cat((out, skipx), dim=1)
        out = self.ops(out)
        out = self.reduce_conv(out)
        out = self.prelu_out(out)
        return out

    def center_crop(self, tensor, target_spatial):
        # tensor: shape [B, C, D, H, W]
        # target_spatial: (D_out, H_out, W_out)
        _, _, d, h, w = tensor.shape
        td, th, tw = target_spatial
        sd = max((d - td) // 2, 0)
        sh = max((h - th) // 2, 0)
        sw = max((w - tw) // 2, 0)
        return tensor[:, :, sd:sd+td, sh:sh+th, sw:sw+tw]

class OutputTransition(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, n_classes, kernel_size=5, padding=2)
    def forward(self, x):
        return self.conv1(x)

class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, base_filters=16):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(base_filters)
        self.down_tr32 = DownTransition(base_filters, 1)
        self.down_tr64 = DownTransition(base_filters*2, 2)
        self.down_tr128 = DownTransition(base_filters*4, 3)
        self.down_tr256 = DownTransition(base_filters*8, 3)
        self.up_tr256 = UpTransition(base_filters*16, base_filters*8, 3)
        self.up_tr128 = UpTransition(base_filters*8, base_filters*4, 3)
        self.up_tr64 = UpTransition(base_filters*4, base_filters*2, 2)
        self.up_tr32 = UpTransition(base_filters*2, base_filters, 1)
        self.out_tr = OutputTransition(base_filters, n_classes)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        up256 = self.up_tr256(out256, out128)
        up128 = self.up_tr128(up256, out64)
        up64 = self.up_tr64(up128, out32)
        up32 = self.up_tr32(up64, out16)
        out = self.out_tr(up32)
        return out

##############################
# Dice Loss
##############################

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes)\
                                        .permute(0, 4, 1, 2, 3).float()
        intersection = torch.sum(probs * targets_onehot, dim=(2,3,4))
        union = torch.sum(probs + targets_onehot, dim=(2,3,4))
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - torch.mean(dice_score)

##############################
# Patch-Based Dataset
##############################

class LazyLungDataset(Dataset):
    """
    Patch-based dataset for training/validation. We sample random 3D patches.
    """
    def __init__(self, image_paths, label_paths, patch_size=(64,64,64)):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.image_paths[idx])
        label = sitk.ReadImage(self.label_paths[idx])
        image_np = sitk.GetArrayFromImage(image)  # [D, H, W]
        label_np = sitk.GetArrayFromImage(label)
        
        d, h, w = image_np.shape
        pd, ph, pw = self.patch_size
        start_d = np.random.randint(0, max(1, d - pd))
        start_h = np.random.randint(0, max(1, h - ph))
        start_w = np.random.randint(0, max(1, w - pw))
        
        patch_img = image_np[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
        patch_lbl = label_np[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
        
        patch_img = np.expand_dims(patch_img, axis=0).astype(np.float32)
        patch_lbl = patch_lbl.astype(np.int64)
        return torch.tensor(patch_img), torch.tensor(patch_lbl)

##############################
# Hausdorff Distance
##############################

def compute_hausdorff_distance(pred, gt):
    pred_coords = np.argwhere(pred==1)
    gt_coords = np.argwhere(gt==1)
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.nan, np.nan
    hd_forward = directed_hausdorff(pred_coords, gt_coords)[0]
    hd_backward = directed_hausdorff(gt_coords, pred_coords)[0]
    hd = max(hd_forward, hd_backward)
    
    dists_forward = np.min(cdist(pred_coords, gt_coords), axis=1)
    dists_backward = np.min(cdist(gt_coords, pred_coords), axis=1)
    avg_hd = (np.mean(dists_forward) + np.mean(dists_backward)) / 2.0
    return hd, avg_hd

##############################
# Training & Validation
##############################

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, lbls in tqdm(train_loader, desc="Training", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    dice_scores = 0.0
    hd_scores = []
    avg_hd_scores = []
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader, desc="Validation", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            dice_scores += (1 - loss.item())

            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            preds_np = preds.cpu().numpy()
            lbls_np = lbls.cpu().numpy()
            for pred_patch, gt_patch in zip(preds_np, lbls_np):
                hd, avg_hd = compute_hausdorff_distance(pred_patch, gt_patch)
                if not np.isnan(hd):
                    hd_scores.append(hd)
                    avg_hd_scores.append(avg_hd)
    avg_dice = dice_scores / len(val_loader)
    avg_hd_overall = np.mean(hd_scores) if hd_scores else np.nan
    avg_avg_hd_overall = np.mean(avg_hd_scores) if avg_hd_scores else np.nan
    return avg_dice, avg_hd_overall, avg_avg_hd_overall

##############################
# Visualization
##############################

def visualize_training_samples(image_files, label_files, num_samples=10):
    """
    Display random slices from up to 10 training images/labels before training.
    """
    indices = np.random.choice(len(image_files), size=min(num_samples, len(image_files)), replace=False)
    for idx in indices:
        image = sitk.ReadImage(image_files[idx])
        label = sitk.ReadImage(label_files[idx])
        image_np = sitk.GetArrayFromImage(image)
        label_np = sitk.GetArrayFromImage(label)
        # Show a random slice
        z = np.random.randint(0, image_np.shape[0])
        
        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1)
        plt.imshow(image_np[z], cmap='gray')
        plt.title(f"Image slice z={z}\n{os.path.basename(image_files[idx])}")
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(label_np[z], cmap='gray', alpha=1)
        plt.title("Label slice")
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(image_np[z], cmap='gray')
        plt.imshow(label_np[z], cmap='gray', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        plt.show()

def visualize_random_slices_from_full_volume(model, image_path, label_path, device='cpu', num_slices=10):
    """
    Load the full volume (no patch extraction), run inference,
    then pick 'num_slices' random slices. For each slice, display
    a single figure with 2 subplots:
      (left) image + ground truth overlay
      (right) image + predicted mask overlay
    """
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)
    image_np = sitk.GetArrayFromImage(image)
    label_np = sitk.GetArrayFromImage(label)

    vol_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(vol_tensor)
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

    # If prediction depth != original, center-crop the image and label
    d_pred = preds.shape[0]
    d_orig = image_np.shape[0]
    if d_pred != d_orig:
        start = (d_orig - d_pred) // 2
        image_np = image_np[start:start+d_pred]
        label_np = label_np[start:start+d_pred]

    total_slices = preds.shape[0]
    slice_indices = np.sort(np.random.choice(total_slices, size=min(num_slices, total_slices), replace=False))

    # For each chosen slice, create a new figure with 2 subplots
    for z in slice_indices:
        fig, axes = plt.subplots(1, 2, figsize=(10,5))

        # Left subplot => image + ground truth overlay
        axes[0].imshow(image_np[z], cmap='gray')
        axes[0].imshow(label_np[z], cmap='jet', alpha=0.5)
        axes[0].set_title(f"Slice {z}\nImage + GT")
        axes[0].axis('off')

        # Right subplot => image + predicted mask overlay
        axes[1].imshow(image_np[z], cmap='gray')
        axes[1].imshow(preds[z], cmap='jet', alpha=0.5)
        axes[1].set_title(f"Slice {z}\nImage + Prediction")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

##############################
# Main
##############################

def main():
    dataset_json_path = '/home/yanghehao/Task06_Lung/dataset.json'
    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)
    base_dir = os.path.dirname(dataset_json_path)
    training_entries = dataset_info.get("training", [])
    train_image_files = [os.path.join(base_dir, e["image"]) for e in training_entries]
    train_label_files = [os.path.join(base_dir, e["label"]) for e in training_entries]
    
    # Split train/val
    num_samples = len(train_image_files)
    split_index = int(0.7 * num_samples)
    train_img_files = train_image_files[:split_index]
    train_lbl_files = train_label_files[:split_index]
    val_img_files = train_image_files[split_index:]
    val_lbl_files = train_label_files[split_index:]
    
    print(f"Total volumes: {num_samples}")
    print(f"Training: {len(train_img_files)} volumes, Validation: {len(val_img_files)} volumes")

    # 1) Visualize up to 10 training samples (random slices) before training
    visualize_training_samples(train_img_files, train_lbl_files, num_samples=10)

    # 2) Create patch-based datasets & loaders for training
    train_dataset = LazyLungDataset(train_img_files, train_lbl_files, patch_size=(64,64,64))
    val_dataset = LazyLungDataset(val_img_files, val_lbl_files, patch_size=(64,64,64))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model, loss, optimizer
    model = VNet(n_channels=1, n_classes=2, base_filters=16).to(device)
    criterion = DiceLoss(smooth=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_dice, val_hd, val_avg_hd = validate(model, val_loader, criterion, device)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Dice Score: {val_dice:.4f}")
        print(f"  Val Hausdorff Distance: {val_hd:.4f}")
        print(f"  Val Average Hausdorff Distance: {val_avg_hd:.4f}")
    
    # 3) Visualize final segmentation on up to 10 validation volumes
    #    and for each volume, display 10 random slices, each in a separate figure
    if len(val_img_files) > 0:
        sample_index = 0  # you can choose another index if desired
        sample_image_path = val_img_files[sample_index]
        sample_label_path = val_lbl_files[sample_index]
        print(f"\nVisualizing volume  from validation set")
        visualize_random_slices_from_full_volume(
            model,
            sample_image_path,
            sample_label_path,
            device=device,
            num_slices=10
        )

if __name__ == "__main__":
    main()
