import os
import glob
import math
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

###############################################################################
#                        USER OPTIONS (EDIT HERE)                             #
###############################################################################
images_dir = "imagesTr"       # Directory containing training images
labels_dir = "labelsTr"        # Directory containing training labels
model_save_path = "best_vnet_model.pth"        # Path to save the best model during training
val_ratio = 0.2                                # Fixed split: last 20% as validation

num_epochs = 200
learning_rate = 1e-4
batch_size = 4
num_workers = 4

# Parameters for sliding-window inference (used during training patch extraction and later testing)
patch_size = (64, 64, 64)
step_size  = (32, 32, 32)
inference_threshold = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
#                      HELPER: COMPUTE START POSITIONS                          #
###############################################################################
def compute_start_positions(volume_dim, patch_dim, step):
    if patch_dim >= volume_dim:
        return [0]
    positions = []
    start = 0
    while True:
        positions.append(start)
        next_start = start + step
        if next_start + patch_dim > volume_dim:
            final_start = volume_dim - patch_dim
            if final_start > start:
                positions.append(final_start)
            break
        start = next_start
    return sorted(set(positions))

###############################################################################
#                   FOREGROUND-AWARE DATASET (For Training)                   #
###############################################################################
class ForegroundAwareLungDataset(Dataset):
    """
    Loads entire 3D volumes from images_dir & labels_dir and extracts patches.
    With 90% probability, a patch is sampled to include foreground; otherwise, a random patch is sampled.
    """
    def __init__(self, image_input, label_input, patch_size=(64,64,64)):
        if isinstance(image_input, list):
            self.image_paths = image_input
        else:
            self.image_paths = sorted(glob.glob(os.path.join(image_input, "*.nii*")))
        if isinstance(label_input, list):
            self.label_paths = label_input
        else:
            self.label_paths = sorted(glob.glob(os.path.join(label_input, "*.nii*")))
        assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels."
        self.patch_size = patch_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
        
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        
        image = img_nii.get_fdata(dtype=np.float32)
        label = lbl_nii.get_fdata(dtype=np.float32)
        
        # Normalize image
        mean_val = np.mean(image)
        std_val = np.std(image) + 1e-8
        image = (image - mean_val) / std_val
        
        # With 90% probability, sample a foreground patch; else random patch.
        if random.random() < 0.5:
            patch_img, patch_lbl = self._sample_foreground_patch(image, label)
        else:
            patch_img, patch_lbl = self._sample_random_patch(image, label)
        
        patch_img = np.expand_dims(patch_img, axis=0)
        patch_lbl = np.expand_dims(patch_lbl, axis=0)
        return torch.from_numpy(patch_img).float(), torch.from_numpy(patch_lbl).float()
    
    def _sample_random_patch(self, image, label):
        D, H, W = image.shape
        pD, pH, pW = self.patch_size
        if D < pD or H < pH or W < pW:
            padD = max(0, pD - D)
            padH = max(0, pH - H)
            padW = max(0, pW - W)
            image = np.pad(image, ((0, padD), (0, padH), (0, padW)), mode='constant')
            label = np.pad(label, ((0, padD), (0, padH), (0, padW)), mode='constant')
            D, H, W = image.shape
        d0 = random.randint(0, D - pD)
        h0 = random.randint(0, H - pH)
        w0 = random.randint(0, W - pW)
        patch_img = image[d0:d0+pD, h0:h0+pH, w0:w0+pW]
        patch_lbl = label[d0:d0+pD, h0:h0+pH, w0:w0+pW]
        return patch_img, patch_lbl
    
    def _sample_foreground_patch(self, image, label):
        foreground_voxels = np.argwhere(label > 0.5)
        if len(foreground_voxels) == 0:
            return self._sample_random_patch(image, label)
        pD, pH, pW = self.patch_size
        D, H, W = image.shape
        voxel_idx = random.choice(foreground_voxels)
        d_center, h_center, w_center = voxel_idx
        d0 = d_center - pD // 2
        h0 = h_center - pH // 2
        w0 = w_center - pW // 2
        d0 = max(0, min(d0, D - pD))
        h0 = max(0, min(h0, H - pH))
        w0 = max(0, min(w0, W - pW))
        patch_img = image[d0:d0+pD, h0:h0+pH, w0:w0+pW]
        patch_lbl = label[d0:d0+pD, h0:h0+pH, w0:w0+pW]
        return patch_img, patch_lbl

###############################################################################
#                     FULL VOLUME DATASET (For Validation Testing)              #
###############################################################################
class FullVolumeDataset(Dataset):
    """
    Loads full 3D volumes (without patch extraction) from provided file lists.
    """
    def __init__(self, image_input, label_input):
        if isinstance(image_input, list):
            self.image_paths = image_input
        else:
            self.image_paths = sorted(glob.glob(os.path.join(image_input, "*.nii*")))
        if isinstance(label_input, list):
            self.label_paths = label_input
        else:
            self.label_paths = sorted(glob.glob(os.path.join(label_input, "*.nii*")))
        assert len(self.image_paths) == len(self.label_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
        image_nii = nib.load(img_path)
        label_nii = nib.load(lbl_path)
        image_data = image_nii.get_fdata(dtype=np.float32)
        label_data = label_nii.get_fdata(dtype=np.float32)
        return (image_data, label_data), (img_path, lbl_path)

###############################################################################
#                      V-NET MODEL DEFINITION (TESTING ONLY)                  #
###############################################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.prelu2 = nn.PReLU()
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        out = out + identity
        out = self.prelu2(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(EncoderBlock, self).__init__()
        self.resBlock = ResidualBlock(in_channels, out_channels)
        self.downsample = downsample
        if downsample:
            self.downConv = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.downConv = None
    
    def forward(self, x):
        x = self.resBlock(x)
        if self.downsample:
            x = self.downConv(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.transConv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.resBlock = ResidualBlock(out_channels * 2, out_channels)
    
    def forward(self, x, skip):
        x = self.transConv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.resBlock(x)
        return x

class VNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super(VNet, self).__init__()
        self.enc1 = EncoderBlock(in_channels,      base_filters,      downsample=False)
        self.enc2 = EncoderBlock(base_filters,     base_filters * 2,  downsample=True)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4,  downsample=True)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8,  downsample=True)
        self.enc5 = EncoderBlock(base_filters * 8, base_filters * 16, downsample=True)
        
        self.dec5 = DecoderBlock(base_filters * 16, base_filters * 8)
        self.dec4 = DecoderBlock(base_filters * 8,  base_filters * 4)
        self.dec3 = DecoderBlock(base_filters * 4,  base_filters * 2)
        self.dec2 = DecoderBlock(base_filters * 2,  base_filters)
        
        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d5 = self.dec5(e5, e4)
        d4 = self.dec4(d5, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        out = self.out_conv(d2)
        return out

###############################################################################
#                          SOFT DICE LOSS DEFINITION                          #
###############################################################################
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

###############################################################################
#                          SLIDING-WINDOW INFERENCE                           #
###############################################################################
def sliding_window_inference(model, volume_np, patch_size=(64,64,64),
                             step_size=(32,32,32), device="cuda"):
    model.eval()
    D, H, W = volume_np.shape
    pD, pH, pW = patch_size
    sD, sH, sW = step_size
    
    prob_map = np.zeros((D, H, W), dtype=np.float32)
    weight_map = np.zeros((D, H, W), dtype=np.float32)
    
    mean_val = volume_np.mean()
    std_val = volume_np.std() + 1e-8
    norm_volume = (volume_np - mean_val) / std_val
    
    d_starts = compute_start_positions(D, pD, sD)
    h_starts = compute_start_positions(H, pH, sH)
    w_starts = compute_start_positions(W, pW, sW)
    
    with torch.no_grad():
        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    patch = norm_volume[d0:d0+pD, h0:h0+pH, w0:w0+pW]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    logits = model(patch_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy().squeeze(0).squeeze(0)
                    prob_map[d0:d0+pD, h0:h0+pH, w0:w0+pW] += probs
                    weight_map[d0:d0+pD, h0:h0+pH, w0:w0+pW] += 1.0
    weight_map[weight_map == 0] = 1e-8
    prob_map /= weight_map
    return prob_map

###############################################################################
#                          METRICS FUNCTIONS                                  #
###############################################################################
def dice_coefficient(pred, target, smooth=1e-5):
    pred_flat = pred.flatten().astype(np.float32)
    target_flat = target.flatten().astype(np.float32)
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice

def hausdorff_distance(pred, target):
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    hd_forward = directed_hausdorff(pred_points, target_points)[0]
    hd_backward = directed_hausdorff(target_points, pred_points)[0]
    return max(hd_forward, hd_backward)

###############################################################################
#              TRAINING FUNCTION (SAVE BEST MODEL BASED ON VAL DICE)           #
###############################################################################
def train_vnet(train_loader, val_loader, device="cuda", epochs=200, lr=1e-4):
    model = VNet(in_channels=1, out_channels=1, base_filters=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = SoftDiceLoss()
    
    best_val_dice = -1.0
    best_model_state = None

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        train_loss /= len(train_loader)
        
        # Validation phase: compute Dice for each sample
        model.eval()
        val_dice_scores = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} (Val)")
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                for i in range(preds.size(0)):
                    pred_np = preds[i].cpu().numpy().squeeze()
                    mask_np = masks[i].cpu().numpy().squeeze()
                    d = dice_coefficient(pred_np, mask_np)
                    val_dice_scores.append(d)
            avg_val_dice = np.mean(val_dice_scores) if len(val_dice_scores) > 0 else 0.0
        
        print(f"[Epoch {epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
        
        # Save best model based on validation Dice
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_model_state = model.state_dict()
            print(f"  --> New best model (Val Dice: {best_val_dice:.4f}) saved.")
    
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to {model_save_path}")
    
    return model

###############################################################################
#                               MAIN SCRIPT                                 #
###############################################################################
def main():
    import math
    # 1. Create the full dataset from training folders
    dataset = ForegroundAwareLungDataset(images_dir, labels_dir, patch_size=patch_size)
    total = len(dataset)
    print(f"Total volumes in dataset: {total}")
    
    # 2. Fixed split: use the LAST 20% as the validation set
    train_size = total - int(total * val_ratio)
    # Slice sorted lists:
    train_image_paths = dataset.image_paths[:train_size]
    train_label_paths = dataset.label_paths[:train_size]
    val_image_paths = dataset.image_paths[train_size:]
    val_label_paths = dataset.label_paths[train_size:]
    
    train_dataset = ForegroundAwareLungDataset(train_image_paths, train_label_paths, patch_size=patch_size)
    val_dataset = ForegroundAwareLungDataset(val_image_paths, val_label_paths, patch_size=patch_size)
    
    print(f"Train set: {len(train_dataset)} volumes, Validation set: {len(val_dataset)} volumes")
    
    # 3. Create DataLoaders for training and validation (patch-based)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 4. Train the V-Net and save the best model based on validation Dice
    trained_model = train_vnet(train_loader, val_loader, device=device, epochs=num_epochs, lr=learning_rate)
    
    # 5. Create a full-volume dataset for validation testing
    class FullVolumeDataset(Dataset):
        def __init__(self, image_paths, label_paths):
            self.image_paths = image_paths
            self.label_paths = label_paths
            assert len(self.image_paths) == len(self.label_paths)
        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            lbl_path = self.label_paths[idx]
            image_nii = nib.load(img_path)
            label_nii = nib.load(lbl_path)
            image_data = image_nii.get_fdata(dtype=np.float32)
            label_data = label_nii.get_fdata(dtype=np.float32)
            return (image_data, label_data), (img_path, lbl_path)
    
    val_full_dataset = FullVolumeDataset(val_image_paths, val_label_paths)
    print(f"\nRunning sliding-window inference on all {len(val_full_dataset)} validation volumes...")
    
    all_dice = []
    all_hd = []
    
    for idx in range(len(val_full_dataset)):
        (vol, lbl), (img_path, lbl_path) = val_full_dataset[idx]
        file_name = os.path.basename(img_path)
        print(f"\nProcessing {file_name} ...")
        
        prob_map = sliding_window_inference(trained_model, vol, patch_size, step_size, device=device)
        pred_mask = (prob_map > inference_threshold).astype(np.uint8)
        
        lbl_bin = (lbl > 0.5).astype(np.uint8)
        dice_val = dice_coefficient(pred_mask, lbl_bin)
        hd_val = hausdorff_distance(pred_mask, lbl_bin)
        all_dice.append(dice_val)
        all_hd.append(hd_val)
        
        print(f"  Dice = {dice_val:.4f} | Hausdorff = {hd_val:.2f}")
    
    if all_dice:
        print("\n==== Overall Validation Metrics ====")
        print(f"Mean Dice = {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
        print(f"Mean Hausdorff = {np.mean(all_hd):.2f} ± {np.std(all_hd):.2f}")

if __name__ == "__main__":
    main()
