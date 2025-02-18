import os
import glob
import random

import nibabel as nib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff


# ------------------------------------------------------------------
# 1) DATASET: FIXED-PATCH (128^3) with FOREGROUND-AWARE SAMPLING
# ------------------------------------------------------------------
class FixedPatchLungDataset(Dataset):
    """
    Loads 3D volumes (image + label), pads them if needed, 
    and extracts exactly (128,128,128) patches.
    
    - 50% chance: purely random patch
    - 50% chance: guaranteed foreground patch
    This ensures shape consistency -> No dimension mismatch in V-Net.
    """

    def __init__(self, image_dir, label_dir, patch_size=(128,128,128)):
        """
        Args:
            image_dir (str): Directory of image NIfTI files.
            label_dir (str): Directory of label NIfTI files.
            patch_size (tuple): (D, H, W) for the patch. Default is 128^3.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))
        
        assert len(self.image_paths) == len(self.label_paths), \
            "Mismatch in number of images vs. labels."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1) Load volumes
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        image_nii = nib.load(img_path)
        label_nii = nib.load(lbl_path)

        image = image_nii.get_fdata(dtype=np.float32)
        label = label_nii.get_fdata(dtype=np.float32)

        # 2) Normalize image (z-score)
        mean_val = np.mean(image)
        std_val  = np.std(image) + 1e-8
        image = (image - mean_val) / std_val

        # 3) 50% random, 50% guaranteed foreground
        if random.random() < 0.5:
            patch_img, patch_lbl = self._sample_foreground_patch(image, label)
        else:
            patch_img, patch_lbl = self._sample_random_patch(image, label)

        # 4) Expand dims -> (1, D, H, W)
        patch_img = np.expand_dims(patch_img, axis=0)  # => (1,128,128,128)
        patch_lbl = np.expand_dims(patch_lbl, axis=0)  # => (1,128,128,128)

        return torch.from_numpy(patch_img).float(), torch.from_numpy(patch_lbl).float()

    def _pad_to_min_size(self, image, label):
        """
        Pad volume so each dimension >= 128 if smaller. 
        Ensures we can safely extract a 128^3 patch without out-of-bounds or partial dimension.
        """
        D, H, W = image.shape
        pD, pH, pW = self.patch_size

        padD = max(0, pD - D)
        padH = max(0, pH - H)
        padW = max(0, pW - W)

        if padD > 0 or padH > 0 or padW > 0:
            image = np.pad(image, ((0,padD),(0,padH),(0,padW)), mode='constant')
            label = np.pad(label, ((0,padD),(0,padH),(0,padW)), mode='constant')

        return image, label

    def _sample_random_patch(self, image, label):
        # 1) Pad to ensure shape >= (128,128,128)
        image, label = self._pad_to_min_size(image, label)

        D, H, W = image.shape
        pD, pH, pW = self.patch_size

        # 2) Random indices in [0, D-128], etc.
        d0 = random.randint(0, D - pD)
        h0 = random.randint(0, H - pH)
        w0 = random.randint(0, W - pW)

        # 3) Extract
        patch_img = image[d0:d0+pD, h0:h0+pH, w0:w0+pW]
        patch_lbl = label[d0:d0+pD, h0:h0+pH, w0:w0+pW]

        return patch_img, patch_lbl

    def _sample_foreground_patch(self, image, label):
        # 1) Pad
        image, label = self._pad_to_min_size(image, label)

        D, H, W = image.shape
        pD, pH, pW = self.patch_size
        
        # 2) Find all foreground voxels
        fg_voxels = np.argwhere(label > 0.5)
        if len(fg_voxels) == 0:
            # No foreground => random
            return self._sample_random_patch(image, label)

        # 3) Pick random foreground voxel
        voxel_idx = random.choice(fg_voxels)
        d_center, h_center, w_center = voxel_idx

        # 4) Center patch around that voxel
        d0 = d_center - pD//2
        h0 = h_center - pH//2
        w0 = w_center - pW//2

        # 5) Clip to valid range
        d0 = max(0, min(d0, D - pD))
        h0 = max(0, min(h0, H - pH))
        w0 = max(0, min(w0, W - pW))

        patch_img = image[d0:d0+pD, h0:h0+pH, w0:w0+pW]
        patch_lbl = label[d0:d0+pD, h0:h0+pH, w0:w0+pW]

        return patch_img, patch_lbl


# ------------------------------------------------------------------
# 2) 3D V-NET MODEL
# ------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """3D residual block: x -> Conv -> Norm -> PReLU -> Conv -> Norm -> add -> PReLU"""
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
    """Encoder stage with optional downsampling (stride=2)."""
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
    """Decoder stage: transposed conv (upsample) + skip connection + residual block."""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.transConv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.resBlock = ResidualBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.transConv(x)  # upsample
        x = torch.cat([x, skip], dim=1)  # skip connection
        x = self.resBlock(x)
        return x


class VNet(nn.Module):
    """
    5-level encoder (4 downsamplings), 4-level decoder (4 upsamplings).
    Patch size must be divisible by 16 to avoid shape mismatch.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super(VNet, self).__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_filters, downsample=False)
        self.enc2 = EncoderBlock(base_filters, base_filters*2, downsample=True)
        self.enc3 = EncoderBlock(base_filters*2, base_filters*4, downsample=True)
        self.enc4 = EncoderBlock(base_filters*4, base_filters*8, downsample=True)
        self.enc5 = EncoderBlock(base_filters*8, base_filters*16, downsample=True)
        
        # Decoder
        self.dec5 = DecoderBlock(base_filters*16, base_filters*8)
        self.dec4 = DecoderBlock(base_filters*8,  base_filters*4)
        self.dec3 = DecoderBlock(base_filters*4,  base_filters*2)
        self.dec2 = DecoderBlock(base_filters*2,  base_filters)
        
        # Final conv
        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)    
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)  # bottleneck

        d5 = self.dec5(e5, e4)
        d4 = self.dec4(d5, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)

        out = self.out_conv(d2)
        return out


# ------------------------------------------------------------------
# 3) DICE LOSS, METRICS, TRAINING
# ------------------------------------------------------------------
class SoftDiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""
    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B,1,D,H,W) raw output
        targets: (B,1,D,H,W)
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice_coeff


def dice_coefficient(pred, target, smooth=1e-5):
    """
    pred, target: (D,H,W) binary arrays
    """
    pred_flat = pred.flatten().astype(np.float32)
    target_flat = target.flatten().astype(np.float32)
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice


def hausdorff_distance(pred, target):
    """Compute bidirectional Hausdorff for 3D binary arrays."""
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    hd_forward = directed_hausdorff(pred_points, target_points)[0]
    hd_backward = directed_hausdorff(target_points, pred_points)[0]
    return max(hd_forward, hd_backward)


def train_vnet(train_loader, val_loader, device='cuda', epochs=200, lr=1e-3):
    """
    Train V-Net with patch size=128^3. 
    Uses tqdm for progress bars.
    """
    model = VNet(in_channels=1, out_channels=1, base_filters=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = SoftDiceLoss()

    for epoch in range(1, epochs+1):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)")
        
        for (images, masks) in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} (Val)")
        with torch.no_grad():
            for (images, masks) in val_pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"[Epoch {epoch}/{epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    return model


# ------------------------------------------------------------------
# 4) SLIDING-WINDOW INFERENCE (OPTIONAL)
# ------------------------------------------------------------------
def infer_volume_sliding_window(
    model,
    volume_data,
    patch_size=(128,128,128),
    step_size=(64,64,64),
    device='cuda'
):
    """
    Sliding-window inference for volumes too large to fit in GPU at once.
    
    volume_data: (D,H,W) numpy float32 array
    patch_size: e.g. (128,128,128)
    step_size: e.g. (64,64,64) for 50% overlap
    """
    model.eval()
    D, H, W = volume_data.shape

    prob_accum = np.zeros((D, H, W), dtype=np.float32)
    count_accum = np.zeros((D, H, W), dtype=np.float32)

    pD, pH, pW = patch_size
    sD, sH, sW = step_size

    for d0 in range(0, D, sD):
        d1 = min(d0 + pD, D)
        for h0 in range(0, H, sH):
            h1 = min(h0 + pH, H)
            for w0 in range(0, W, sW):
                w1 = min(w0 + pW, W)

                patch = volume_data[d0:d1, h0:h1, w0:w1]
                patch_tensor = torch.from_numpy(patch[None, None, ...]).float().to(device)

                with torch.no_grad():
                    logits = model(patch_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()[0,0]

                prob_accum[d0:d0+probs.shape[0],
                           h0:h0+probs.shape[1],
                           w0:w0+probs.shape[2]] += probs

                count_accum[d0:d0+probs.shape[0],
                            h0:h0+probs.shape[1],
                            w0:w0+probs.shape[2]] += 1

    avg_prob = prob_accum / np.maximum(count_accum, 1e-5)
    final_mask = (avg_prob > 0.5).astype(np.uint8)
    return final_mask


# ------------------------------------------------------------------
# 5) MAIN: EXAMPLE TRAIN + EVAL
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Adjust these paths:
    image_dir = "/home/yanghehao/Task06_Lung/imagesTr"
    label_dir = "/home/yanghehao/Task06_Lung/labelsTr"

    # 2) Create dataset -> patch_size=128^3
    dataset = FixedPatchLungDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        patch_size=(128,128,128)
    )

    # (Optional) Quick check on shapes
    for i in range(3):
        idx = random.randint(0, len(dataset)-1)
        patch_img, patch_lbl = dataset[idx]
        print(f"Check patch {i+1}: idx={idx}")
        print(f"  patch_img shape: {patch_img.shape}, patch_lbl shape: {patch_lbl.shape}")
        print(f"  foreground count: {(patch_lbl.numpy()>0.5).sum()}")
        print("-------------------------------------------------------")

    # 3) Train/Val split
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 4) DataLoaders
    # Because 128^3 patches are big -> batch_size=1 recommended
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # 5) Train the V-Net
    model = train_vnet(train_loader, val_loader, device=device, epochs=200, lr=1e-3)

    # 6) Save the model
    save_path = "vnet_lungseg_fixed128.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 7) Example: sliding-window inference on ONE validation volume
    example_idx = val_dataset.indices[0]  # pick first val sample
    example_vol_path = dataset.image_paths[example_idx]
    example_lbl_path = dataset.label_paths[example_idx]

    # Load & normalize
    vol_nii = nib.load(example_vol_path)
    vol_data = vol_nii.get_fdata(dtype=np.float32)
    vol_data = (vol_data - vol_data.mean()) / (vol_data.std() + 1e-8)

    # Sliding-window inference (optional)
    pred_mask = infer_volume_sliding_window(
        model=model,
        volume_data=vol_data,
        patch_size=(128,128,128),
        step_size=(64,64,64),  # 50% overlap
        device=device
    )

    # Compare to ground truth
    gt_nii = nib.load(example_lbl_path)
    gt_data = gt_nii.get_fdata(dtype=np.float32)
    gt_data = (gt_data > 0.5).astype(np.uint8)

    dsc = dice_coefficient(pred_mask, gt_data)
    hd  = hausdorff_distance(pred_mask, gt_data)
    print(f"Inference on {os.path.basename(example_vol_path)}:")
    print(f"  Dice = {dsc:.4f}")
    print(f"  Hausdorff Distance = {hd:.2f}")

    # Optionally save the predicted mask
    pred_nii = nib.Nifti1Image(pred_mask, vol_nii.affine)
    nib.save(pred_nii, "pred_lung_fixed128.nii.gz")
    print("Done.")
