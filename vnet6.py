import os
import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Optional: for Hausdorff distance
from scipy.spatial.distance import directed_hausdorff

# Optional: for progress bars
from tqdm import tqdm


# -------------------------------------------------
# 1) DATASET: PURE COVERAGE (SLIDING-WINDOW)
# -------------------------------------------------
class SlidingWindowFullCoverageDataset(Dataset):
    """
    Enumerates ALL 3D patches of size `patch_size` in each volume, stepping by `stride`.
    Ensures complete coverage of the volume with no random sampling.
    """
    def __init__(self, image_dir, label_dir, patch_size=(64,64,64), stride=(64,64,64)):
        """
        Args:
            image_dir (str): Directory with NIfTI image volumes (.nii / .nii.gz)
            label_dir (str): Directory with matching label volumes
            patch_size (tuple): (D, H, W) patch dimensions
            stride (tuple): (Sz, Sy, Sx) step size in each dimension
        """
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.stride = stride
        
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))
        assert len(self.image_paths) == len(self.label_paths), \
            "Mismatch between number of images and labels."
        
        # Pre-compute a list of (img_path, lbl_path, z, y, x) for all patches
        self.samples = []
        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            # Load to get shape (we won't store the volume data here)
            img_nii = nib.load(img_path)
            image_data = img_nii.get_fdata(dtype=np.float32)
            
            D, H, W = image_data.shape
            pD, pH, pW = self.patch_size
            sD, sH, sW = self.stride
            
            # Enumerate all valid patches
            # e.g. range(0, D-pD+1, sD)
            for z in range(0, max(1, D - pD + 1), sD):
                for y in range(0, max(1, H - pH + 1), sH):
                    for x in range(0, max(1, W - pW + 1), sW):
                        self.samples.append((img_path, lbl_path, z, y, x))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, lbl_path, z, y, x = self.samples[idx]
        
        # Load volumes on-the-fly
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        image = img_nii.get_fdata(dtype=np.float32)
        label = lbl_nii.get_fdata(dtype=np.float32)
        
        pD, pH, pW = self.patch_size
        
        # Extract patch
        patch_img = image[z:z+pD, y:y+pH, x:x+pW]
        patch_lbl = label[z:z+pD, y:y+pH, x:x+pW]
        
        # (Optional) Intensity normalization
        # This is a simple example (z-score).
        mean_val = np.mean(patch_img)
        std_val = np.std(patch_img) + 1e-8
        patch_img = (patch_img - mean_val) / std_val
        
        # Expand dims => (1, D, H, W)
        patch_img = np.expand_dims(patch_img, axis=0)
        patch_lbl = np.expand_dims(patch_lbl, axis=0)
        
        return torch.from_numpy(patch_img).float(), torch.from_numpy(patch_lbl).float()


# -------------------------------------------------
# 2) V-NET ARCHITECTURE
# -------------------------------------------------
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
            # 2x downsample
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
    """
    5-level encoder (4 downsamplings) + 4-level decoder, typical V-Net style.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super(VNet, self).__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_filters, downsample=False)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2, downsample=True)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4, downsample=True)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8, downsample=True)
        self.enc5 = EncoderBlock(base_filters * 8, base_filters * 16, downsample=True)
        
        # Decoder
        self.dec5 = DecoderBlock(base_filters * 16, base_filters * 8)
        self.dec4 = DecoderBlock(base_filters * 8, base_filters * 4)
        self.dec3 = DecoderBlock(base_filters * 4, base_filters * 2)
        self.dec2 = DecoderBlock(base_filters * 2, base_filters)
        
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


# -------------------------------------------------
# 3) LOSS & METRICS
# -------------------------------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, 1, D, H, W) raw output
        targets: (B, 1, D, H, W)
        """
        probs = torch.sigmoid(logits)
        
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / \
                     (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice_coeff


def dice_coefficient(pred, target, smooth=1e-5):
    """
    pred, target: 3D binary arrays
    """
    pred_flat = pred.flatten().astype(np.float32)
    target_flat = target.flatten().astype(np.float32)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice


def hausdorff_distance(pred, target):
    """
    pred, target: 3D binary (D, H, W).
    """
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    
    hd_forward = directed_hausdorff(pred_points, target_points)[0]
    hd_backward = directed_hausdorff(target_points, pred_points)[0]
    return max(hd_forward, hd_backward)


# -------------------------------------------------
# 4) TRAINING LOOP
# -------------------------------------------------
def train_vnet(train_loader, val_loader, device='cuda', epochs=20, lr=1e-3):
    model = VNet(in_channels=1, out_channels=1, base_filters=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = SoftDiceLoss()

    for epoch in range(1, epochs+1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)")
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} (Val)")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"[Epoch {epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return model


# -------------------------------------------------
# 5) FULL-VOLUME INFERENCE
# -------------------------------------------------
def infer_full_volume(model, volume_path, device='cuda'):
    model.eval()
    
    vol_nii = nib.load(volume_path)
    vol_data = vol_nii.get_fdata(dtype=np.float32)
    
    # Simple normalization
    vol_data = (vol_data - vol_data.mean()) / (vol_data.std() + 1e-8)
    
    # shape => (1, 1, D, H, W)
    vol_tensor = torch.from_numpy(vol_data).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(vol_tensor)
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        pred = (probs > 0.5).astype(np.uint8)
    
    return pred


# -------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------
if __name__ == "__main__":
    # Adjust paths
    image_dir = "/home/yanghehao/Task06_Lung/imagesTr"
    label_dir = "/home/yanghehao/Task06_Lung/labelsTr"
    
    # Build the dataset with full coverage
    # Example: patch_size=64, stride=64 => non-overlapping patches
    dataset = SlidingWindowFullCoverageDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        patch_size=(64,64,64),
        stride=(64,64,64)
    )
    
    print(f"Dataset size (total patches): {len(dataset)}")
    
    # Train/val split
    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=2)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Train
    model = train_vnet(train_loader, val_loader, device=device, epochs=20, lr=1e-3)
    
    # Save
    torch.save(model.state_dict(), "vnet_fullcoverage.pth")
    print("Model saved as vnet_fullcoverage.pth")
    
    # Example: Inference on one volume from the dataset
    example_idx = val_dataset.indices[0]  # pick one index from the val set
    example_vol_path = dataset.image_paths[example_idx]
    example_lbl_path = dataset.label_paths[example_idx]
    
    # Full-volume inference
    pred_mask = infer_full_volume(model, example_vol_path, device=device)
    
    # Load GT label
    gt_data = nib.load(example_lbl_path).get_fdata(dtype=np.float32)
    gt_data = (gt_data > 0.5).astype(np.uint8)
    
    # Evaluate
    dice_val = dice_coefficient(pred_mask, gt_data)
    hd_val = hausdorff_distance(pred_mask, gt_data)
    print(f"Inference on {os.path.basename(example_vol_path)}:")
    print(f"  Dice: {dice_val:.4f}")
    print(f"  Hausdorff Distance: {hd_val:.2f}")
