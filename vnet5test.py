import os
import glob
import math
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from scipy.spatial.distance import directed_hausdorff

###############################################################################
#                        USER OPTIONS (EDIT HERE)                             #
###############################################################################
images_dir = "imagesTr"      # Training images directory
labels_dir = "labelsTr"       # Training labels directory
model_path = "vnet_lungseg_sliding_window.pth"  # Pretrained model (.pth file)
test_ratio = 0.2                            # Use the last 20% of files as test set

patch_size = (64, 64, 64)   # Patch size for sliding-window inference
step_size  = (32, 32, 32)   # Step size for sliding-window inference
threshold  = 0.5            # Threshold for binary segmentation
device     = "cuda"         # "cuda" or "cpu"

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
#                          DATASET DEFINITION                                 #
###############################################################################
class LungDataset(Dataset):
    """
    Loads entire 3D volumes (images and labels). If a list of file paths is provided,
    it uses them directly; otherwise, it assumes a directory is provided and uses glob.
    """
    def __init__(self, images_input, labels_input):
        # If images_input is a list, use it; otherwise, assume it's a directory.
        if isinstance(images_input, list):
            self.image_paths = images_input
        else:
            self.image_paths = sorted(glob.glob(os.path.join(images_input, "*.nii*")))
        
        if isinstance(labels_input, list):
            self.label_paths = labels_input
        else:
            self.label_paths = sorted(glob.glob(os.path.join(labels_input, "*.nii*")))
        
        assert len(self.image_paths) == len(self.label_paths), (
            f"Number of images ({len(self.image_paths)}) does not match number of labels ({len(self.label_paths)})."
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
        
        image_nii = nib.load(img_path)
        label_nii = nib.load(lbl_path)
        
        image_data = image_nii.get_fdata(dtype=np.float32)  # shape (D,H,W)
        label_data = label_nii.get_fdata(dtype=np.float32)  # shape (D,H,W)
        
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
        self.enc2 = EncoderBlock(base_filters,     base_filters*2,    downsample=True)
        self.enc3 = EncoderBlock(base_filters*2,   base_filters*4,    downsample=True)
        self.enc4 = EncoderBlock(base_filters*4,   base_filters*8,    downsample=True)
        self.enc5 = EncoderBlock(base_filters*8,   base_filters*16,   downsample=True)
        
        self.dec5 = DecoderBlock(base_filters*16, base_filters*8)
        self.dec4 = DecoderBlock(base_filters*8,  base_filters*4)
        self.dec3 = DecoderBlock(base_filters*4,  base_filters*2)
        self.dec2 = DecoderBlock(base_filters*2,  base_filters)
        
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
#                          SLIDING-WINDOW INFERENCE                           #
###############################################################################
def sliding_window_inference(model, volume_np, patch_size=(64,64,64), step_size=(32,32,32), device="cuda"):
    model.eval()
    D, H, W = volume_np.shape
    pD, pH, pW = patch_size
    sD, sH, sW = step_size
    
    prob_map = np.zeros((D, H, W), dtype=np.float32)
    weight_map = np.zeros((D, H, W), dtype=np.float32)
    
    mean_val = volume_np.mean()
    std_val  = volume_np.std() + 1e-8
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
#                          METRICS (OPTIONAL)
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
#                               MAIN TEST SCRIPT                            #
###############################################################################
def main():
    import math
    # 1. Load the full dataset from the training folders
    full_dataset = LungDataset(images_dir, labels_dir)
    total = len(full_dataset)
    print(f"Total volumes in dataset: {total}")
    
    # 2. Take the LAST 20% as the test set (fixed, not random)
    test_size = int(math.floor(test_ratio * total))
    train_size = total - test_size
    # Since files are sorted, take the last test_size elements:
    test_image_paths = full_dataset.image_paths[train_size:]
    test_label_paths = full_dataset.label_paths[train_size:]
    
    # Create a test dataset using these lists
    test_dataset = LungDataset(test_image_paths, test_label_paths)
    print(f"Test set contains {len(test_dataset)} volumes (last 20% of training set)")
    
    # 3. Load the pretrained model
    print(f"Loading pretrained model from: {model_path}")
    model = VNet(in_channels=1, out_channels=1, base_filters=16).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. Loop over all volumes in the test set and perform sliding-window inference
    all_dice = []
    all_hd = []
    
    for idx in range(len(test_dataset)):
        (vol, lbl), (img_path, lbl_path) = test_dataset[idx]
        file_name = os.path.basename(img_path)
        print(f"\n[Test {idx+1}/{len(test_dataset)}] Processing {file_name}")
        
        # vol and lbl are NumPy arrays with shape (D,H,W)
        prob_map = sliding_window_inference(model, vol, patch_size, step_size, device=device)
        pred_mask = (prob_map > threshold).astype(np.uint8)
        
        lbl_bin = (lbl > 0.5).astype(np.uint8)
        dice_val = dice_coefficient(pred_mask, lbl_bin)
        hd_val = hausdorff_distance(pred_mask, lbl_bin)
        all_dice.append(dice_val)
        all_hd.append(hd_val)
        
        print(f"  Dice = {dice_val:.4f} | Hausdorff = {hd_val:.2f}")
    
    # 5. Print overall test metrics
    if all_dice:
        print("\n==== Overall Test Set Metrics ====")
        print(f"Mean Dice = {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
        print(f"Mean Hausdorff = {np.mean(all_hd):.2f} ± {np.std(all_hd):.2f}")

if __name__ == "__main__":
    main()
