import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

###############################################
# 1. Data Loading, Preprocessing and Patching
###############################################

def load_volume(path):
    """Load a 3D volume using SimpleITK."""
    sitk_img = sitk.ReadImage(path)
    img_array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    return img_array, sitk_img

def extract_patches(volume, patch_size=(128, 128, 128), stride=(64, 64, 64)):
    """
    Extract overlapping patches from a 3D volume.
    Returns a list of patches.
    """
    patches = []
    z, y, x = volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride
    for i in range(0, z - pz + 1, sz):
        for j in range(0, y - py + 1, sy):
            for k in range(0, x - px + 1, sx):
                patch = volume[i:i+pz, j:j+py, k:k+px]
                patches.append(patch)
    return patches

class MedicalDataset(Dataset):
    """
    Dataset for 3D medical images. Loads images and corresponding labels,
    partitions them into patches, and returns tensor pairs.
    """
    def __init__(self, image_dir, label_dir, patch_size=(128,128,128)):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.patch_size = patch_size
        self.data = []  # list of (image_patch, label_patch) tuples
        self.prepare_data()

    def prepare_data(self):
        for img_path, lab_path in zip(self.image_paths, self.label_paths):
            img, _ = load_volume(img_path)
            lab, _ = load_volume(lab_path)
            img_patches = extract_patches(img, patch_size=self.patch_size)
            lab_patches = extract_patches(lab, patch_size=self.patch_size)
            # Ensure same number of patches for image and label
            for ip, lp in zip(img_patches, lab_patches):
                self.data.append((ip, lp))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_patch, lab_patch = self.data[idx]
        # Add channel dimension (C, D, H, W)
        img_patch = np.expand_dims(img_patch, axis=0)
        lab_patch = np.expand_dims(lab_patch, axis=0)
        return torch.tensor(img_patch), torch.tensor(lab_patch)

###############################################
# 2. Full VNet Implementation
###############################################

# Basic convolution block used in VNet
class LUConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super(LUConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# Input transition layer: converts input volume to feature maps
class InputTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputTransition, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        # Replicate input channels to match out_channels for the residual connection
        repeat_factor = out.size(1) // x.size(1)
        x_rep = x.repeat(1, repeat_factor, 1, 1, 1)
        out = self.relu(out + x_rep)
        return out

# Down transition block: downsampling with residual LUConv blocks
class DownTransition(nn.Module):
    def __init__(self, in_channels, n_conv, dropout=False):
        super(DownTransition, self).__init__()
        out_channels = 2 * in_channels
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.do1 = nn.Dropout3d() if dropout else None
        ops = [LUConv(out_channels, out_channels) for _ in range(n_conv)]
        self.ops = nn.Sequential(*ops)
    def forward(self, x):
        down = self.relu(self.bn(self.down_conv(x)))
        if self.do1 is not None:
            down = self.do1(down)
        out = self.ops(down)
        out = self.relu(out + down)
        return out

# Up transition block: upsampling with transposed convolution and skip connections
class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv, dropout=False):
        super(UpTransition, self).__init__()
        # Transposed convolution reduces channels by half
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn = nn.InstanceNorm3d(out_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.do1 = nn.Dropout3d() if dropout else None
        self.do2 = nn.Dropout3d() if dropout else None
        ops = [LUConv(out_channels, out_channels) for _ in range(n_conv)]
        self.ops = nn.Sequential(*ops)
    def forward(self, x, skipx):
        if self.do1 is not None:
            x = self.do1(x)
            skipx = self.do2(skipx)
        out = self.relu(self.bn(self.up_conv(x)))
        # Concatenate along channel dimension
        xcat = torch.cat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu(out + xcat)
        return out

# Output transition: maps features to the number of classes
class OutputTransition(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, n_classes, kernel_size=1)
        self.bn = nn.InstanceNorm3d(n_classes)
    def forward(self, x):
        out = self.conv1(x)
        return out

# Complete VNet model combining all components
class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(n_channels, 16)
        self.down_tr32 = DownTransition(16, 1)              # 16 -> 32 channels
        self.down_tr64 = DownTransition(32, 2)              # 32 -> 64 channels
        self.down_tr128 = DownTransition(64, 3, dropout=True) # 64 -> 128 channels
        self.down_tr256 = DownTransition(128, 2, dropout=True) # 128 -> 256 channels
        
        self.up_tr256 = UpTransition(256, 256, 2, dropout=True) # up: 256 channels
        self.up_tr128 = UpTransition(256, 128, 2, dropout=True) # up: 128 channels
        self.up_tr64 = UpTransition(128, 64, 1)                 # up: 64 channels
        self.up_tr32 = UpTransition(64, 32, 1)                  # up: 32 channels
        
        self.out_tr = OutputTransition(32, n_classes)
        
    def forward(self, x):
        # Encoder path
        out16 = self.in_tr(x)         # (B,16,D,H,W)
        out32 = self.down_tr32(out16)   # (B,32, D/2,H/2,W/2)
        out64 = self.down_tr64(out32)   # (B,64, D/4,H/4,W/4)
        out128 = self.down_tr128(out64)  # (B,128, D/8,H/8,W/8)
        out256 = self.down_tr256(out128)  # (B,256, D/16,H/16,W/16)
        
        # Decoder path with skip connections
        up256 = self.up_tr256(out256, out128)
        up128 = self.up_tr128(up256, out64)
        up64 = self.up_tr64(up128, out32)
        up32 = self.up_tr32(up64, out16)
        
        out = self.out_tr(up32)
        return out

###############################################
# 3. Loss Function and Hausdorff Distance
###############################################

def dice_loss(pred, target, smooth=1.):
    """
    Compute Dice loss.
    This implementation flattens the volumes and calculates the Dice coefficient,
    then returns one minus the Dice coefficient.
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice_coeff

def compute_hausdorff(pred, target):
    """
    Compute the Hausdorff Distance between two binary volumes.
    Both pred and target should be numpy arrays.
    If both are empty, returns 0; if one is empty, returns np.inf.
    """
    pred_coords = np.argwhere(pred)
    target_coords = np.argwhere(target)
    
    if len(pred_coords) == 0 and len(target_coords) == 0:
        return 0.0
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return np.inf
    
    d1 = directed_hausdorff(pred_coords, target_coords)[0]
    d2 = directed_hausdorff(target_coords, pred_coords)[0]
    return max(d1, d2)

###############################################
# 4. Training and Validation Functions with tqdm
###############################################

def train_model(model, dataloader, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Use tqdm progress bar for each epoch
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Apply sigmoid activation for binary segmentation
            outputs = torch.sigmoid(outputs)
            loss = dice_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return model

def validate_model(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    dice_scores = []
    hausdorff_scores = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()
            # Compute Dice Score for each sample in the batch
            batch_size = preds.shape[0]
            for i in range(batch_size):
                pred_np = preds[i, 0].cpu().numpy()
                label_np = labels[i, 0].cpu().numpy()
                dice = 1 - dice_loss(torch.tensor(pred_np), torch.tensor(label_np)).item()
                dice_scores.append(dice)
                hd = compute_hausdorff(pred_np, label_np)
                hausdorff_scores.append(hd)
    mean_dice = np.mean(dice_scores)
    mean_hd = np.mean([hd for hd in hausdorff_scores if not np.isinf(hd)])
    print(f"Validation Dice Score: {mean_dice:.4f}")
    print(f"Validation Hausdorff Distance: {mean_hd:.4f}")
    return mean_dice, mean_hd

###############################################
# 5. Main Execution
###############################################

if __name__ == "__main__":
    # Directories containing training images and labels
    image_dir = "/home/yanghehao/Task06_Lung/imagesTr"
    label_dir = "/home/yanghehao/Task06_Lung/labelsTr"
    
    # Prepare the dataset and split into training and validation sets (80/20 split)
    full_dataset = MedicalDataset(image_dir, label_dir, patch_size=(128,128,128))
    split_index = int(0.8 * len(full_dataset))
    train_dataset = torch.utils.data.Subset(full_dataset, list(range(split_index)))
    val_dataset = torch.utils.data.Subset(full_dataset, list(range(split_index, len(full_dataset))))
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # Initialize the VNet model.
    # For binary segmentation, set n_classes=1 and apply sigmoid activation during training.
    model = VNet(n_channels=1, n_classes=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    print("Starting training...")
    trained_model = train_model(model, train_loader, optimizer, num_epochs=50, device='cuda')
    
    # Validate the model
    print("Validating model...")
    val_dice, val_hd = validate_model(trained_model, val_loader, device='cuda')
    
    # Final metrics are printed during training and validation.
    print("Training complete.")
    print("Final Validation Dice Score: {:.4f}".format(val_dice))
    print("Final Validation Hausdorff Distance: {:.4f}".format(val_hd))
    
    # Note: For a more detailed evaluation, consider reconstructing full volumes from patches
    # and computing the metrics on the full volume.
