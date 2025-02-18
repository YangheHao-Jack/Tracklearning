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
from torch.amp import GradScaler, autocast

###############################################
# Global Cache Directories for Images and Labels
###############################################
IMAGE_CACHE_DIR = "/media/yanghehao/新加卷3/Task06_Lung/cache/images"
LABEL_CACHE_DIR = "/media/yanghehao/新加卷3/Task06_Lung/cache/labels"
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
os.makedirs(LABEL_CACHE_DIR, exist_ok=True)

###############################################
# 1. Utility Functions with Separate Caching
###############################################
def get_cache_path(path, cache_dir):
    """
    Generate a cache filename for a given volume path inside the specified cache directory.
    """
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    # Handle .nii.gz extension: remove .nii.gz and replace with .npy
    if ext == '.gz':
        name, _ = os.path.splitext(name)
    cache_filename = name + ".npy"
    return os.path.join(cache_dir, cache_filename)

def load_volume_cached(path, cache_dir):
    """
    Load a volume using caching.
    If a .npy file exists in the specified cache_dir for the given .nii.gz file, load it.
    Otherwise, load the volume using SimpleITK, save it as .npy in cache_dir, then return it.
    """
    cache_path = get_cache_path(path, cache_dir)
    if os.path.exists(cache_path):
        volume = np.load(cache_path)
    else:
        sitk_img = sitk.ReadImage(path)
        volume = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        np.save(cache_path, volume)
        print(f"Converted and cached: {path} -> {cache_path}")
    return volume, None  # Returning None for the sitk image

def convert_all_files(input_dir, cache_dir):
    """
    Convert all .nii.gz files in input_dir to .npy files in cache_dir.
    If the .npy file already exists, skip the conversion.
    """
    files = sorted([os.path.join(input_dir, f)
                    for f in os.listdir(input_dir)
                    if not f.startswith('.') and f.endswith('.nii.gz')])
    for file in tqdm(files, desc=f"Converting files in {input_dir}"):
        cache_path = get_cache_path(file, cache_dir)
        if os.path.exists(cache_path):
            continue  # Skip conversion if already exists
        load_volume_cached(file, cache_dir)

###############################################
# 2. Patch Coordinate Extraction
###############################################
def extract_patch_coords(volume_shape, patch_size=(64, 64, 64), stride=(64, 64, 64)):
    """
    Compute patch coordinates for a volume.
    Returns a list of (start_z, start_y, start_x) tuples.
    """
    coords = []
    z, y, x = volume_shape
    pz, py, px = patch_size
    sz, sy, sx = stride
    for i in range(0, z - pz + 1, sz):
        for j in range(0, y - py + 1, sy):
            for k in range(0, x - px + 1, sx):
                coords.append((i, j, k))
    return coords

###############################################
# 3. Lazy Dataset Implementation with Separate Caching
###############################################
class LazyMedicalDataset(Dataset):
    """
    A lazy-loading dataset that loads each volume on-the-fly and extracts patches
    using precomputed indices. Images and labels are cached separately.
    """
    def __init__(self, image_dir, label_dir, patch_size=(64,64,64), stride=(64,64,64)):
        self.image_paths = sorted([os.path.join(image_dir, f) 
                                   for f in os.listdir(image_dir) 
                                   if not f.startswith('.') and f.endswith('.nii.gz')])
        self.label_paths = sorted([os.path.join(label_dir, f) 
                                   for f in os.listdir(label_dir) 
                                   if not f.startswith('.') and f.endswith('.nii.gz')])
        self.patch_size = patch_size
        self.stride = stride
        self.indices = []  # List of (image_idx, start_z, start_y, start_x)
        self.volume_shapes = []  # Cache volume shapes for each image
        for img_idx, img_path in enumerate(self.image_paths):
            # Load volume using cached version to get its shape (for images)
            img_array, _ = load_volume_cached(img_path, IMAGE_CACHE_DIR)
            self.volume_shapes.append(img_array.shape)
            coords = extract_patch_coords(img_array.shape, patch_size, stride)
            for coord in coords:
                self.indices.append((img_idx, *coord))
                
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_idx, i, j, k = self.indices[idx]
        # Load full image and label using separate caches
        img_array, _ = load_volume_cached(self.image_paths[img_idx], IMAGE_CACHE_DIR)
        lab_array, _ = load_volume_cached(self.label_paths[img_idx], LABEL_CACHE_DIR)
        pz, py, px = self.patch_size
        img_patch = img_array[i:i+pz, j:j+py, k:k+px]
        lab_patch = lab_array[i:i+pz, j:j+py, k:k+px]
        # Add channel dimension: image becomes (1, D, H, W); label becomes (1, D, H, W)
        img_patch = np.expand_dims(img_patch, axis=0)
        lab_patch = np.expand_dims(lab_patch, axis=0)
        return torch.tensor(img_patch), torch.tensor(lab_patch)

###############################################
# 4. VNet Model Components (Two-class Output)
###############################################
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

class InputTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputTransition, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        repeat_factor = out.size(1) // x.size(1)
        x_rep = x.repeat(1, repeat_factor, 1, 1, 1)
        out = self.relu(out + x_rep)
        return out

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

class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv, dropout=False):
        super(UpTransition, self).__init__()
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
        xcat = torch.cat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu(out + xcat)
        return out

class OutputTransition(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, n_classes, kernel_size=1)
        self.bn = nn.InstanceNorm3d(n_classes)
    def forward(self, x):
        out = self.conv1(x)
        return out

class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(n_channels, 16)
        self.down_tr32 = DownTransition(16, 1)
        self.down_tr64 = DownTransition(32, 2)
        self.down_tr128 = DownTransition(64, 3, dropout=True)
        self.down_tr256 = DownTransition(128, 2, dropout=True)
        
        self.up_tr256 = UpTransition(256, 256, 2, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1)
        self.up_tr32 = UpTransition(64, 32, 1)
        
        self.out_tr = OutputTransition(32, n_classes)
        
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

###############################################
# 5. Loss Function and Hausdorff Distance
###############################################
def dice_loss(pred, target, smooth=1.):
    """
    Compute Dice Loss for binary segmentation.
    
    Args:
        pred (torch.Tensor): Predicted probabilities (after sigmoid), shape (B, 1, D, H, W).
        target (torch.Tensor): Ground truth binary mask, shape (B, 1, D, H, W).
        smooth (float): Smoothing constant.
        
    Returns:
        torch.Tensor: Dice loss.
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
    Returns 0 if both are empty, or np.inf if one is empty.
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
# 6. Training, Validation, and Reconstruction Functions
###############################################
def train_model(model, dataloader, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    scaler = GradScaler('cuda')  # For mixed precision training
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False):
            images = images.to(device)
            # Squeeze channel dimension from labels: (B,1,D,H,W) -> (B,D,H,W)
            labels = labels.squeeze(1).to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(images)
                # Apply softmax then extract the foreground probability (channel 1)
                outputs = F.softmax(outputs, dim=1)[:, 1:2, ...]
                loss = dice_loss(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
            labels = labels.squeeze(1).to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)[:, 1:2, ...]
            preds = (outputs > 0.5).float()  # Binary prediction
            batch_size = preds.shape[0]
            for i in range(batch_size):
                pred_np = preds[i, 0].cpu().numpy()
                label_np = labels[i].cpu().numpy()
                dice = 1 - dice_loss(torch.tensor(pred_np), torch.tensor(label_np)).item()
                dice_scores.append(dice)
                hd = compute_hausdorff(pred_np, label_np)
                hausdorff_scores.append(hd)
    mean_dice = np.mean(dice_scores)
    mean_hd = np.mean([hd for hd in hausdorff_scores if not np.isinf(hd)])
    print(f"Validation Dice Score: {mean_dice:.4f}")
    print(f"Validation Hausdorff Distance: {mean_hd:.4f}")
    return mean_dice, mean_hd

def reconstruct_full_volume(model, full_image, patch_size=(64,64,64), stride=(64,64,64), device='cuda'):
    """
    Reconstruct the full segmentation using a sliding window approach.
    Overlapping patch predictions (after sigmoid) are averaged and then thresholded.
    """
    model.to(device)
    model.eval()
    full_image = full_image.astype(np.float32)
    vol_shape = full_image.shape
    prediction_volume = np.zeros(vol_shape, dtype=np.float32)
    count_volume = np.zeros(vol_shape, dtype=np.float32)
    coords = extract_patch_coords(vol_shape, patch_size, stride)
    
    with torch.no_grad():
        for (i, j, k) in tqdm(coords, total=len(coords), desc="Reconstructing full volume", leave=False):
            patch = full_image[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]]
            patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device)
            output = model(patch_tensor)
            output = torch.sigmoid(output).cpu().numpy()[0, 0]
            prediction_volume[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]] += output
            count_volume[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]] += 1
    count_volume[count_volume == 0] = 1
    averaged_prediction = prediction_volume / count_volume
    full_segmentation = (averaged_prediction > 0.5).astype(np.int32)
    return full_segmentation

def evaluate_full_volume(model, image_path, label_path, patch_size=(64,64,64), stride=(64,64,64), device='cuda'):
    full_image, _ = load_volume_cached(image_path, IMAGE_CACHE_DIR)
    full_label, _ = load_volume_cached(label_path, LABEL_CACHE_DIR)
    full_segmentation = reconstruct_full_volume(model, full_image, patch_size, stride, device)
    full_pred_target = (full_segmentation == 1).astype(np.float32)
    full_label_target = (full_label == 1).astype(np.float32)
    dice = 1 - dice_loss(torch.tensor(full_pred_target), torch.tensor(full_label_target)).item()
    hd = compute_hausdorff(full_pred_target, full_label_target)
    print(f"Full Volume Dice: {dice:.4f}")
    print(f"Full Volume Hausdorff Distance: {hd:.4f}")
    return dice, hd

###############################################
# 7. Main Execution: Convert Files Then Train
###############################################
if __name__ == "__main__":
    # Directories containing training images and labels
    image_dir = "/home/yanghehao/Task06_Lung/imagesTr"
    label_dir = "/home/yanghehao/Task06_Lung/labelsTr"
    
    # Step 1: Convert all nii.gz files to npy files beforehand
    print("Converting image files to npy format...")
    convert_all_files(image_dir, IMAGE_CACHE_DIR)
    print("Converting label files to npy format...")
    convert_all_files(label_dir, LABEL_CACHE_DIR)
    
    # Step 2: Create lazy dataset instance with patch size (64,64,64) and stride (64,64,64)
    lazy_dataset = LazyMedicalDataset(image_dir, label_dir, patch_size=(64,64,64), stride=(64,64,64))
    
    # Split dataset (e.g., 80/20 split)
    split_index = int(0.8 * len(lazy_dataset))
    train_dataset = torch.utils.data.Subset(lazy_dataset, list(range(split_index)))
    val_dataset = torch.utils.data.Subset(lazy_dataset, list(range(split_index, len(lazy_dataset))))
    
    # Use 2 DataLoader workers if system memory permits
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Initialize VNet with 2 output channels for binary segmentation.
    model = VNet(n_channels=1, n_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting training...")
    trained_model = train_model(model, train_loader, optimizer, num_epochs=50, device='cuda')
    
    print("Validating model on patches...")
    val_dice, val_hd = validate_model(trained_model, val_loader, device='cuda')
    
    # For full volume evaluation, specify a test volume (update filenames as needed)
    test_image_path = os.path.join(image_dir, "lung_001.nii.gz")
    test_label_path = os.path.join(label_dir, "lung_001.nii.gz")
    
    print("Evaluating full volume segmentation...")
    full_dice, full_hd = evaluate_full_volume(trained_model, test_image_path, test_label_path,
                                               patch_size=(64,64,64), stride=(64,64,64), device='cuda')
    
    print("Training complete.")
    print("Final Patch-based Validation Dice Score: {:.4f}".format(val_dice))
    print("Final Patch-based Validation Hausdorff Distance: {:.4f}".format(val_hd))
    print("Final Full Volume Dice Score: {:.4f}".format(full_dice))
    print("Final Full Volume Hausdorff Distance: {:.4f}".format(full_hd))
