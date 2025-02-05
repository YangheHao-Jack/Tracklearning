import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms import InterpolationMode

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
class ResizeAndToTensor:
    def __init__(self, size=(512, 512)):
        self.size = size  # (height, width)

    def __call__(self, image, mask):
        # Debug: Print shape of image & mask before resize
       # print(f"[Transform] Original Image size: {image.size}, Mask shape: {mask.shape}")
        
        # Resize image
        image = resize(
            image,
            self.size,
            interpolation=InterpolationMode.BILINEAR
        )
        # Resize mask with NEAREST
        mask_pil = Image.fromarray(mask)
        mask_pil = resize(
            mask_pil,
            self.size,
            interpolation=InterpolationMode.NEAREST
        )
        # Convert back to NumPy
        mask = np.array(mask_pil, dtype=np.uint8)

        # Debug: Print shape after resize
        #print(f"[Transform] Resized to {self.size}, Mask shape: {mask.shape}")

        image_tensor = to_tensor(image)             # (3, H, W)
        mask_tensor  = torch.from_numpy(mask).long()# (H, W)
        return image_tensor, mask_tensor


class BinarySegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.file_list  = file_list
        self.transform  = transform

        # Debug: Print how many files in this split
        print(f"[Dataset] Initialized with {len(file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Debug: Print index access
        #print(f"[Dataset] __getitem__ called with idx={idx}")
        
        img_filename = self.file_list[idx]
        base_name = os.path.splitext(img_filename)[0]

        # Load image
        img_path = os.path.join(self.images_dir, img_filename)
        # Debug: Check if file really exists
        if not os.path.isfile(img_path):
            print(f"[Error] Image file not found: {img_path}")
        image = Image.open(img_path).convert("RGB")

        # Load mask
        mask_path = os.path.join(self.masks_dir, base_name + ".npy")
        # Debug: Check if mask file really exists
        if not os.path.isfile(mask_path):
            print(f"[Error] Mask file not found: {mask_path}")
        mask = np.load(mask_path)

        # Remap label >1 to 1
        mask[mask > 1] = 1

        # Apply transform
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


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

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        up_4 = self.up4(bn)
        merge4 = torch.cat([d4, up_4], dim=1)
        c4 = self.conv4(merge4)

        up_3 = self.up3(c4)
        merge3 = torch.cat([d3, up_3], dim=1)
        c3 = self.conv3(merge3)

        up_2 = self.up2(c3)
        merge2 = torch.cat([d2, up_2], dim=1)
        c2 = self.conv2(merge2)

        up_1 = self.up1(c2)
        merge1 = torch.cat([d1, up_1], dim=1)
        c1 = self.conv1(merge1)

        out = self.out(c1)
        return out


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx):
    model.train()
    total_loss = 0.0
    print(f"--- [Train] Starting epoch {epoch_idx+1} ---")
    # Use tqdm to show progress
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
    return total_loss / len(loader.dataset)

def validate_one_epoch(model, loader, criterion, device, epoch_idx):
    model.eval()
    total_loss = 0.0
    print(f"--- [Val] Starting epoch {epoch_idx+1} ---")
    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Validation", leave=False)):
        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)
        if batch_idx % 10 == 0:
            print(f"[Val] Batch {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss / len(loader.dataset)

def main():
    # Adjust these paths
    images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/images/"
    masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_train/masks/"
    
    
    batch_size = 4
    num_epochs = 20 # set small for debugging
    learning_rate = 1e-4
    
    # 1) Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Main] Using device:", device)

    # 2) Collect image files
    all_images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_images)} image files in {images_dir}")

    if len(all_images) == 0:
        print("[Error] No image files found. Check your path!")
        return
    all_masks = sorted([
        f for f in os.listdir(masks_dir)
        if f.lower().endswith((".npy"))
    ])
    print(f"[Main] Found {len(all_masks)} mask files in {masks_dir}")

    if len(all_masks) == 0:
        print("[Error] No mask files found. Check your path!")
        return


    # 3) Train-Val split
    train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42)

    transform = ResizeAndToTensor((512, 512))
    train_dataset = BinarySegDataset(images_dir, masks_dir, train_files, transform=transform)
    val_dataset   = BinarySegDataset(images_dir, masks_dir, val_files,   transform=transform)

    # 4) DataLoaders, use num_workers=0 to avoid multiprocessing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    if len(train_dataset) == 0:
        print("[Error] train_dataset is empty. Check file names/paths.")
        return
    if len(val_dataset) == 0:
        print("[Warning] val_dataset is empty. No validation will happen.")

    # 5) Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_train = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                        patience=2, verbose=True)
    scheduler_val   = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                        patience=3, verbose=True)

    # 6) Training
    print("[Main] Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss   = validate_one_epoch(model, val_loader, criterion, device, epoch)
        scheduler_train.step(train_loss)  # if train loss doesn't improve, reduce LR
        scheduler_val.step(val_loss)      # if val loss doesn't improve, reduce LR

        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 7) Save model
    torch.save(model.state_dict(), "catheter_seg_model_gpu_debug.pth")
    print("[Main] Training complete. Model saved as 'catheter_seg_model_gpu_debug.pth'.")


if __name__ == "__main__":
    main()
