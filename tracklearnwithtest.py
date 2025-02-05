import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms import InterpolationMode

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------------------------
# 1. Dataset and Transforms
# ------------------------------

class ResizeAndToTensor:
    """
    Custom transform to resize images and masks to a fixed size and convert them to tensors.
    """
    def __init__(self, size=(512, 512)):
        self.size = size  # (height, width)

    def __call__(self, image, mask):
        """
        Args:
            image (PIL Image): Input image.
            mask (np.array): Corresponding mask as a NumPy array.
        
        Returns:
            image_tensor (torch.Tensor): Resized image tensor.
            mask_tensor (torch.Tensor): Resized mask tensor.
        """
        # Resize image using bilinear interpolation
        image = resize(
            image,
            self.size,
            interpolation=InterpolationMode.BILINEAR
        )
        # Resize mask using nearest-neighbor interpolation to preserve label integrity
        mask_pil = Image.fromarray(mask)
        mask_pil = resize(
            mask_pil,
            self.size,
            interpolation=InterpolationMode.NEAREST
        )
        # Convert back to NumPy array
        mask = np.array(mask_pil, dtype=np.uint8)

        # Convert image and mask to tensors
        image_tensor = to_tensor(image)             # [C, H, W], float32 in [0,1]
        mask_tensor  = torch.from_numpy(mask).long()# [H, W], dtype=torch.int64

        return image_tensor, mask_tensor

class BinarySegDataset(Dataset):
    """
    Custom Dataset for binary image segmentation.
    """
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        """
        Args:
            images_dir (str): Directory with input images.
            masks_dir (str): Directory with corresponding mask files (.npy).
            file_list (list): List of image filenames.
            transform (callable, optional): Transform to be applied on a sample.
        """
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
        """
        Retrieves the image and mask pair at the specified index.
        """
        # Debug: Uncomment for detailed logs
        # print(f"[Dataset] __getitem__ called with idx={idx}")
        
        img_filename = self.file_list[idx]
        base_name = os.path.splitext(img_filename)[0]

        # Load image
        img_path = os.path.join(self.images_dir, img_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"[Error] Image file not found: {img_path}")
        image = Image.open(img_path).convert("RGB")

        # Load mask
        mask_path = os.path.join(self.masks_dir, base_name + ".npy")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"[Error] Mask file not found: {mask_path}")
        mask = np.load(mask_path)

        # Remap any label >1 to 1 to ensure binary segmentation
        mask = np.where(mask > 1, 1, mask).astype(np.uint8)

        # Apply transformations if any
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

# ------------------------------
# 2. U-Net Model Definition
# ------------------------------

class DoubleConv(nn.Module):
    """
    A block consisting of two convolutional layers each followed by BatchNorm and ReLU.
    """
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
    """
    U-Net architecture for image segmentation.
    """
    def __init__(self, in_channels=3, out_channels=2):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            out_channels (int): Number of output channels/classes.
        """
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
        """
        Forward pass through the U-Net.
        """
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

# ------------------------------
# 3. Training and Validation Functions
# ------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx):
    """
    Trains the model for one epoch.
    
    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        device (torch.device): Device to run on.
        epoch_idx (int): Current epoch index.
    
    Returns:
        float: Average training loss for the epoch.
    """
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
    """
    Validates the model for one epoch.
    
    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the validation set.
        criterion (Loss): Loss function.
        device (torch.device): Device to run on.
        epoch_idx (int): Current epoch index.
    
    Returns:
        float: Average validation loss for the epoch.
    """
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

# ------------------------------
# 4. Testing and Metrics
# ------------------------------

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Computes the Dice coefficient for binary masks.
    
    Args:
        pred (torch.Tensor): Predicted mask [H, W].
        target (torch.Tensor): Ground truth mask [H, W].
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: Dice coefficient.
    """
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()

def test_segmentation(model, loader, device):
    """
    Evaluates the model on the test set and computes the average Dice coefficient.
    
    Args:
        model (nn.Module): The trained segmentation model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run on.
    
    Returns:
        float: Average Dice coefficient on the test set.
    """
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Testing"):
            images = images.to(device)
            masks  = masks.to(device)  # [B, H, W]

            outputs = model(images)    # [B, 2, H, W]
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            for pred, mask in zip(preds, masks):
                dice = dice_coefficient(pred, mask)
                dice_scores.append(dice)
    
    mean_dice = np.mean(dice_scores) if dice_scores else 0
    return mean_dice

def visualize_predictions(model, dataset, device, num_samples=3):
    """
    Visualizes predictions alongside the original images and ground truth masks.
    
    Args:
        model (nn.Module): The trained segmentation model.
        dataset (Dataset): The test dataset.
        device (torch.device): Device to run on.
        num_samples (int): Number of samples to visualize.
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)  # [1, C, H, W]
        
        with torch.no_grad():
            output = model(image_batch)  # [1, 2, H, W]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]
        
        image_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        mask_np = mask.cpu().numpy()  # [H, W]
        
        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Handle image range for visualization
        if image_np.max() > 1:
            image_np = image_np.astype(np.uint8)
        else:
            image_np = (image_np * 255).astype(np.uint8)
        
        axs[0].imshow(image_np)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")
        
        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")
        
        plt.tight_layout()
        plt.show()

# ------------------------------
# 5. Main Function
# ------------------------------

def main():
    # ------------------------------
    # Paths Configuration
    # ------------------------------
    # Define your directories here
    # Training and Validation directories (split from the same folder)
    images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/images/"
    masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_train/masks/"

    # Testing directories (completely separate)
    test_images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/images/"
    test_masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_test/masks/"

    # Hyperparameters
    batch_size    = 4
    num_epochs    = 10
    learning_rate = 1e-4

    # ------------------------------
    # Device Configuration
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Main] Using device:", device)

    # ------------------------------
    # Collect File Lists
    # ------------------------------
    all_train_images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_train_images)} training image files in {images_dir}")

    if len(all_train_images) == 0:
        print("[Error] No training image files found. Check your training path!")
        return

    all_train_masks = sorted([
        f for f in os.listdir(masks_dir)
        if f.lower().endswith(".npy")
    ])
    print(f"[Main] Found {len(all_train_masks)} training mask files in {masks_dir}")

    if len(all_train_masks) == 0:
        print("[Error] No training mask files found. Check your training mask path!")
        return

    # Ensure that the number of images and masks match
    if len(all_train_images) != len(all_train_masks):
        print("[Error] Number of training images and masks do not match!")
        return

    # ------------------------------
    # Train/Validation Split
    # ------------------------------
    train_files, val_files = train_test_split(
        all_train_images, 
        test_size=0.2,  # 20% for validation
        random_state=42
    )
    print(f"[Main] Training samples: {len(train_files)}")
    print(f"[Main] Validation samples: {len(val_files)}")

    # ------------------------------
    # Create Datasets
    # ------------------------------
    transform = ResizeAndToTensor(size=(512, 512))

    train_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        file_list=train_files,
        transform=transform
    )
    val_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        file_list=val_files,
        transform=transform
    )

    # ------------------------------
    # Create DataLoaders
    # ------------------------------
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )

    # ------------------------------
    # Create Test Dataset and DataLoader
    # ------------------------------
    all_test_images = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_test_images)} test image files in {test_images_dir}")

    if len(all_test_images) == 0:
        print("[Error] No test image files found. Check your test path!")
        return

    all_test_masks = sorted([
        f for f in os.listdir(test_masks_dir)
        if f.lower().endswith(".npy")
    ])
    print(f"[Main] Found {len(all_test_masks)} test mask files in {test_masks_dir}")

    if len(all_test_masks) == 0:
        print("[Error] No test mask files found. Check your test mask path!")
        return

    # Ensure that the number of test images and masks match
    if len(all_test_images) != len(all_test_masks):
        print("[Error] Number of test images and masks do not match!")
        return

    test_dataset = BinarySegDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        file_list=all_test_images,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"[Main] Test samples: {len(test_dataset)}")

    # ------------------------------
    # Initialize Model, Loss, Optimizer, Scheduler
    # ------------------------------
    model = UNet(in_channels=3, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )

    # ------------------------------
    # Training Loop
    # ------------------------------
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss   = validate_one_epoch(model, val_loader, criterion, device, epoch)
        
        # Step scheduler with validation loss
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f">> Saved best model with Val Loss: {best_val_loss:.4f}")

    print(f"\nTraining complete. Best Validation Loss: {best_val_loss:.4f}")
    print("Best model saved as 'best_model.pth'.")

    # ------------------------------
    # Testing
    # ------------------------------
    print("\n>>> Loading best model for testing...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_dice = test_segmentation(model, test_loader, device)
    print(f"Test Dice Coefficient: {test_dice:.4f}")

    # ------------------------------
    # Visualization of Test Samples
    # ------------------------------
    if len(test_dataset) > 0:
        print("\n>>> Visualizing predictions on test samples...")
        visualize_predictions(model, test_dataset, device, num_samples=3)
    else:
        print("[Warning] No test samples available for visualization.")

if __name__ == "__main__":
    main()
