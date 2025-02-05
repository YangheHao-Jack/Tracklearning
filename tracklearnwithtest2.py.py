import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp  # For pre-trained models
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchmetrics
from torch.cuda.amp import GradScaler, autocast

import cv2  # For post-processing

# ------------------------------
# 1. Dataset and Transforms
# ------------------------------

class SegmentationTransform:
    """
    Custom transform using Albumentations to apply augmentations and preprocessing.
    """
    def __init__(self, resize=(512, 512), augment=True):
        self.resize = resize
        self.augment = augment
        self.transform = self.get_transform()

    def get_transform(self):
        transforms = []
        if self.augment:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.ElasticTransform(p=0.5),
                A.GridDistortion(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                # Add more augmentations as needed
            ])
        transforms.extend([
            A.Resize(height=self.resize[0], width=self.resize[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406),  # Using ImageNet means
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        return A.Compose(transforms)

    def __call__(self, image, mask):
        """
        Args:
            image (PIL Image): Input image.
            mask (np.array): Corresponding mask as a NumPy array.

        Returns:
            image_tensor (torch.Tensor): Transformed image tensor.
            mask_tensor (torch.Tensor): Transformed mask tensor (Long type).
        """
        augmented = self.transform(image=np.array(image), mask=np.array(mask))
        image = augmented['image']
        mask = augmented['mask']

        # Ensure the mask is of type Long (torch.int64)
        mask = mask.long()

        return image, mask


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

def get_model(pretrained=True):
    """
    Initializes the U-Net model with a pre-trained encoder.

    Args:
        pretrained (bool): Whether to use pre-trained weights for the encoder.

    Returns:
        smp.Unet: Segmentation model.
    """
    model = smp.Unet(
        encoder_name="resnet34",        # Choose encoder, e.g., resnet34
        encoder_weights="imagenet" if pretrained else None,  # Use pre-trained weights
        in_channels=3,                  # Input channels (RGB)
        classes=2,                      # Number of output classes
        activation=None                 # We'll apply activation in loss or during evaluation
    )
    return model

# ------------------------------
# 3. Loss Functions
# ------------------------------

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        """
        Args:
            logits (torch.Tensor): Predicted logits [B, C, H, W].
            true (torch.Tensor): Ground truth masks [B, H, W].

        Returns:
            torch.Tensor: Dice loss.
        """
        probs = torch.softmax(logits, dim=1)
        true_one_hot = nn.functional.one_hot(true, num_classes=probs.shape[1])  # [B, H, W, C]
        true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (torch.sum(probs, dims) + torch.sum(true_one_hot, dims) + self.smooth)
        loss = 1 - dice.mean()
        return loss

class CombinedLoss(nn.Module):
    """
    Combines CrossEntropyLoss and DiceLoss.
    """
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, true):
        """
        Args:
            logits (torch.Tensor): Predicted logits [B, C, H, W].
            true (torch.Tensor): Ground truth masks [B, H, W].

        Returns:
            torch.Tensor: Combined loss.
        """
        loss_ce = self.ce(logits, true)
        loss_dice = self.dice(logits, true)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice

# ------------------------------
# 4. Training and Validation Functions
# ------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx, scaler, clip_grad=1.0):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): Optimizer.
        criterion (Loss): Loss function.
        device (torch.device): Device to run on.
        epoch_idx (int): Current epoch index.
        scaler (GradScaler): Gradient scaler for mixed precision.
        clip_grad (float): Maximum gradient norm for clipping.

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

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale gradients before clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        if (batch_idx + 1) % 10 == 0:
            print(f"[Train] Epoch {epoch_idx+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    return total_loss / len(loader.dataset)

def validate_one_epoch(model, loader, criterion, device, epoch_idx, metrics):
    """
    Validates the model for one epoch.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): DataLoader for the validation set.
        criterion (Loss): Loss function.
        device (torch.device): Device to run on.
        epoch_idx (int): Current epoch index.
        metrics (dict): Dictionary of TorchMetrics.

    Returns:
        tuple: (Average validation loss, Dice score, IoU score, Accuracy score)
    """
    model.eval()
    total_loss = 0.0
    print(f"--- [Val] Starting epoch {epoch_idx+1} ---")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Validation", leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            metrics['dice'](preds, masks)
            metrics['iou'](preds, masks)
            metrics['accuracy'](preds, masks)

            if (batch_idx + 1) % 10 == 0:
                print(f"[Val] Epoch {epoch_idx+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    epoch_loss = total_loss / len(loader.dataset)
    epoch_dice = metrics['dice'].compute().item()
    epoch_iou = metrics['iou'].compute().item()
    epoch_acc = metrics['accuracy'].compute().item()

    # Reset metrics after computation
    for metric in metrics.values():
        metric.reset()

    print(f"[Val] Epoch {epoch_idx+1}: Loss={epoch_loss:.4f}, Dice={epoch_dice:.4f}, IoU={epoch_iou:.4f}, Accuracy={epoch_acc:.4f}")
    return epoch_loss, epoch_dice, epoch_iou, epoch_acc

# ------------------------------
# 5. Testing and Metrics
# ------------------------------

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Computes the Dice coefficient for binary masks.

    Args:
        pred (np.array or torch.Tensor): Predicted mask [H, W].
        target (np.array or torch.Tensor): Ground truth mask [H, W].
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred_flat = pred.flatten().astype(np.float32)
    target_flat = target.flatten().astype(np.float32)

    intersection = np.sum(pred_flat * target_flat)
    dice = (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)
    return dice

def test_segmentation(model, loader, device, metrics=None, post_process_flag=False):
    """
    Evaluates the model on the test set and computes the average Dice coefficient.

    Args:
        model (nn.Module): The trained segmentation model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run on.
        metrics (dict, optional): Dictionary of TorchMetrics.
        post_process_flag (bool): Whether to apply post-processing to predictions.

    Returns:
        tuple: (Average Dice coefficient, IoU, Accuracy)
    """
    model.eval()
    dice_scores = []
    iou_scores = []
    accuracy_scores = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Testing", leave=False)):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)    # [B, C, H, W]
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # Apply post-processing if needed
            if post_process_flag:
                # Convert to numpy for morphological operations
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()
                preds_post = []
                for pred in preds_np:
                    pred = post_process_mask(pred)
                    preds_post.append(pred)
                preds = torch.tensor(preds_post).to(device)

            for pred, mask in zip(preds, masks):
                dice = dice_coefficient(pred, mask)
                dice_scores.append(dice)

                # If metrics are provided
                if metrics:
                    metrics['dice'](pred, mask)
                    metrics['iou'](pred, mask)
                    metrics['accuracy'](pred, mask)

    mean_dice = np.mean(dice_scores) if dice_scores else 0
    mean_iou = None
    mean_acc = None
    if metrics:
        mean_dice = metrics['dice'].compute().item()
        mean_iou = metrics['iou'].compute().item()
        mean_acc = metrics['accuracy'].compute().item()
        # Reset metrics
        for metric in metrics.values():
            metric.reset()

    return mean_dice, mean_iou, mean_acc

def post_process_mask(mask, kernel_size=3, iterations=1):
    """
    Applies morphological operations to refine the predicted mask.

    Args:
        mask (np.array): Predicted mask [H, W].
        kernel_size (int): Size of the morphological kernel.
        iterations (int): Number of times the operation is applied.

    Returns:
        np.array: Refined mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return mask

# ------------------------------
# 6. Visualization Function
# ------------------------------

def visualize_predictions(model, dataset, device, num_samples=3, post_process=False):
    """
    Visualizes predictions alongside the original images and ground truth masks.

    Args:
        model (nn.Module): The trained segmentation model.
        dataset (Dataset): The test dataset.
        device (torch.device): Device to run on.
        num_samples (int): Number of samples to visualize.
        post_process (bool): Whether to apply post-processing to predictions.
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)  # [1, C, H, W]

        with torch.no_grad():
            output = model(image_batch)  # [1, C, H, W]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]

        if post_process:
            pred = post_process_mask(pred)

        image_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        mask_np = mask.cpu().numpy()  # [H, W]

        # Handle image normalization for visualization
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image_np)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title("Predicted Mask" + (" (Post-Processed)" if post_process else ""))
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

# ------------------------------
# 7. Early Stopping Class
# ------------------------------

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0.0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"[EarlyStopping] Initial best loss: {self.best_loss:.4f}")
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("[EarlyStopping] Early stopping triggered.")
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Validation loss improved to: {self.best_loss:.4f}")

# ------------------------------
# 8. Main Function
# ------------------------------

def main():
    # ------------------------------
    # Define Your Directories Here
    # ------------------------------
    # Training and Validation directories (split from the same folder)
    images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/images/"
    masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_train/masks/"

    # Testing directories (completely separate)
    test_images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/images/"
    test_masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_test/masks/"

    # Hyperparameters
    batch_size    = 4
    num_epochs    = 20
    learning_rate = 1e-4
    val_split     = 0.2
    save_path     = "best_model.pth"
    patience      = 7
    weight_ce     = 1.0
    weight_dice   = 1.0
    post_process_flag = True  # Set to True to apply post-processing

    # ------------------------------
    # Device Configuration
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # ------------------------------
    # Collect Training File Lists
    # ------------------------------
    # List all training image files
    all_train_images = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_train_images)} training image files in {images_dir}")

    if len(all_train_images) == 0:
        print("[Error] No training image files found. Check your training path!")
        return

    # Ensure corresponding mask files exist
    all_train_images = sorted([
        f for f in all_train_images
        if os.path.isfile(os.path.join(masks_dir, os.path.splitext(f)[0] + ".npy"))
    ])
    print(f"[Main] {len(all_train_images)} training images have corresponding masks.")

    if len(all_train_images) == 0:
        print("[Error] No training mask files found or mismatched filenames. Check your training mask path!")
        return

    # List all test image files
    all_test_images = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_test_images)} test image files in {test_images_dir}")

    if len(all_test_images) == 0:
        print("[Error] No test image files found. Check your test path!")
        return

    # Ensure corresponding test mask files exist
    all_test_images = sorted([
        f for f in all_test_images
        if os.path.isfile(os.path.join(test_masks_dir, os.path.splitext(f)[0] + ".npy"))
    ])
    print(f"[Main] {len(all_test_images)} test images have corresponding masks.")

    if len(all_test_images) == 0:
        print("[Error] No test mask files found or mismatched filenames. Check your test mask path!")
        return

    # ------------------------------
    # Train/Validation Split
    # ------------------------------
    train_files, val_files = train_test_split(
        all_train_images,
        test_size=val_split,
        random_state=42
    )
    print(f"[Main] Training samples: {len(train_files)}")
    print(f"[Main] Validation samples: {len(val_files)}")

    # ------------------------------
    # Create Datasets with Transforms
    # ------------------------------
    train_transform = SegmentationTransform(resize=(512, 512), augment=True)
    val_transform = SegmentationTransform(resize=(512, 512), augment=False)
    test_transform = SegmentationTransform(resize=(512, 512), augment=False)

    train_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        file_list=train_files,
        transform=train_transform
    )
    val_dataset = BinarySegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        file_list=val_files,
        transform=val_transform
    )
    test_dataset = BinarySegDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        file_list=all_test_images,
        transform=test_transform
    )

    # ------------------------------
    # Create DataLoaders
    # ------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4  # Adjust based on your CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # ------------------------------
    # Initialize Model, Loss, Optimizer, Scheduler
    # ------------------------------
    model = get_model(pretrained=True).to(device)
    criterion = CombinedLoss(weight_ce=weight_ce, weight_dice=weight_dice)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize Metrics
    metrics_val = {
        'dice': torchmetrics.Dice(num_classes=2, average='macro').to(device),
        'iou': torchmetrics.JaccardIndex(num_classes=2, average='macro',task='binary').to(device),
        'accuracy': torchmetrics.Accuracy(task='binary').to(device)
    }

    metrics_test = {
        'dice': torchmetrics.Dice(num_classes=2, average='macro').to(device),
        'iou': torchmetrics.JaccardIndex(num_classes=2, average='macro',task='binary').to(device),
        'accuracy': torchmetrics.Accuracy(task='binary').to(device)
    }


 
    # ------------------------------
    # Training Loop
    # ------------------------------
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)

        # Validate
        val_loss, val_dice, val_iou, val_acc = validate_one_epoch(model, val_loader, criterion, device, epoch, metrics_val)

        # Scheduler step
        scheduler.step()


        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Val Acc: {val_acc:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f">> Saved best model with Val Loss: {best_val_loss:.4f}")
           
            early_stopping.counter = 0  # Reset early stopping counter
        else:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("[Main] Early stopping triggered. Stopping training.")
                break

    print(f"\n[Main] Training complete. Best Validation Loss: {best_val_loss:.4f}")
    print(f"[Main] Best model saved at: {save_path}")

  
    # ------------------------------
    # Testing
    # ------------------------------
    print("\n>>> Loading best model for testing...")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_dice, test_iou, test_acc = test_segmentation(model, test_loader, device, metrics_test, post_process_flag)
    print(f"Test Dice Coefficient: {test_dice:.4f}, Test IoU: {test_iou:.4f}, Test Accuracy: {test_acc:.4f}")

    # ------------------------------
    # Visualization
    # ------------------------------
    if len(test_dataset) > 0:
        print("\n>>> Visualizing predictions on test samples...")
        visualize_predictions(model, test_dataset, device, num_samples=10, post_process=post_process_flag)
    else:
        print("[Warning] No test samples available for visualization.")

if __name__ == "__main__":
    """
    Example usage:
    Ensure that the directories are correctly set at the beginning of the main() function.
    Simply run the script without additional command-line arguments:
    
    python train_evaluate_segmentation_improved.py
    """
    main()
