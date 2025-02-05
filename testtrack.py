import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import torchvision.models as models  # For ResNet encoder

from tqdm import tqdm

import torchmetrics
import random

# ------------------------------
# 0. Reproducibility (Optional)
# ------------------------------

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# 1. Dataset and Transforms
# ------------------------------

class ResizeAndToTensor:
    """
    Custom transform to resize images and masks to a fixed size and convert them to tensors.
    """
    def __init__(self, size=(512, 512), augment=False):
        self.size = size  # (height, width)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, image, mask):
        """
        Args:
            image (PIL Image): Input image.
            mask (np.array or PIL Image): Corresponding mask as a NumPy array or PIL Image.
        
        Returns:
            image_tensor (torch.Tensor): Resized and normalized image tensor.
            mask_tensor (torch.Tensor): Resized mask tensor (Long type).
        """
        # Resize image using bilinear interpolation
        image = resize(
            image,
            self.size,
            interpolation=InterpolationMode.BILINEAR
        )
        # Resize mask using nearest-neighbor interpolation to preserve label integrity
        if isinstance(mask, np.ndarray):
            mask_pil = Image.fromarray(mask)
        elif isinstance(mask, Image.Image):
            mask_pil = mask
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")
        
        mask_pil = resize(
            mask_pil,
            self.size,
            interpolation=InterpolationMode.NEAREST
        )
        # Convert back to NumPy array if it was initially
        if isinstance(mask, np.ndarray):
            mask = np.array(mask_pil, dtype=np.uint8)
        else:
            mask = np.array(mask_pil, dtype=np.uint8)

        # Convert image and mask to tensors
        image_tensor = to_tensor(image)             # [C, H, W], float32 in [0,1]
        image_tensor = self.normalize(image_tensor)

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
            masks_dir (str): Directory with corresponding mask files (.npy or image files).
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
        # Attempt to find mask with the same base name and supported extensions
        mask_extensions = ['.npy', '.png', '.jpg', '.jpeg']
        mask_path = None
        for ext in mask_extensions:
            potential_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.isfile(potential_path):
                mask_path = potential_path
                break
        if mask_path is None:
            raise FileNotFoundError(f"[Error] Mask file not found for image: {img_filename}")

        # Load mask
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
            mask = Image.fromarray(mask)
        else:
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        # Remap any label >1 to 1 to ensure binary segmentation
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 1, 1, mask_np).astype(np.uint8)
        mask = Image.fromarray(mask_np)

        # Apply transformations if any
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

# ------------------------------
# 2. U-Net Model Definition with ResNet Encoder
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

class ResNetUNet(nn.Module):
    """
    U-Net architecture with ResNet encoder.
    """
    def __init__(self, n_classes=2, encoder_name='resnet34', pretrained=True):
        """
        Args:
            n_classes (int): Number of output classes.
            encoder_name (str): Name of the ResNet encoder to use ('resnet18', 'resnet34', 'resnet50', etc.).
            pretrained (bool): Whether to use pretrained ResNet weights.
        """
        super(ResNetUNet, self).__init__()
        
        # Initialize ResNet encoder
        if encoder_name == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet34':
            self.encoder = models.resnet34(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif encoder_name == 'resnet101':
            self.encoder = models.resnet101(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError("Unsupported ResNet variant")

        # Encoder layers
        self.initial = nn.Sequential(
            self.encoder.conv1,  # [B, 64, H/2, W/2]
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool  # [B, 64, H/4, W/4]
        )
        self.encoder_layer1 = self.encoder.layer1  # [B, 64 or 256, H/4, W/4]
        self.encoder_layer2 = self.encoder.layer2  # [B, 128 or 512, H/8, W/8]
        self.encoder_layer3 = self.encoder.layer3  # [B, 256 or 1024, H/16, W/16]
        self.encoder_layer4 = self.encoder.layer4  # [B, 512 or 2048, H/32, W/32]

        # Decoder layers
        self.up4 = nn.ConvTranspose2d(encoder_channels[4], encoder_channels[3], kernel_size=2, stride=2)
        self.conv4 = DoubleConv(encoder_channels[3] + encoder_channels[3], encoder_channels[3])

        self.up3 = nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], kernel_size=2, stride=2)
        self.conv3 = DoubleConv(encoder_channels[2] + encoder_channels[2], encoder_channels[2])

        self.up2 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], kernel_size=2, stride=2)
        self.conv2 = DoubleConv(encoder_channels[1] + encoder_channels[1], encoder_channels[1])

        # Removed up1 and conv1 to prevent spatial dimension mismatch
        # If you need more upsampling steps, ensure corresponding encoder layers are present

        self.out_conv = nn.Conv2d(encoder_channels[0], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.initial(x)           # [B, 64, H/4, W/4]
        x1 = self.encoder_layer1(x0)   # [B, 64, H/4, W/4] for resnet34
        x2 = self.encoder_layer2(x1)   # [B, 128, H/8, W/8]
        x3 = self.encoder_layer3(x2)   # [B, 256, H/16, W/16]
        x4 = self.encoder_layer4(x3)   # [B, 512, H/32, W/32]

        # Decoder
        up4 = self.up4(x4)             # [B, 256, H/16, W/16]
        merge4 = torch.cat([up4, x3], dim=1)  # [B, 512, H/16, W/16]
        conv4 = self.conv4(merge4)     # [B, 256, H/16, W/16]

        up3 = self.up3(conv4)          # [B, 128, H/8, W/8]
        merge3 = torch.cat([up3, x2], dim=1)  # [B, 256, H/8, W/8]
        conv3 = self.conv3(merge3)     # [B, 128, H/8, W/8]

        up2 = self.up2(conv3)          # [B, 64, H/4, W/4]
        merge2 = torch.cat([up2, x1], dim=1)  # [B, 128, H/4, W/4]
        conv2 = self.conv2(merge2)     # [B, 64, H/4, W/4]

        out = self.out_conv(conv2)     # [B, n_classes, H/4, W/4]

        # Upsample to original size
        out = nn.functional.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)  # [B, n_classes, H, W]

        return out

# ------------------------------
# 6. Testing and Metrics
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

def test_segmentation(model, loader, device, metrics=None):
    """
    Evaluates the model on the test set and computes the average Dice coefficient and other metrics.
    
    Args:
        model (nn.Module): The trained segmentation model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run on.
        metrics (dict, optional): Dictionary of TorchMetrics.
    
    Returns:
        dict: Dictionary of average metrics on the test set.
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Testing", leave=False)):
            images = images.to(device)
            masks = masks.to(device)  # [B, H, W]

            outputs = model(images)    # [B, n_classes, H, W]
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # Compute metrics
            if metrics:
                for metric in metrics.values():
                    metric(preds, masks)

    # Compute metric values
    if metrics:
        metric_values = {k: v.compute().item() for k, v in metrics.items()}
        # Reset metrics
        for metric in metrics.values():
            metric.reset()
    else:
        # If metrics are not provided, compute only Dice
        dice_scores = []
        for pred, mask in zip(preds, masks):
            dice = dice_coefficient(pred, mask)
            dice_scores.append(dice)
        metric_values = {
            'dice': np.mean(dice_scores) if dice_scores else 0
        }

    # Print metrics
    if metrics:
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metric_values.items()])
        print(f"--- [Test] Metrics: {metric_str} ---")
    else:
        print(f"--- [Test] Dice Coefficient: {metric_values['dice']:.4f} ---")

    return metric_values

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
            output = model(image_batch)  # [1, n_classes, H, W]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]

        image_np = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        mask_np = mask.cpu().numpy()  # [H, W]

        # Handle image normalization for visualization
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
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
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

# ------------------------------
# 7. Main Function
# ------------------------------

def main():
    # ------------------------------
    # Define Your Directories Here
    # ------------------------------
    # Testing directories
    test_images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/images/"
    test_masks_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_test/masks/"

    # Path to the saved best model
    model_path = "best_model.pth"

    # ------------------------------
    # Hyperparameters
    # ------------------------------
    batch_size    = 4
    num_samples_to_visualize = 3

    # ------------------------------
    # Device Configuration
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # ------------------------------
    # Collect Test File Lists
    # ------------------------------
    # List all test image files
    all_test_images = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"[Main] Found {len(all_test_images)} test image files in {test_images_dir}")

    if len(all_test_images) == 0:
        print("[Error] No test image files found. Check your test path!")
        return

    # Ensure corresponding mask files exist
    all_test_images = sorted([
        f for f in all_test_images
        if any(
            os.path.isfile(os.path.join(test_masks_dir, os.path.splitext(f)[0] + ext))
            for ext in ['.npy', '.png', '.jpg', '.jpeg']
        )
    ])
    print(f"[Main] {len(all_test_images)} test images have corresponding masks.")

    if len(all_test_images) == 0:
        print("[Error] No test mask files found or mismatched filenames. Check your test mask path!")
        return

    # ------------------------------
    # Create Test Dataset and DataLoader
    # ------------------------------
    test_transform = ResizeAndToTensor(size=(512, 512), augment=False)

    test_dataset = BinarySegDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        file_list=all_test_images,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"[Main] Test samples: {len(test_dataset)}")

    # ------------------------------
    # Initialize Model
    # ------------------------------
    model = ResNetUNet(n_classes=2, encoder_name='resnet34', pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Main] Loaded model from {model_path}")

    # ------------------------------
    # Initialize Metrics
    # ------------------------------
    metrics_test = {
        'dice': torchmetrics.Dice(num_classes=2, average='macro').to(device),
        'iou': torchmetrics.JaccardIndex(task='binary').to(device),
        'accuracy': torchmetrics.Accuracy(task='binary').to(device)
    }

    # ------------------------------
    # Testing
    # ------------------------------
    test_metrics = test_segmentation(model, test_loader, device, metrics=metrics_test)
    print(f"Test Metrics: {test_metrics}")

    # ------------------------------
    # Visualization of Test Samples
    # ------------------------------
    if len(test_dataset) > 0:
        print("\n>>> Visualizing predictions on test samples...")
        visualize_predictions(model, test_dataset, device, num_samples=num_samples_to_visualize)
    else:
        print("[Warning] No test samples available for visualization.")

# ------------------------------
# 8. Run the Main Function
# ------------------------------

if __name__ == "__main__":
    main()
