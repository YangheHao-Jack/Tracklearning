import os
import glob
import time
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ToTensord,
    Compose,
)
from monai.networks.nets import VNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot
from tqdm import tqdm  # Import tqdm for progress bars

# Data preparation (same as before)
images_dir = "/home/yanghehao/Task06_Lung/imagesTr"
labels_dir = "/home/yanghehao/Task06_Lung/labelsTr"
data_dicts = [
    {"image": img, "label": lbl}
    for img, lbl in zip(
        sorted(glob.glob(os.path.join(images_dir, "*.nii.gz"))),
        sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    )
]
train_split = 0.8
num_train = int(len(data_dicts) * train_split)
train_files = data_dicts[:num_train]
val_files = data_dicts[num_train:]

# Define training transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=600, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(112, 112, 96),
        pos=1,
        neg=1,
        num_samples=4,
        image_key="image",
        image_threshold=0
    ),
    ToTensord(keys=["image", "label"]),
])

# Define validation transforms
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=600, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(112, 112, 96),
        pos=1,
        neg=1,
        num_samples=4,
        image_key="image",
        image_threshold=0
    ),
    ToTensord(keys=["image", "label"]),
])

# Create datasets and dataloaders
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

# Model, loss function, and optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Two output channels for binary segmentation
model = VNet(spatial_dims=3, in_channels=1, out_channels=2).to(device)

# Use DiceLoss with softmax and one-hot encoding for the target labels
dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop parameters
max_epochs = 5
val_interval = 1
best_metric = -1
best_metric_epoch = -1
dice_metric = DiceMetric(include_background=True, reduction="mean")

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    # Use tqdm to display the training progress bar for each batch
    for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} Training", unit="batch"):
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)  # Labels remain as integers (0 or 1)
        optimizer.zero_grad()
        outputs = model(inputs)  # Output shape: (B, 2, D, H, W)
        loss = dice_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}")
    
    # Validation step every 'val_interval' epochs
    if (epoch+1) % val_interval == 0:
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            # Use tqdm to display the validation progress bar
            for val_data in tqdm(val_loader, desc="Validation", unit="batch"):
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                val_outputs = model(val_inputs)
                # Apply softmax then argmax to obtain predicted class labels
                val_outputs = torch.softmax(val_outputs, dim=1)
                pred_labels = torch.argmax(val_outputs, dim=1, keepdim=True)
                dice_metric(y_pred=pred_labels, y=val_labels)
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        print(f"Validation Dice Score: {metric:.4f}")
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model.pth")
            print("Saved new best metric model")

print(f"Training completed. Best validation Dice Score: {best_metric:.4f} at epoch {best_metric_epoch}")

# Hausdorff Distance Evaluation using softmax predictions
hausdorff_metric = HausdorffDistanceMetric(include_background=True)
hd_scores = []
model.eval()
with torch.no_grad():
    # Use tqdm to display progress for Hausdorff distance evaluation
    for val_data in tqdm(val_loader, desc="Hausdorff Evaluation", unit="batch"):
        val_inputs = val_data["image"].to(device)
        val_labels = val_data["label"].to(device)
        val_outputs = model(val_inputs)
        val_outputs = torch.softmax(val_outputs, dim=1)
        pred_labels = torch.argmax(val_outputs, dim=1, keepdim=True)
        hd_score = hausdorff_metric(y_pred=pred_labels, y=val_labels)
        hd_scores.append(hd_score.mean().item())
avg_hd = np.mean(hd_scores)
print(f"Average Hausdorff Distance on validation set: {avg_hd:.4f}")
