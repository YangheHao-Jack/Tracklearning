import os
import glob
import torch
import numpy as np

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandSpatialCropd,
    EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import VNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from tqdm import tqdm

def main():
    """
    Example: 3D binary segmentation (background vs. cancer) using:
      - VNet (with 2 output channels)
      - Dice Loss
      - DiceMetric + HausdorffDistanceMetric (euclidean)
      - Patch-based training (RandSpatialCropd)
      - Sliding window inference for validation/testing
    """

    # ---------------------
    # 1) Basic Setup
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    max_epochs = 100
    val_interval = 5
    learning_rate = 1e-4
    train_batch_size = 2
    val_batch_size = 1
    patch_size = (128, 128, 64)  # Example 3D patch size
    num_classes = 2  # 0=background, 1=cancer

    # Paths (adjust to your data organization)
    images_dir = "imagesTr"   # directory with training image volumes
    labels_dir = "labelsTr"   # directory with training labels
    model_save_dir = "models"
    os.makedirs(model_save_dir, exist_ok=True)

    # ---------------------
    # 2) Collect Data
    # ---------------------
    # Suppose your training set has 63 volumes. You might also have 32 test volumes in another folder.
    # Here we assume imagesTr/labelsTr contain all 63 training volumes.
    # We'll do a train/val split (e.g., 53 for training, 10 for validation).
    # Then we assume you have a separate directory for test data, or 32 more volumes after that.

    # Example: if imagesTr has exactly 63 volumes, we read them here:
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    # data_dicts => [{"image": <path>, "label": <path>}, ...]
    data_dicts = [
        {"image": img, "label": seg}
        for img, seg in zip(image_paths, label_paths)
    ]

    # Let's do 53 for training, 10 for validation (adjust as needed)
    train_files = data_dicts[:53]
    val_files = data_dicts[53:63]
    # If you have an additional 32 files in another folder for testing,
    # handle them similarly. Otherwise, you can place them in the same folder
    # and split accordingly. For demonstration, let's assume a separate directory:
    # test_images_dir = "imagesTs"  # or some directory with 32 test volumes
    # test_label_dir = "labelsTs"   # if test labels are available
    # (But if you don't have test labels, you can skip the metric calculation.)

    # ---------------------
    # 3) Define Transforms
    # ---------------------
    # Training transforms (random cropping for patch-based training)
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Optional: Spacingd if you want to resample to e.g. (1.0, 1.0, 1.0)
        # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        # allow_smaller=True to match older behavior, avoid future warning
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=patch_size,
            random_size=False
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

    # Validation transforms (no random cropping)
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    # ---------------------
    # 4) Create Datasets & Loaders
    # ---------------------
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=2)

    # If you have a separate test set, define test_ds and test_loader similarly.

    # ---------------------
    # 5) Define Model (2 classes: background + cancer)
    # ---------------------
    # If you do not want any dropout, use None. 
    # (Some MONAI versions require a sequence if you pass a float.)
    model = VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes
    ).to(device)

    # ---------------------
    # 6) Loss & Optimizer
    # ---------------------
    # For binary (2-class) segmentation, we can still use DiceLoss with to_onehot_y=True + softmax=True
    loss_function = DiceLoss(
        to_onehot_y=True,
        softmax=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track best validation dice
    best_val_dice = -1.0

    # ---------------------
    # 7) Training Loop
    # ---------------------
    for epoch in range(max_epochs):
        print(f"Epoch [{epoch+1}/{max_epochs}]")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader, desc="Training"):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)  # shape: [B, 2, D, H, W]
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        epoch_loss /= step
        print(f"  Train loss: {epoch_loss:.4f}")

        # ---------------------
        # 8) Validation
        # ---------------------
        if (epoch + 1) % val_interval == 0:
            model.eval()

            # Initialize metrics
            dice_metric = DiceMetric(
                include_background=False,  # we only care about label=1 (cancer)
                reduction="mean"
            )
            hausdorff_metric = HausdorffDistanceMetric(
                include_background=False,
                distance_metric="euclidean",  # NOT "euclid"
                reduction="mean"
            )

            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validation"):
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    # Use sliding window inference for large volumes
                    val_outputs = sliding_window_inference(
                        inputs=val_images,
                        roi_size=patch_size,
                        sw_batch_size=1,
                        predictor=model
                    )
                    val_outputs = torch.softmax(val_outputs, dim=1)
                    val_pred = torch.argmax(val_outputs, dim=1, keepdim=True)

                    # Accumulate Dice
                    dice_metric(y_pred=val_pred, y=val_labels)

                    # Accumulate Hausdorff (convert to one-hot)
                    val_pred_onehot = one_hot(val_pred, num_classes=num_classes)
                    val_labels_onehot = one_hot(val_labels, num_classes=num_classes)
                    hausdorff_metric(y_pred=val_pred_onehot, y=val_labels_onehot)

            # Aggregate metrics
            mean_dice = dice_metric.aggregate().item()
            mean_hausdorff = hausdorff_metric.aggregate().item()

            # Reset metrics if used again
            dice_metric.reset()
            hausdorff_metric.reset()

            print(f"  Validation Mean Dice (Cancer): {mean_dice:.4f}")
            print(f"  Validation Mean Hausdorff (Cancer): {mean_hausdorff:.4f}")

            # Save best model based on Dice
            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                best_model_path = os.path.join(model_save_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best model (Dice={best_val_dice:.4f}) saved to: {best_model_path}")

    # ---------------------
    # 9) (Optional) Test / Final Evaluation
    # ---------------------
    # If you have a separate test set, define it similarly with val_transforms and do:
    # test_loader = ...
    # Then load best_model and evaluate:
    """
    print("Testing on the held-out test set...")
    best_model_path = os.path.join(model_save_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        dice_metric_test = DiceMetric(
            include_background=False,
            reduction="mean"
        )
        hausdorff_metric_test = HausdorffDistanceMetric(
            include_background=False,
            distance_metric="euclidean",
            reduction="mean"
        )

        with torch.no_grad():
            for test_data in tqdm(test_loader, desc="Testing"):
                test_images = test_data["image"].to(device)
                test_labels = test_data["label"].to(device)

                test_outputs = sliding_window_inference(
                    inputs=test_images,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model
                )
                test_outputs = torch.softmax(test_outputs, dim=1)
                test_pred = torch.argmax(test_outputs, dim=1, keepdim=True)

                dice_metric_test(y_pred=test_pred, y=test_labels)

                test_pred_onehot = one_hot(test_pred, num_classes=num_classes)
                test_labels_onehot = one_hot(test_labels, num_classes=num_classes)
                hausdorff_metric_test(y_pred=test_pred_onehot, y=test_labels_onehot)

        mean_dice_test = dice_metric_test.aggregate().item()
        mean_hausdorff_test = hausdorff_metric_test.aggregate().item()
        dice_metric_test.reset()
        hausdorff_metric_test.reset()

        print(f"Test Mean Dice (Cancer): {mean_dice_test:.4f}")
        print(f"Test Mean Hausdorff (Cancer): {mean_hausdorff_test:.4f}")
    else:
        print("No saved model found. Skipping test.")
    """

if __name__ == "__main__":
    main()
