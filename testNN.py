#!/usr/bin/env python3
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from nnunetv2.inference.model_restore import restore_model

################################################
# 1) Load nnUNet model from a trained folder
################################################
def load_nnunet_model(model_folder):
    """
    model_folder: path to your nnUNet results, e.g.
      /home/.../nnUNet_results/Dataset101_2d/nnUNetTrainer__...
    """
    model = restore_model(model_folder)
    model.eval()
    model.cuda()
    return model

################################################
# 2) Preprocess image (2D grayscale)
################################################
def preprocess_image(image_path):
    # load as grayscale
    img_pil = Image.open(image_path).convert("L")
    img_arr = np.array(img_pil, dtype=np.float32)  # shape (H,W)
    # If training used [0..1] scaling, do: img_arr /= 255.0

    # add batch & channel dimension => (1,1,H,W)
    input_arr = img_arr[None, None]  # shape (1,1,H,W)
    return input_arr, img_arr  # also returning original 2D for display

################################################
# 3) Inference with nnUNet
################################################
def run_inference(nnunet_model, input_arr):
    """
    input_arr: shape (1,1,H,W)
    returns predicted mask (H,W)
    """
    with torch.no_grad():
        x_tensor = torch.from_numpy(input_arr).cuda()
        pred = nnunet_model(x_tensor)  # shape (1, num_classes, H, W)
        seg = torch.argmax(pred, dim=1).cpu().numpy()  # shape (1,H,W)
    return seg[0]

################################################
# 4) Load ground truth .npy => unify to {0,1}
################################################
def load_npy_mask(mask_path):
    mask_data = np.load(mask_path)
    # unify all non-zero to 1 if it's a single object class
    mask_01 = (mask_data != 0).astype(np.uint8)
    return mask_01

################################################
# 5) Create overlay for visualization
################################################
def overlay_prediction_on_image(orig_img2d, pred_mask):
    """
    orig_img2d: shape (H,W) single-channel
    pred_mask: shape (H,W), 1 => object
    Return an (H,W,3) overlay in RGB for plotting
    """
    # expand grayscale to 3 channels
    h, w = orig_img2d.shape
    orig_rgb = np.stack([orig_img2d]*3, axis=-1).astype(np.uint8)

    # color predicted => red
    overlay = orig_rgb.copy()
    overlay[pred_mask==1] = [255,0,0]  # pure red
    alpha = 0.4
    blended = (alpha * overlay + (1-alpha) * orig_rgb).astype(np.uint8)
    return blended

################################################
# 6) Main
################################################
def main():
    # A) Paths
    model_folder = "/home/yanghehao/nnUNet/nnunetv2/nnUNet_results/Dataset101_Phantom/nnUNetTrainer__nnUNetPlans__2d"
    test_image_dir = "/home/yanghehao/tracklearning/segmentation/phantom_test/images"
    test_mask_dir  = "/home/yanghehao/tracklearning/segmentation/phantom_test/masks"

    # B) Load model
    nnunet_model = load_nnunet_model(model_folder)

    # C) Gather all valid pairs by base name
    #    e.g. Phantom_005.png => Phantom_005.npy
    image_files = sorted([f for f in os.listdir(test_image_dir) if f.lower().endswith(".png")])
    mask_files  = sorted([f for f in os.listdir(test_mask_dir)  if f.lower().endswith(".npy")])

    # create a set of base names in masks
    mask_bases = {os.path.splitext(m)[0] for m in mask_files}
    # create a list of pairs (image_path, mask_path)
    pairs = []
    for img_name in image_files:
        base = os.path.splitext(img_name)[0]  # e.g. Phantom_005
        # expected mask name => base + ".npy"
        mask_name = base + ".npy"
        if mask_name in mask_files:
            pairs.append((img_name, mask_name))

    if len(pairs) == 0:
        print("No matching image->mask pairs found. Check naming & directories.")
        return

    # D) Randomly pick 10 distinct pairs
    n_samples = 10
    if len(pairs) < n_samples:
        n_samples = len(pairs)
    picks = random.sample(pairs, n_samples)

    print(f"Randomly picked {len(picks)} samples from {len(pairs)} total pairs.")

    # E) Create a figure with 10 rows, 4 columns => total subplots = 40
    fig, axes = plt.subplots(nrows=len(picks), ncols=4, figsize=(16, 4*len(picks)))
    if len(picks) == 1:
        # if there's only 1 pick, axes is 1D, make it 2D
        axes = [axes]  # just to handle shape

    # F) For each random pick, run inference and visualize
    for row_idx, (img_name, mask_name) in enumerate(picks):
        img_path  = os.path.join(test_image_dir, img_name)
        mask_path = os.path.join(test_mask_dir,  mask_name)

        # 1) Preprocess
        input_arr, orig_img = preprocess_image(img_path)

        # 2) Run inference
        pred_mask = run_inference(nnunet_model, input_arr)

        # 3) Load GT
        gt_mask = load_npy_mask(mask_path)

        # 4) Overlay
        overlay_img = overlay_prediction_on_image(orig_img, pred_mask)

        # 5) Plot in 4 subplots
        # If we have multiple rows, axes[row_idx] is an array of 4
        ax0, ax1, ax2, ax3 = axes[row_idx]
        
        # Original
        ax0.imshow(orig_img, cmap="gray")
        ax0.set_title(f"Image: {img_name}")
        ax0.axis("off")

        # Ground truth mask
        ax1.imshow(gt_mask, cmap="gray")
        ax1.set_title(f"Mask: {mask_name}")
        ax1.axis("off")

        # Predicted mask
        ax2.imshow(pred_mask, cmap="gray")
        ax2.set_title("Predicted")
        ax2.axis("off")

        # Overlay
        ax3.imshow(overlay_img)
        ax3.set_title("Overlay")
        ax3.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import random
    main()
