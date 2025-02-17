#!/usr/bin/env python3
import os
import shutil
import json
import numpy as np
import imageio
import subprocess
from PIL import Image

# Custom Reader/Writer for nnUNet
#   - Reads PNG for images
#   - Reads NPY for masks
#   - Writes PNG/NPY if needed
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter

class CustomReaderWriter(BaseReaderWriter):
    @staticmethod
    def read(filename):
        # If the extension is .png, assume it's an image
        if filename.lower().endswith(".png"):
            return imageio.imread(filename)
        # If .npy, assume it's a mask or data
        elif filename.lower().endswith(".npy"):
            return np.load(filename)
        else:
            raise ValueError("Unsupported file format: " + filename)

    @staticmethod
    def write(filename, data):
        if filename.lower().endswith(".png"):
            imageio.imwrite(filename, data)
        elif filename.lower().endswith(".npy"):
            np.save(filename, data)
        else:
            raise ValueError("Unsupported file format for writing: " + filename)


def create_folder_structure(output_dataset_dir):
    """
    Creates the nnUNet folder structure in nnUNet_raw:
      output_dataset_dir/
         ├── imagesTr
         └── labelsTr
    """
    imagesTr_dir = os.path.join(output_dataset_dir, "imagesTr")
    labelsTr_dir = os.path.join(output_dataset_dir, "labelsTr")
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)
    return imagesTr_dir, labelsTr_dir


def rename_and_convert_images_to_grayscale(images_dir, imagesTr_dir, prefix="Phantom"):
    """
    1) Copy & rename .png images -> single-channel grayscale
    2) Filenames: {prefix}_{idx:03d}_0000.png
    """
    image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".png"))
    for idx, fname in enumerate(image_files):
        new_fname = f"{prefix}_{idx:03d}_0000.png"
        src = os.path.join(images_dir, fname)
        dst = os.path.join(imagesTr_dir, new_fname)

        # Convert to single-channel grayscale
        img = Image.open(src).convert("L")
        img_arr = np.array(img, dtype=np.uint8)

        imageio.imwrite(dst, img_arr)

    return len(image_files)


def rename_and_copy_npy_masks(masks_dir, labelsTr_dir, prefix="Phantom"):
    """
    Copy & rename each .npy mask to labelsTr.
    Filenames: {prefix}_{idx:03d}.npy

    Also ensures the mask is strictly 0/1:
    For a 2-class problem, any non-zero => 1.
    """
    mask_files = sorted(f for f in os.listdir(masks_dir) if f.lower().endswith(".npy"))
    for idx, fname in enumerate(mask_files):
        src = os.path.join(masks_dir, fname)
        dst = os.path.join(labelsTr_dir, f"{prefix}_{idx:03d}.npy")

        mask_data = np.load(src)
        # If the mask might have 255 or other non-zero, unify them to 1
        mask_01 = (mask_data != 0).astype(np.uint8)

        np.save(dst, mask_01)

    return len(mask_files)


def create_dataset_json(output_dataset_dir, num_training,
                        image_file_ending=".png", mask_file_ending=".npy",
                        channel_name="Grayscale",
                        label_mapping={"background": 0, "object": 1},
                        reader_writer_class="CustomReaderWriter"):
    """
    Creates dataset.json in the raw dataset folder, referencing .png for images & .npy for masks.
    """
    dataset_info = {
        "channel_names": {"0": channel_name},
        "labels": label_mapping,
        "numTraining": num_training,
        "file_ending": image_file_ending,  # images end with .png
        "overwrite_image_reader_writer": reader_writer_class,
        "segmentation_file_ending": mask_file_ending  # masks end with .npy
    }
    json_path = os.path.join(output_dataset_dir, "dataset.json")
    with open(json_path, "w") as jf:
        json.dump(dataset_info, jf, indent=4)
    return json_path


def run_planning(dataset_id):
    """
    Runs the nnUNetv2 automatic planning & preprocessing step.
    Example uses the residual encoder planner.
    """
    command = f"nnUNetv2_plan_and_preprocess -d {dataset_id} -pl nnUNetPlannerResEncL"
    print("Running planning command:", command)
    subprocess.check_call(command, shell=True)
    print("Planning and preprocessing complete.")


def main():
    # Update paths as needed
    images_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/images/"  # PNG images
    masks_dir = "/home/yanghehao/tracklearning/segmentation/phantom_train/masks/"   # NPY masks
    dataset_id = 101
    prefix = "Phantom"

    # Output folder in nnUNet_raw
    output_dataset_dir = f"/home/yanghehao/nnUNet/nnunetv2/nnUNet_raw/Dataset{dataset_id}_Phantom"

    # 1) Create the folder structure
    imagesTr_dir, labelsTr_dir = create_folder_structure(output_dataset_dir)
    print(f"Created folder structure: {output_dataset_dir}")

    # 2) Convert images => single-channel .png
    num_images = rename_and_convert_images_to_grayscale(images_dir, imagesTr_dir, prefix)
    print(f"Copied + converted {num_images} images to {imagesTr_dir}")

    # 3) Copy .npy masks => .npy, unify label values => 0 or 1
    num_masks = rename_and_copy_npy_masks(masks_dir, labelsTr_dir, prefix)
    print(f"Copied + unified labels for {num_masks} masks in {labelsTr_dir}")

    if num_images != num_masks:
        print("WARNING: #images != #masks. Double-check your dataset files.")

    # 4) Create dataset.json (PNG images, NPY masks)
    dataset_json = create_dataset_json(
        output_dataset_dir,
        num_training=num_images,
        image_file_ending=".png",
        mask_file_ending=".npy",
        reader_writer_class="CustomReaderWriter"
    )
    print(f"Created dataset.json at: {dataset_json}")

    # 5) Run automatic planning
    run_planning(dataset_id)


if __name__ == "__main__":
    main()
