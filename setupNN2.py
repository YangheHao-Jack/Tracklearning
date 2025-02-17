#!/usr/bin/env python3
import os
import shutil
import json
import numpy as np
import imageio
import subprocess
from PIL import Image

# Custom Reader/Writer for nnUNet
# Ensure that this class is in your PYTHONPATH.
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter

class CustomReaderWriter(BaseReaderWriter):
    @staticmethod
    def read(filename):
        if filename.lower().endswith(".npy"):
            # Load .npy and return the numpy array
            return np.load(filename)
        elif filename.lower().endswith(".png"):
            return imageio.imread(filename)
        else:
            raise ValueError("Unsupported file format: " + filename)

    @staticmethod
    def write(filename, data):
        if filename.lower().endswith(".npy"):
            np.save(filename, data)
        elif filename.lower().endswith(".png"):
            imageio.imwrite(filename, data)
        else:
            raise ValueError("Unsupported file format for writing: " + filename)


def create_folder_structure(output_dataset_dir):
    """
    Create the nnUNet folder structure in nnUNet_raw:
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
    1) Copy & rename .png images
    2) Convert to single-channel grayscale
    Filenames: {prefix}_{idx:03d}_0000.png
    """
    image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".png"))
    for idx, fname in enumerate(image_files):
        new_fname = f"{prefix}_{idx:03d}_0000.png"
        src = os.path.join(images_dir, fname)
        dst = os.path.join(imagesTr_dir, new_fname)

        # Convert to single-channel grayscale
        img = Image.open(src).convert("L")
        img_arr = np.array(img, dtype=np.uint8)

        # Write grayscale .png
        imageio.imwrite(dst, img_arr)

    return len(image_files)


def rename_and_convert_masks_to_0or1(masks_dir, labelsTr_dir, prefix="Phantom"):
    """
    Loads each .npy mask.
    If mask contains 0 and 1, we keep it that way.
    If your mask was 0 and 255, we convert 255 -> 1, i.e. (mask_data > 0).
    Then writes .png with 0 or 1 values for training.
    
    Filenames: {prefix}_{idx:03d}.png
    """
    mask_files = sorted(f for f in os.listdir(masks_dir) if f.lower().endswith(".npy"))
    for idx, fname in enumerate(mask_files):
        src = os.path.join(masks_dir, fname)
        dst = os.path.join(labelsTr_dir, f"{prefix}_{idx:03d}.png")

        # Load .npy => ensure values are 0 or 1
        mask_data = np.load(src)
        # If it's already 0/1, or if it's 0/255 => unify them:
        # We'll treat all non-zero as 1
        mask_visual= mask_data *255
        mask_01 = (mask_visual > 0).astype(np.uint8)

        # Write .png with values {0,1}
        imageio.imwrite(dst, mask_01)

    return len(mask_files)


def create_dataset_json(output_dataset_dir, num_training,
                        image_file_ending=".png", mask_file_ending=".png",
                        channel_name="Grayscale",
                        label_mapping={"background": 0, "object": 1},
                        reader_writer_class="CustomReaderWriter"):
    """
    Creates dataset.json in the raw dataset folder, referencing .png for images & masks.
    """
    dataset_info = {
        "channel_names": {"0": channel_name},
        "labels": label_mapping,
        "numTraining": num_training,
        "file_ending": image_file_ending,  # image suffix
        "overwrite_image_reader_writer": reader_writer_class,
        "segmentation_file_ending": mask_file_ending  # mask suffix
    }
    json_path = os.path.join(output_dataset_dir, "dataset.json")
    with open(json_path, "w") as jf:
        json.dump(dataset_info, jf, indent=4)
    return json_path


def run_planning(dataset_id):
    """
    Run the nnUNetv2 automatic planning step using the residual encoder planner (example).
    """
    command = f"nnUNetv2_plan_and_preprocess -d {dataset_id} -pl nnUNetPlannerResEncL"
    print("Running planning command:", command)
    subprocess.check_call(command, shell=True)
    print("Planning and preprocessing complete.")


def main():
    # Update these paths as needed
    images_dir = "/media/yanghehao/新加卷2/Trackinglearning/segmentation/phantom_train/images/"
    masks_dir = "/media/yanghehao/新加卷2/Trackinglearning/segmentation/phantom_train/masks/"
    dataset_id = 101  # e.g. for Dataset101_Phantom
    output_dataset_dir = f"/home/yanghehao/nnUNet/nnunetv2/nnUNet_raw/Dataset{dataset_id}_Phantom"
    prefix = "Phantom"

    # Step 1: create raw data folder structure
    imagesTr_dir, labelsTr_dir = create_folder_structure(output_dataset_dir)
    print(f"Created folder structure in: {output_dataset_dir}")

    # Step 2: convert images to grayscale => imagesTr
    num_images = rename_and_convert_images_to_grayscale(images_dir, imagesTr_dir, prefix)
    print(f"Copied + converted {num_images} images to grayscale in {imagesTr_dir}")

    # Step 3: convert .npy masks => 0/1 => .png => labelsTr
    num_masks = rename_and_convert_masks_to_0or1(masks_dir, labelsTr_dir, prefix)
    print(f"Converted + renamed {num_masks} masks to {labelsTr_dir}")

    if num_images != num_masks:
        print("WARNING: #images != #masks. Double-check your dataset.")

    # Step 4: create dataset.json in the raw folder
    dataset_json = create_dataset_json(
        output_dataset_dir,
        num_training=num_images,
        image_file_ending=".png",
        mask_file_ending=".png",
        reader_writer_class="CustomReaderWriter"
    )
    print(f"Created dataset.json at: {dataset_json}")

    # Step 5: run the automatic planning step
    run_planning(dataset_id)


if __name__ == "__main__":
    main()
