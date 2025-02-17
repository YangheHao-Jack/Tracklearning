import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm

from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torchmetrics

# pydensecrf for CRF post-processing (if desired)
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

# ------------------------------
# 0. Reproducibility
# ------------------------------

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# 1. Dice Loss Only
# ------------------------------

class DiceLoss(nn.Module):
    """
    Standard Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        # logits: (B, num_classes, H, W), true: (B, H, W)
        probs = F.softmax(logits, dim=1)
        true_one_hot = F.one_hot(true, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

# ------------------------------
# 2. Positional Encoding (Corrected)
# ------------------------------

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, height, width):
        """
        Args:
            embed_dim (int): Must be even.
            height (int): Expected height of the feature map (during initialization).
            width (int): Expected width of the feature map (during initialization).
        """
        super(PositionalEncoding2D, self).__init__()
        assert embed_dim % 2 == 0, "Embed dimension must be even"
        # Create 4D parameters with singleton spatial dimensions.
        self.row_embed = nn.Parameter(torch.randn(1, embed_dim // 2, height, 1))
        self.col_embed = nn.Parameter(torch.randn(1, embed_dim // 2, 1, width))
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map of shape (B, embed_dim, H, W).
        Returns:
            pos (torch.Tensor): Positional encoding of shape (B, embed_dim, H, W).
        """
        B, C, H, W = x.shape
        # Interpolate row and col embeddings to match current spatial dimensions.
        row = F.interpolate(self.row_embed, size=(H, 1), mode='bilinear', align_corners=False)
        col = F.interpolate(self.col_embed, size=(1, W), mode='bilinear', align_corners=False)
        # Expand them to shape (B, embed_dim//2, H, W)
        row = row.expand(B, -1, H, W)
        col = col.expand(B, -1, H, W)
        # Concatenate along the channel dimension.
        pos = torch.cat([row, col], dim=1)  # shape: (B, embed_dim, H, W)
        return pos

# ------------------------------
# 3. Transformer Encoder Module
# ------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        # x: (S, B, embed_dim)
        return self.transformer(x)

# ------------------------------
# 4. TransUNet Model Definition (Using only DiceLoss)
# ------------------------------

class TransUNet(nn.Module):
    def __init__(self, num_classes=2, img_size=512, encoder_name='resnet34', pretrained=True):
        super(TransUNet, self).__init__()
        self.img_size = img_size

        # Encoder: choose different ResNet backbones (resnet18, resnet34, etc.)
        encoder_fn = getattr(models, encoder_name)
        self.encoder = encoder_fn(pretrained=pretrained)
        # Modify first conv for single-channel input
        self.encoder.conv1 = nn.Conv2d(1, self.encoder.conv1.out_channels,
                                       kernel_size=self.encoder.conv1.kernel_size,
                                       stride=self.encoder.conv1.stride,
                                       padding=self.encoder.conv1.padding,
                                       bias=False)
        if pretrained:
            with torch.no_grad():
                # Average pretrained weights over the RGB channels
                self.encoder.conv1.weight = nn.Parameter(self.encoder.conv1.weight.mean(dim=1, keepdim=True))
        
        # Use layers up to layer4
        self.encoder_layers = nn.Sequential(
            self.encoder.conv1,    # (B,64,H/2,W/2)
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,  # (B,64,H/4,W/4)
            self.encoder.layer1,   # (B,64,H/4,W/4)
            self.encoder.layer2,   # (B,128,H/8,W/8)
            self.encoder.layer3,   # (B,256,H/16,W/16)
            self.encoder.layer4    # (B,512,H/32,W/32)
        )
        
        # Transformer branch on the deepest feature map
        self.transformer_embed_dim = 512  # For resnet34, final channel count is 512
        self.feat_h = img_size // 32
        self.feat_w = img_size // 32
        self.flatten_conv = nn.Conv2d(512, self.transformer_embed_dim, kernel_size=1)
        self.pos_encoding = PositionalEncoding2D(self.transformer_embed_dim, self.feat_h, self.feat_w)
        self.transformer_encoder = TransformerEncoder(embed_dim=self.transformer_embed_dim,
                                                      num_layers=16, num_heads=16, ff_dim=3096)
        
        # Decoder: U-Net–style upsampling with skip connections
        self.up4 = nn.ConvTranspose2d(self.transformer_embed_dim, 256, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass; store skip connections from layer1, layer2, and layer3.
        x1 = self.encoder.conv1(x)
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x2 = self.encoder.maxpool(x1)
        x3 = self.encoder.layer1(x2)  # skip connection 1
        x4 = self.encoder.layer2(x3)  # skip connection 2
        x5 = self.encoder.layer3(x4)  # skip connection 3
        x6 = self.encoder.layer4(x5)  # deepest features

        # Transformer branch
        feat = self.flatten_conv(x6)           # (B,512,H/32,W/32)
        pos  = self.pos_encoding(feat)           # (B,512,H/32,W/32)
        feat = feat + pos
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1).permute(2, 0, 1)  # (S, B, C) where S = H*W
        feat_trans = self.transformer_encoder(feat_flat)
        feat_trans = feat_trans.permute(1, 2, 0).view(B, C, H, W)

        # Decoder with skip connections
        d4 = self.up4(feat_trans)              # (B,256,H/16,W/16)
        d4 = torch.cat([d4, x5], dim=1)          # concatenate skip from layer3
        d4 = self.conv4(d4)
        
        d3 = self.up3(d4)                      # (B,128,H/8,W/8)
        d3 = torch.cat([d3, x4], dim=1)          # skip from layer2
        d3 = self.conv3(d3)
        
        d2 = self.up2(d3)                      # (B,64,H/4,W/4)
        d2 = torch.cat([d2, x3], dim=1)          # skip from layer1
        d2 = self.conv2(d2)
        
        d1 = self.up1(d2)                      # (B,64,H/2,W/2)
        d1 = self.conv1(d1)
        out = self.out_conv(d1)
        # Upsample to original size
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

# ------------------------------
# 5. Model Loading Function
# ------------------------------

def load_model(model_path, device='cpu'):
    """
    Loads the TransUNet model from a saved state dictionary,
    adjusting positional encoding parameters if necessary.
    """
    # Initialize model without pretrained weights (since we'll load our checkpoint)
    model = TransUNet(num_classes=2, encoder_name='resnet34', pretrained=False)
    # Load the state dictionary from checkpoint
    state_dict = torch.load(model_path, map_location=device)
    
    # Check and reshape the positional encoding parameters if needed.
    # For example, if the checkpoint has pos_encoding.row_embed of shape [1, 256, 16],
    # we want to unsqueeze dimension -1 to get [1, 256, 16, 1].
    if 'pos_encoding.row_embed' in state_dict:
        param = state_dict['pos_encoding.row_embed']
        if len(param.shape) == 3:
            state_dict['pos_encoding.row_embed'] = param.unsqueeze(-1)
    if 'pos_encoding.col_embed' in state_dict:
        param = state_dict['pos_encoding.col_embed']
        if len(param.shape) == 3:
            state_dict['pos_encoding.col_embed'] = param.unsqueeze(2)
    
    # Now load the adjusted state dictionary
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ------------------------------
# 6. Video I/O Functions
# ------------------------------

def initialize_video_io(input_path, output_path, fps=30.0, target_size=(512, 512), rotate_clockwise90=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")
    # Get original dimensions (not used if resizing)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Determine output frame size
    resized_width, resized_height = target_size
    if rotate_clockwise90:
        frame_size = (resized_height, resized_width)
    else:
        frame_size = (resized_width, resized_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return cap, out, frame_size

def preprocess_frame(frame, device='cpu', target_size=(512, 512), rotate_clockwise90=False):
    resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    if rotate_clockwise90:
        resized_frame = cv2.rotate(resized_frame, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)
    # Use a simple transform (no augmentation)
    normalize = transforms.Normalize(mean=[0.485], std=[0.229])
    image_tensor = TF.to_tensor(pil_image)
    image_tensor = normalize(image_tensor)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor, resized_frame

def overlay_mask_on_frame(frame, mask, color=(0, 255, 0), alpha=0.5):
    mask_bool = mask.astype(bool)
    color_mask = np.zeros_like(frame)
    color_mask[mask_bool] = color
    overlayed_frame = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return overlayed_frame

# ------------------------------
# 7. Video (or Camera) Processing Function
# ------------------------------

def find_skeleton_endpoints(skel):
    """
    Find endpoints in a skeletonized binary image.
    An endpoint is defined as a pixel with only one 8-connected neighbor.
    
    Args:
        skel (np.ndarray): Binary skeleton image (dtype=bool or 0/1).
    
    Returns:
        List of endpoint coordinates as (x, y) tuples.
    """
    endpoints = []
    # Define the 8-connected neighborhood offsets.
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    h, w = skel.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x]:
                count = 0
                # Count the number of neighbors that are part of the skeleton.
                for dx, dy in neighbors:
                    if skel[y + dy, x + dx]:
                        count += 1
                if count == 1:
                    endpoints.append((x, y))
    return endpoints

def get_tip_from_mask(mask):
    """
    Determine the catheter tip from a binary segmentation mask.
    The approach is:
      1. Skeletonize the mask.
      2. Find endpoints in the skeleton.
      3. Knowing that the catheter base is at the bottom, select the endpoint
         that is not the base as the tip.
    
    If skeletonization fails to yield two endpoints, a fallback method selects
    the point on the contour farthest from the base (the bottom‑most point).
    
    Args:
        mask (np.ndarray): Binary segmentation mask (0s and 1s).
    
    Returns:
        tip (tuple): (x, y) coordinate of the catheter tip, or None if not found.
    """
    tip = None
    binary_mask = (mask > 0).astype(np.uint8)
    # Skeletonize expects a boolean array.
    skel = skeletonize(binary_mask.astype(bool))
    endpoints = find_skeleton_endpoints(skel)
    
    if endpoints and len(endpoints) >= 2:
        # Assume the base is the endpoint with the largest y-coordinate.
        base = max(endpoints, key=lambda p: p[1])
        endpoints.remove(base)
        # If multiple endpoints remain, select the one farthest from base.
        tip = endpoints[0]
        max_distance = np.linalg.norm(np.array(tip) - np.array(base))
        for pt in endpoints:
            distance = np.linalg.norm(np.array(pt) - np.array(base))
            if distance > max_distance:
                tip = pt
                max_distance = distance
    else:
        # Fallback: use the largest contour and select the farthest point from the base.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            base_point = max(largest_contour, key=lambda pt: pt[0][1])[0]
            max_distance = 0
            tip_point = None
            for point in largest_contour:
                pt = point[0]
                distance = np.linalg.norm(np.array(pt) - np.array(base_point))
                if distance > max_distance:
                    max_distance = distance
                    tip_point = pt
            if tip_point is not None:
                tip = (int(tip_point[0]), int(tip_point[1]))
    if tip is not None:
        return (int(tip[0]), int(tip[1]))
    else:
        return None

def enhance_frame_max_contrast(frame):
    """
    Enhance the input frame by maximizing its contrast.
    
    This function performs the following steps:
      1. Converts the frame from BGR to LAB color space.
      2. Applies CLAHE on the L (lightness) channel with an increased clip limit.
      3. Merges the processed L-channel with the original A and B channels.
      4. Converts back to BGR and normalizes the intensity to [0, 255].
      5. Applies a sharpening filter to further emphasize edges.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_frame = cv2.normalize(enhanced_frame, None, 0, 255, cv2.NORM_MINMAX)
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened_frame = cv2.filter2D(enhanced_frame, -1, sharpening_kernel)
    return sharpened_frame

def process_video(model, cap, out, device='cpu', target_size=(512, 512),
                  rotate_clockwise90=False, post_process_flag=False, crf_flag=False,
                  smooth_flag=False, thin_mask_flag=False, enhance_flag=False):
    """
    Processes video frames using the segmentation model to detect and visualize the catheter tip.
    In addition to overlaying the segmentation mask and tip on the frame, this version:
      - Computes and displays the skeleton (the one-pixel wide representation of the mask) on top
        of the original frame in a separate window ("Skeleton Overlay").
      - Combines the overlayed frame and original frame side by side in a window ("Combined View").
    
    Args:
        model: The segmentation model.
        cap: VideoCapture object.
        out: VideoWriter object.
        device (str): Device to run the model.
        target_size (tuple): Target frame size (width, height).
        rotate_clockwise90 (bool): Flag for rotation.
        post_process_flag, crf_flag, smooth_flag, thin_mask_flag (bool): Flags for mask processing.
        enhance_flag (bool): Flag to apply maximum contrast enhancement.
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {frame_count}")

    with torch.no_grad():
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {idx}.")
                break

            # Preprocess the frame (resize, rotation, grayscale conversion, normalization, etc.)
            input_tensor, resized_frame = preprocess_frame(frame, device=device,
                                                           target_size=target_size,
                                                           rotate_clockwise90=rotate_clockwise90)
            # Optionally enhance the frame for maximum contrast.
            if enhance_flag:
                resized_frame = enhance_frame_max_contrast(resized_frame)

            # Run the segmentation model.
            output = model(input_tensor)
            # Get the segmentation mask (binary: 0/1).
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Optional post-processing on the segmentation mask.
            if post_process_flag:
                kernel = np.ones((3, 3), np.uint8)
                pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel, iterations=1)
                pred = cv2.morphologyEx(pred, cv2.MORPH_DILATE, kernel, iterations=1)
            if smooth_flag:
                pred = cv2.GaussianBlur(pred.astype(np.float32), (5, 5), 0)
                _, pred = cv2.threshold(pred, 0.5, 1, cv2.THRESH_BINARY)
                pred = pred.astype(np.uint8)
            if crf_flag:
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                probs = np.stack([1 - pred, pred], axis=0).astype(np.float32)
                unary = unary_from_softmax(probs)
                d = dcrf.DenseCRF2D(resized_frame.shape[1], resized_frame.shape[0], 2)
                d.setUnaryEnergy(unary)
                d.addPairwiseGaussian(sxy=3, compat=3)
                d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=rgb_frame, compat=10)
                Q = d.inference(5)
                refined_mask = np.argmax(Q, axis=0).reshape((resized_frame.shape[0], resized_frame.shape[1]))
                pred = refined_mask.astype(np.uint8)
            if thin_mask_flag:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                pred = cv2.erode(pred, kernel, iterations=1)

            # Determine the catheter tip from the segmentation mask.
            tip = get_tip_from_mask(pred)

            # Create the overlayed frame: segmentation mask overlaid on the (enhanced) resized frame.
            overlayed_frame = overlay_mask_on_frame(resized_frame, pred, color=(0, 255, 0), alpha=0.2)
            
                

            # --- Compute and display the skeleton overlay ---
            # First, compute the skeleton of the binary mask.
            binary_mask = (pred > 0).astype(np.uint8)
            skel = skeletonize(binary_mask.astype(bool))
            # Convert the skeleton (boolean) to an 8-bit image.
            skel_uint8 = (skel * 255).astype(np.uint8)
            # Convert skeleton to BGR so we can color it.
            skel_bgr = cv2.cvtColor(skel_uint8, cv2.COLOR_GRAY2BGR)
            # Create an overlay: draw the skeleton in red on top of the original resized frame.
            skeleton_overlay = resized_frame.copy()
            # Where the skeleton is present, draw a red color.
            skeleton_overlay[skel_uint8 > 0] = [0, 0, 255]
            # Optionally, you can blend the skeleton overlay with the original frame.
            skeleton_overlay = cv2.addWeighted(resized_frame, 0.7, skeleton_overlay, 0.3, 0)
            if tip is not None:
                cv2.circle(overlayed_frame, tip, 1, (0, 0, 255), -1)
                cv2.putText(overlayed_frame, "Tip", (tip[0] + 10, tip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(skeleton_overlay, tip, 1, (0, 255,0), -1)
                cv2.putText(skeleton_overlay, "Tip", (tip[0] + 10, tip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2)
            # --- Create a side-by-side view (Combined View) ---
            # Stack the overlayed frame (with segmentation and tip) with the original frame.
            combined_view = cv2.hconcat([resized_frame,overlayed_frame, skeleton_overlay])

            # Display the various windows.
            cv2.imshow('Overlayed Video', overlayed_frame)
            cv2.imshow('Skeleton Overlay', skeleton_overlay)
            cv2.imshow('Combined View', combined_view)
            cv2.imshow('Segmented Mask', (pred * 255).astype(np.uint8))

            # Write the overlayed frame to output video (or you can choose another combination).
            out.write(skeleton_overlay)
            idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early termination triggered by user.")
                break

    print("Video processing complete.")
    cv2.destroyAllWindows()




# ------------------------------
# 8. Main Function: Process Video and/or Camera Frames
# ------------------------------

def main():
    # Set model file path (make sure to train and save your TransUNet model beforehand)
    model_path = "best_transunet_model2.pth"
    # For video processing, set input_video_path; for camera, set it to None.
    input_video_path = "/home/yanghehao/tracklearning/DATA/Tom.mp4"  # e.g., real video file
    # If you want to use the camera, set input_video_path = None and use device index 0.
    output_video_path = "/home/yanghehao/tracklearning/DATA/TomOut2.mp4"
    target_size = (480, 320)  # (width, height)
    rotate_clockwise90 = True # Adjust if needed
    fps = 30.0  # Frames per second for the output video

    # Optional post-processing flags
    post_process_flag = False
    crf_flag = False
    smooth_flag = False
    thin_mask_flag = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    print("[Main] Loading the model...")
    model = load_model(model_path, device=device)
    print("Model loaded successfully.")

    if input_video_path is not None:
        # Process a video file
        print("[Main] Initializing video capture and writer for video file...")
        cap, out, frame_size = initialize_video_io(input_video_path, output_video_path, fps=fps,
                                                    target_size=target_size, rotate_clockwise90=rotate_clockwise90)
    else:
        # Open the camera (device index 0)
        print("[Main] Opening camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open camera")
        # Set frame size if needed (for consistency with target_size)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
        # Create a video writer to save the output (optional)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)
    
    print("[Main] Starting video (or camera) processing...")
    process_video(model=model, cap=cap, out=out, device=device, target_size=target_size,
                  rotate_clockwise90=rotate_clockwise90, post_process_flag=post_process_flag,
                  crf_flag=crf_flag, smooth_flag=smooth_flag, thin_mask_flag=thin_mask_flag)
    
    cap.release()
    out.release()
    print("[Main] Processing complete. Output video saved.")

if __name__ == "__main__":
    main()
