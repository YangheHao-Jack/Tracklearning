import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from skimage.morphology import skeletonize
import torchvision.models as models
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torchmetrics

# pydensecrf for CRF post-processing (if desired)
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

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
# 2. Positional Encoding
# ------------------------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, height, width):
        """
        Args:
            embed_dim (int): Must be even.
            height (int): Base height for the learnable embeddings.
            width (int): Base width for the learnable embeddings.
        """
        super(PositionalEncoding2D, self).__init__()
        assert embed_dim % 2 == 0, "Embed dimension must be even"
        self.row_embed = nn.Parameter(torch.randn(1, embed_dim // 2, height))
        self.col_embed = nn.Parameter(torch.randn(1, embed_dim // 2, width))

    def forward(self, x):
        B, C, H, W = x.shape
        # Interpolate embeddings to match current H and W
        row_embed = F.interpolate(self.row_embed, size=H, mode='linear', align_corners=False)
        col_embed = F.interpolate(self.col_embed, size=W, mode='linear', align_corners=False)
        row_embed = row_embed.unsqueeze(-1).expand(1, -1, H, W)
        col_embed = col_embed.unsqueeze(-2).expand(1, -1, H, W)
        pos = torch.cat([row_embed, col_embed], dim=1)  # (1, embed_dim, H, W)
        return pos.repeat(B, 1, 1, 1)

# ------------------------------
# 3. Transformer Encoder Module
# ------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                    dim_feedforward=ff_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        # x: (S, B, embed_dim)
        return self.transformer(x)

# ------------------------------
# 4. nnU-Net Model Definition (Simplified Version)
# ------------------------------
class ConvBlock(nn.Module):
    """
    A basic convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class nnUNet(nn.Module):
    """
    A simplified nnU-Net–style architecture for 2D segmentation.
    """
    def __init__(self, in_channels=1, out_channels=2, base_channels=32):
        super(nnUNet, self).__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        # Output layer
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x2 = self.enc2(x2)
        x3 = self.pool2(x2)
        x3 = self.enc3(x3)
        x4 = self.pool3(x3)
        x4 = self.enc4(x4)
        x5 = self.pool4(x4)
        b = self.bottleneck(x5)
        d4 = self.up4(b)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        # Upsample to recover original resolution (if needed)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

# ------------------------------
# 5. Model Loading, Quantization, and TorchScript Conversion
# ------------------------------
def load_model(model_path, device='cpu', quantize=False, torchscript=False):
    """
    Loads the nnU-Net model from a saved state dictionary, with options for quantization
    and TorchScript conversion.
    """
    # Use a smaller network for faster inference if possible.
    model = nnUNet(in_channels=1, out_channels=2, base_channels=8)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if quantize:
        model = quantize_model(model, device)
    
    if torchscript:
        # Convert to TorchScript using tracing.
        example_input = torch.randn(1, 1, 320, 480).to(device)
        model = torch.jit.trace(model, example_input)
    
    return model

def disable_qconfig_for_deconv(model):
    # Traverse the model and disable qconfig for ConvTranspose2d modules.
    for name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose2d):
            module.qconfig = None

def quantize_model(model, device):
    """
    Perform static quantization on the model.
    For layers not supported (e.g. ConvTranspose2d), disable quantization.
    """
    model.eval()
    # Move model to CPU for static quantization.
    model.to('cpu')
    
    # Set default quantization configuration.
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Disable quantization for unsupported layers.
    disable_qconfig_for_deconv(model)
    
    # Prepare model for static quantization.
    torch.quantization.prepare(model, inplace=True)
    
    # Calibration with dummy input (replace with representative data if available).
    for _ in range(10):
        dummy_input = torch.randn(1, 1, 320, 480)
        model(dummy_input)
    
    # Convert model to quantized version.
    torch.quantization.convert(model, inplace=True)
    
    # Move back to the desired device (CPU or GPU if supported).
    model.to(device)
    return model


# ------------------------------
# 6. Video I/O Functions
# ------------------------------
def initialize_video_io(input_path, output_path, fps=30.0, target_size=(480, 320), rotate_clockwise90=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")
    if rotate_clockwise90:
        frame_size = (target_size[1], target_size[0])
    else:
        frame_size = target_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return cap, out, frame_size

def preprocess_frame(frame, device='cpu', target_size=(480, 320), rotate_clockwise90=False):
    """
    Resize and normalize the frame. Here we use a lower resolution for faster processing.
    """
    resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    if rotate_clockwise90:
        resized_frame = cv2.rotate(resized_frame, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)
    image_tensor = TF.to_tensor(pil_image)
    normalize = transforms.Normalize(mean=[0.485], std=[0.229])
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
# 7. ROI Extraction Function
# ------------------------------
def extract_roi(frame, mask, pad=10):
    """
    Extracts a region of interest (ROI) from the frame based on the segmentation mask.
    This function finds the bounding box of the mask and returns the cropped frame.
    """
    coords = cv2.findNonZero((mask > 0).astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add a small padding around the ROI
        x = max(x - pad, 0)
        y = max(y - pad, 0)
        w = min(w + 2 * pad, frame.shape[1] - x)
        h = min(h + 2 * pad, frame.shape[0] - y)
        roi_frame = frame[y:y+h, x:x+w]
        return roi_frame, (x, y, w, h)
    else:
        return frame, (0, 0, frame.shape[1], frame.shape[0])

# ------------------------------
# 8. Filtering and Merging Near Components
# ------------------------------
def merge_close_components(mask, threshold=20):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if num_labels <= 1:
        return mask
    parent = {i: i for i in range(num_labels)}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
    boxes = {}
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        boxes[i] = (x - threshold, y - threshold, x + w + threshold, y + h + threshold)
    for i in range(1, num_labels):
        for j in range(i+1, num_labels):
            box1 = boxes[i]
            box2 = boxes[j]
            if (box1[2] > box2[0] and box1[3] > box2[1] and
                box2[2] > box1[0] and box2[3] > box1[1]):
                union(i, j)
    groups = {}
    for i in range(1, num_labels):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    merged_mask = np.zeros_like(mask)
    for group in groups.values():
        group_mask = np.isin(labels, group)
        merged_mask[group_mask] = 1
    return merged_mask

def filter_far_components(mask, distance_threshold=50, min_area=100):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    image_height = mask.shape[0]
    main_label = None
    max_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        component_mask = (labels == i).astype(np.uint8)
        ys, _ = np.where(component_mask)
        if ys.size > 0 and ys.max() > image_height * 0.8 and area > max_area:
            main_label = i
            max_area = area
    if main_label is None:
        valid = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)
                 if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if valid:
            main_label = max(valid, key=lambda x: x[1])[0]
        else:
            return mask
    main_component = (labels == main_label).astype(np.uint8)
    main_tip = get_tip_from_mask(main_component)
    if main_tip is None:
        return main_component
    final_mask = np.copy(main_component)
    for i in range(1, num_labels):
        if i == main_label:
            continue
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        centroid = centroids[i]
        dist = np.linalg.norm(np.array(centroid) - np.array(main_tip))
        if dist <= distance_threshold:
            final_mask[labels == i] = 1
    return final_mask

# ------------------------------
# 9. Tracking Functions: Skeletonization and Tip Extraction
# ------------------------------
def find_skeleton_endpoints(skel):
    endpoints = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    h, w = skel.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x]:
                count = 0
                for dx, dy in neighbors:
                    if skel[y + dy, x + dx]:
                        count += 1
                if count == 1:
                    endpoints.append((x, y))
    return endpoints

def get_tip_from_mask(mask):
    tip = None
    binary_mask = (mask > 0).astype(np.uint8)
    skel = skeletonize(binary_mask.astype(bool))
    endpoints = find_skeleton_endpoints(skel)
    if endpoints and len(endpoints) >= 2:
        base = max(endpoints, key=lambda p: p[1])
        endpoints.remove(base)
        tip = endpoints[0]
        max_distance = np.linalg.norm(np.array(tip) - np.array(base))
        for pt in endpoints:
            distance = np.linalg.norm(np.array(pt) - np.array(base))
            if distance > max_distance:
                tip = pt
                max_distance = distance
    else:
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

# ------------------------------
# 10. Video Processing Function with Optimizations
# ------------------------------
import torch.cuda.amp as amp

def process_video(model, cap, out, device='cpu', target_size=(480, 320),
                  rotate_clockwise90=False, post_process_flag=False, crf_flag=False,
                  smooth_flag=False, thin_mask_flag=False, enhance_flag=False, 
                  apply_filter=True, merge_threshold=20, distance_threshold=50,
                  use_roi=False):
    """
    Processes video frames using the nnU-Net model with various optimizations.
    Optionally uses ROI extraction if enabled.
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {frame_count}")
    
    # For demonstration, we process frames one-by-one.
    # (Batch processing could be implemented by collecting frames into a batch tensor.)
    if frame_count > 0:
        iterator = tqdm(range(frame_count), desc="Processing Video")
    else:
        iterator = iter(int, 1)  # For live camera mode

    for idx in iterator:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached at frame {idx}.")
            break
        
        # Preprocess frame at low resolution.
        input_tensor, resized_frame = preprocess_frame(frame, device=device, target_size=target_size, 
                                                       rotate_clockwise90=rotate_clockwise90)
        if enhance_flag:
            resized_frame = enhance_frame_max_contrast(resized_frame)
        
        # Use mixed precision for model inference.
        with amp.autocast():
            output = model(input_tensor)
        
        # Obtain prediction mask.
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, (resized_frame.shape[1], resized_frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        
        # Optionally streamline post-processing.
        if post_process_flag:
            kernel = np.ones((3, 3), np.uint8)
            pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel, iterations=1)
            pred = cv2.morphologyEx(pred, cv2.MORPH_DILATE, kernel, iterations=1)
        if smooth_flag:
            pred = cv2.GaussianBlur(pred.astype(np.float32), (5, 5), 0)
            _, pred = cv2.threshold(pred, 0.5, 1, cv2.THRESH_BINARY)
            pred = pred.astype(np.uint8)
        if crf_flag:
            # CRF can be expensive; disable if speed is paramount.
            resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            probs = np.stack([1 - pred, pred], axis=0).astype(np.float32)
            unary = unary_from_softmax(probs)
            d = dcrf.DenseCRF2D(resized_frame.shape[1], resized_frame.shape[0], 2)
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=resized_frame_rgb, compat=10)
            Q = d.inference(5)
            refined_mask = np.argmax(Q, axis=0).reshape((resized_frame.shape[0], resized_frame.shape[1]))
            pred = refined_mask.astype(np.uint8)
        if thin_mask_flag:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            pred = cv2.erode(pred, kernel, iterations=1)
        
        # Apply filtering.
        if apply_filter:
            pred = merge_close_components(pred, threshold=merge_threshold)
            pred = filter_far_components(pred, distance_threshold=distance_threshold)
        
        # Optionally perform ROI extraction.
        if use_roi:
            roi_frame, bbox = extract_roi(resized_frame, pred)
            # You could then re-run segmentation on roi_frame for a higher resolution prediction.
            # For simplicity, we will draw the ROI on the output.
            x, y, w, h = bbox
            cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        tip = get_tip_from_mask(pred)
        overlayed_frame = overlay_mask_on_frame(resized_frame, pred, color=(0, 255, 0), alpha=0.2)
        if tip is not None:
            cv2.circle(overlayed_frame, tip, 3, (0, 0, 255), -1)
            cv2.putText(overlayed_frame, "Tip", (tip[0] + 10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        
        binary_mask = (pred > 0).astype(np.uint8)
        skel = skeletonize(binary_mask.astype(bool))
        skel_uint8 = (skel * 255).astype(np.uint8)
        skel_bgr = cv2.cvtColor(skel_uint8, cv2.COLOR_GRAY2BGR)
        skeleton_overlay = resized_frame.copy()
        skeleton_overlay[skel_uint8 > 0] = [0, 0, 255]
        skeleton_overlay = cv2.addWeighted(resized_frame, 0.7, skeleton_overlay, 0.3, 0)
        if tip is not None:
            cv2.circle(skeleton_overlay, tip, 3, (0, 255, 0), -1)
            cv2.putText(skeleton_overlay, "Tip", (tip[0] + 10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        combined_view = cv2.hconcat([resized_frame, overlayed_frame, skeleton_overlay])
        
        # For speed tests, consider commenting out display windows.
        cv2.imshow('Original Video', resized_frame)
        cv2.imshow('Overlayed Video', overlayed_frame)
        cv2.imshow('Skeleton Overlay', skeleton_overlay)
        cv2.imshow('Combined View', combined_view)
        cv2.imshow('Segmented Mask', (pred * 255).astype(np.uint8))
        out.write(skeleton_overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Early termination triggered by user.")
            break
    print("Video processing complete.")
    cv2.destroyAllWindows()

# ------------------------------
# 11. Main Function: Toggle Between Live Camera and Recorded Video
# ------------------------------
def main():
    # Toggle between live camera and video file.
    use_camera = False  # Set True for live camera; False for recorded video.
    
    model_path = "best_nnunet_model2.pth"  # Path to your trained nnU-Net checkpoint.
    input_video_path = "/home/yanghehao/tracklearning/DATA/input_large2.mp4"  # Update with your video file path
    output_video_path = "/home/yanghehao/tracklearning/DATA/TomOut2nn2.mp4"
    target_size = (480, 320)  # Lower resolution for real-time performance.
    rotate_clockwise90 = True
    fps = 60.0  # Adjust FPS as needed.
    
    # Optional post-processing and optimization flags.
    post_process_flag = False
    crf_flag = False
    smooth_flag = False
    thin_mask_flag = False
    enhance_flag = False
    apply_filter = True
    merge_threshold = 20
    distance_threshold = 50
    use_roi = False  # Enable ROI extraction
    
    # Enable quantization and TorchScript conversion as desired.
    use_quantization = False
    use_torchscript = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")
    
    print("[Main] Loading the nnU-Net model with optimizations...")
    model = load_model(model_path, device=device, quantize=use_quantization, torchscript=use_torchscript)
    print("Model loaded successfully.")
    
    if use_camera:
        print("[Main] Opening live camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open camera")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)
    else:
        print("[Main] Opening video file...")
        cap, out, _ = initialize_video_io(input_video_path, output_video_path,
                                          fps=fps, target_size=target_size,
                                          rotate_clockwise90=rotate_clockwise90)
    
    print("[Main] Starting video processing with optimizations...")
    process_video(model=model, cap=cap, out=out, device=device, target_size=target_size,
                  rotate_clockwise90=rotate_clockwise90, post_process_flag=post_process_flag,
                  crf_flag=crf_flag, smooth_flag=smooth_flag, thin_mask_flag=thin_mask_flag,
                  enhance_flag=enhance_flag, apply_filter=apply_filter, merge_threshold=merge_threshold,
                  distance_threshold=distance_threshold, use_roi=use_roi)
    
    cap.release()
    out.release()
    print("[Main] Processing complete. Output video saved.")

if __name__ == "__main__":
    main()
