import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torchmetrics
from skimage.morphology import skeletonize

# pydensecrf for CRF post-processing (if desired)
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

# ------------------------------
# 0. Reproducibility (Optional)
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
    Dice Loss for binary segmentation.
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
            height (int): Expected height of the feature map.
            width (int): Expected width of the feature map.
        """
        super(PositionalEncoding2D, self).__init__()
        assert embed_dim % 2 == 0, "Embed dimension must be even"
        self.row_embed = nn.Parameter(torch.randn(1, embed_dim // 2, height, 1))
        self.col_embed = nn.Parameter(torch.randn(1, embed_dim // 2, 1, width))
    
    def forward(self, x):
        B, C, H, W = x.shape
        row = F.interpolate(self.row_embed, size=(H, 1), mode='bilinear', align_corners=False)
        col = F.interpolate(self.col_embed, size=(1, W), mode='bilinear', align_corners=False)
        row = row.expand(B, -1, H, W)
        col = col.expand(B, -1, H, W)
        pos = torch.cat([row, col], dim=1)
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
        return self.transformer(x)

# ------------------------------
# 4. SegViT Model Definition
# ------------------------------
class TransformerBridge(nn.Module):
    def __init__(self, in_channels, embed_dim=768, num_heads=12, num_layers=12, ff_dim=3072, dropout=0.1):
        super(TransformerBridge, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.InstanceNorm2d(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0).view(B, -1, H, W)
        x = self.norm(x)
        return x

class DoubleConv(nn.Module):
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

class SegViT(nn.Module):
    def __init__(self, encoder_name='resnet50', pretrained=True, num_classes=2):
        super(SegViT, self).__init__()
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
            raise ValueError("Unsupported encoder. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101'.")
        
        self.encoder.conv1 = nn.Conv2d(1, self.encoder.conv1.out_channels,
                                       kernel_size=self.encoder.conv1.kernel_size,
                                       stride=self.encoder.conv1.stride,
                                       padding=self.encoder.conv1.padding,
                                       bias=False)
        if pretrained:
            with torch.no_grad():
                self.encoder.conv1.weight = nn.Parameter(
                    self.encoder.conv1.weight.mean(dim=1, keepdim=True)
                )
        
        self.encoder_layers = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        )
        
        self.transformer = TransformerBridge(
            in_channels=encoder_channels[-1],
            embed_dim=768,
            num_heads=24,
            num_layers=24,
            ff_dim=3072,
            dropout=0.1
        )
        
        if encoder_name in ['resnet50', 'resnet101']:
            decoder_channels = [512, 256, 128, 64]
        else:
            decoder_channels = [256, 128, 64, 64]
        
        self.decoder = nn.ModuleDict({
            'up1': nn.ConvTranspose2d(768, decoder_channels[0], kernel_size=2, stride=2),
            'conv1': DoubleConv(decoder_channels[0] + encoder_channels[3], decoder_channels[0]),
            'up2': nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2),
            'conv2': DoubleConv(decoder_channels[1] + encoder_channels[2], decoder_channels[1]),
            'up3': nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2),
            'conv3': DoubleConv(decoder_channels[2] + encoder_channels[1], decoder_channels[2]),
            'up4': nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2),
            'conv4': DoubleConv(decoder_channels[3] + encoder_channels[0], decoder_channels[3])
        })
        
        self.out_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, x):
        skip_features = []
        out = x
        for i, layer in enumerate(self.encoder_layers):
            out = layer(out)
            if i == 0:
                skip_features.append(out)  # after conv1
            elif i == 4:
                skip_features.append(out)  # after layer1
            elif i == 5:
                skip_features.append(out)  # after layer2
            elif i == 6:
                skip_features.append(out)  # after layer3
        transformed = self.transformer(out)
        up1 = self.decoder['up1'](transformed)
        skip1 = skip_features[3]
        conv1 = self.decoder['conv1'](torch.cat([up1, skip1], dim=1))
        up2 = self.decoder['up2'](conv1)
        skip2 = skip_features[2]
        conv2 = self.decoder['conv2'](torch.cat([up2, skip2], dim=1))
        up3 = self.decoder['up3'](conv2)
        skip3 = skip_features[1]
        conv3 = self.decoder['conv3'](torch.cat([up3, skip3], dim=1))
        up4 = self.decoder['up4'](conv3)
        skip4 = skip_features[0]
        conv4 = self.decoder['conv4'](torch.cat([up4, skip4], dim=1))
        out = self.out_conv(conv4)
        out = nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

# ------------------------------
# 5. Model Loading Function (Modified for SegViT)
# ------------------------------

def load_model(model_path, device='cpu'):
    """
    Loads the SegViT model from the saved state dictionary.
    """
    model = SegViT(encoder_name='resnet50', pretrained=False, num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
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
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotate_clockwise90:
        frame_size = (target_size[1], target_size[0])
    else:
        frame_size = target_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return cap, out, frame_size

def preprocess_frame(frame, device='cpu', target_size=(512, 512), rotate_clockwise90=False):
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

def apply_color_map(mask):
    colored_mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return colored_mask

# ------------------------------
# 7. Filtering and Merging Near Components
# ------------------------------

def merge_close_components(mask, threshold=20):
    """
    Merges nearby discontinuous components of the mask by expanding each component's
    bounding box and merging those that overlap.
    
    Args:
        mask (np.ndarray): Binary mask (0 and 1).
        threshold (int): Pixel margin for expanding bounding boxes.
    
    Returns:
        merged_mask (np.ndarray): Binary mask with near components merged.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if num_labels <= 1:
        return mask  # No component found.
    
    # Initialize union-find structure.
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
    
    # Create expanded bounding boxes for each component (skip background label 0).
    boxes = {}
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        boxes[i] = (x - threshold, y - threshold, x + w + threshold, y + h + threshold)
    
    # Merge components whose expanded bounding boxes intersect.
    for i in range(1, num_labels):
        for j in range(i+1, num_labels):
            box1 = boxes[i]
            box2 = boxes[j]
            if (box1[2] > box2[0] and box1[3] > box2[1] and
                box2[2] > box1[0] and box2[3] > box1[1]):
                union(i, j)
    
    # Group labels by their root.
    groups = {}
    for i in range(1, num_labels):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # Create merged mask.
    merged_mask = np.zeros_like(mask)
    for group in groups.values():
        group_mask = np.isin(labels, group)
        merged_mask[group_mask] = 1
    return merged_mask

def filter_far_components(mask, distance_threshold=50, min_area=200):
    """
    Filters out any component in the mask that is far away from the tip of the main component,
    and ignores tiny components below a given area threshold.
    
    Steps:
      1. Compute connected components of the mask.
      2. Identify the main component as the one that touches the bottom (or has the largest area among those
         exceeding min_area).
      3. Compute the tip of the main component using get_tip_from_mask().
      4. For every other component that is large enough (area >= min_area), compute its centroid.
         If the Euclidean distance between its centroid and the main tip is less than distance_threshold,
         merge it into the main component.
    
    Args:
        mask (np.ndarray): Binary mask (values 0 and 1).
        distance_threshold (int): Maximum allowable distance (in pixels) from the main tip to merge a component.
        min_area (int): Minimum area (in pixels) for a component to be considered.
    
    Returns:
        filtered_mask (np.ndarray): The binary mask after filtering out far-away and tiny components.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask  # Only background

    image_height = mask.shape[0]
    main_label = None
    max_area = 0
    # Identify the main component (preferably one that touches the bottom and is above min_area)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Ignore tiny components
        if area < min_area:
            continue
        component_mask = (labels == i).astype(np.uint8)
        ys, _ = np.where(component_mask)
        if ys.size > 0 and ys.max() > image_height * 0.8 and area > max_area:
            main_label = i
            max_area = area
    if main_label is None:
        # Fallback: choose the largest component (ignoring tiny ones)
        valid_areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if valid_areas:
            main_label = max(valid_areas, key=lambda x: x[1])[0]
        else:
            # If no component meets the minimum area requirement, return the original mask.
            return mask

    main_component = (labels == main_label).astype(np.uint8)
    main_tip = get_tip_from_mask(main_component)
    if main_tip is None:
        return main_component  # Unable to compute tip; return main component only.
    
    # Build the final mask: start with the main component.
    final_mask = np.copy(main_component)
    # Process other components: if they are large enough and near the main tip, merge them.
    for i in range(1, num_labels):
        if i == main_label:
            continue
        # Skip tiny components.
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        # Get the centroid for this component.
        centroid = centroids[i]  # (x, y)
        dist = np.linalg.norm(np.array(centroid) - np.array(main_tip))
        if dist <= distance_threshold:
            final_mask[labels == i] = 1
    return final_mask


# ------------------------------
# 8. Video Processing Function with Tracking, Skeletonization, and Advanced Filtering
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
    """
    Determine the catheter tip from a binary segmentation mask.
    The approach is:
      1. Skeletonize the mask.
      2. Find endpoints in the skeleton.
      3. Knowing that the catheter base is at the bottom, select the endpoint that is not the base.
    If skeletonization fails to yield two endpoints, a fallback method selects the point on the contour
    farthest from the base.
    
    Args:
        mask (np.ndarray): Binary segmentation mask (0s and 1s).
    
    Returns:
        tip (tuple): (x, y) coordinate of the catheter tip, or None if not found.
    """
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
    """
    Enhance the input frame by maximizing its contrast.
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

def process_video(model, cap, out, device='cpu', target_size=(512, 512), rotate_clockwise90=False, 
                  post_process_flag=False, crf_flag=False, smooth_flag=False, thin_mask_flag=False, 
                  enhance_flag=False, apply_filter=True, merge_threshold=20, distance_threshold=50):
    """
    Processes video frames using the SegViT model to segment the catheter and extract its tip.
    The pipeline now includes advanced filtering:
      - First, nearby discontinuous components are merged (using merge_close_components).
      - Then, components that are far away from the main component's tip are filtered out.
    The resulting mask is then used for skeletonization and tip extraction.
    
    The processed frame (with segmentation overlay and skeleton overlay, with tip marked) is written to the output video.
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {frame_count}")

    with torch.no_grad():
        for idx in tqdm(range(frame_count), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {idx}.")
                break

            # Preprocess frame.
            input_tensor, resized_frame = preprocess_frame(frame, device=device, target_size=target_size, 
                                                           rotate_clockwise90=rotate_clockwise90)
            if enhance_flag:
                resized_frame = enhance_frame_max_contrast(resized_frame)

            # Run segmentation model.
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred = cv2.resize(pred, (resized_frame.shape[1], resized_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Optional post-processing.
            if post_process_flag:
                kernel = np.ones((3, 3), np.uint8)
                pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel, iterations=1)
                pred = cv2.morphologyEx(pred, cv2.MORPH_DILATE, kernel, iterations=1)
            if smooth_flag:
                pred = cv2.GaussianBlur(pred.astype(np.float32), (5, 5), 0)
                _, pred = cv2.threshold(pred, 0.5, 1, cv2.THRESH_BINARY)
                pred = pred.astype(np.uint8)
            if crf_flag:
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
            
            # Merge nearby components if desired.
            if apply_filter:
                pred = merge_close_components(pred, threshold=merge_threshold)
                pred = filter_far_components(pred, distance_threshold=distance_threshold)

            # Compute the catheter tip.
            tip = get_tip_from_mask(pred)

            # Create overlays.
            overlayed_frame = overlay_mask_on_frame(resized_frame, pred, color=(0, 255, 0), alpha=0.2)
            if tip is not None:
                cv2.circle(overlayed_frame, tip, 3, (0, 0, 255), -1)
                cv2.putText(overlayed_frame, "Tip", (tip[0] + 10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2)
            
            # Compute skeleton overlay.
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
            
            # Create a combined view.
            combined_view = cv2.hconcat([resized_frame, overlayed_frame, skeleton_overlay])
            
            # Display overlays.
            cv2.imshow('Original Video', resized_frame)
            cv2.imshow('Overlayed Video', overlayed_frame)
            cv2.imshow('Skeleton Overlay', skeleton_overlay)
            cv2.imshow('Combined View', combined_view)
            cv2.imshow('Segmented Mask', (pred * 255).astype(np.uint8))
            
            # Write skeleton overlay to output video.
            out.write(skeleton_overlay)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early termination triggered by user.")
                break

    print("Video processing complete.")
    cv2.destroyAllWindows()

# ------------------------------
# 9. Main Function: Process Video and/or Camera Frames
# ------------------------------

def main():
    model_path = "best_segvit_model10.pth"  # Update to your SegViT checkpoint path.
    input_video_path = "/home/yanghehao/tracklearning/DATA/Tom.mp4"  # e.g., video file
    output_video_path = "/home/yanghehao/tracklearning/DATA/Out.mp4"
    target_size = (480, 320)  # (width, height) for processing.
    rotate_clockwise90 = True
    fps = 60.0  # Frames per second for the output video.
    
    # Optional flags for post-processing.
    post_process_flag = False
    crf_flag = False
    smooth_flag = False
    thin_mask_flag = False
    enhance_flag = False  # Set True to apply maximum contrast enhancement.
    apply_filter = True   # Enable advanced filtering.
    merge_threshold = 20  # Threshold for merging nearby components.
    distance_threshold = 100  # Maximum allowable distance from the main tip.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    print("[Main] Loading the model...")
    model = load_model(model_path, device=device)
    print("Model loaded successfully.")

    if input_video_path is not None:
        print("[Main] Initializing video capture and writer for video file...")
        cap, out, frame_size = initialize_video_io(input_video_path, output_video_path, fps=fps,
                                                    target_size=target_size, rotate_clockwise90=rotate_clockwise90)
    else:
        print("[Main] Opening camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open camera")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)

    print("[Main] Starting video processing...")
    process_video(model=model, cap=cap, out=out, device=device, target_size=target_size,
                  rotate_clockwise90=rotate_clockwise90, post_process_flag=post_process_flag,
                  crf_flag=crf_flag, smooth_flag=smooth_flag, thin_mask_flag=thin_mask_flag,
                  enhance_flag=enhance_flag, apply_filter=apply_filter, merge_threshold=merge_threshold,
                  distance_threshold=distance_threshold)
    
    cap.release()
    out.release()
    print("[Main] Processing complete. Output video saved.")

if __name__ == "__main__":
    main() 