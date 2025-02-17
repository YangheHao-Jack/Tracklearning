import os
import cv2
import time
import threading
import queue
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.cuda.amp as amp

# Optional post-processing (CRF)
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# ------------------------------
# Global stop event for graceful shutdown
# ------------------------------
stop_event = threading.Event()

# ------------------------------
# 0. Reproducibility
# ------------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(42)

# ------------------------------
# 1. Dice Loss (for training if needed)
# ------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        probs = F.softmax(logits, dim=1)
        true_one_hot = F.one_hot(true, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

# ------------------------------
# 2. Positional Encoding (if needed)
# ------------------------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(PositionalEncoding2D, self).__init__()
        assert embed_dim % 2 == 0, "Embed dimension must be even"
        self.row_embed = nn.Parameter(torch.randn(1, embed_dim // 2, height))
        self.col_embed = nn.Parameter(torch.randn(1, embed_dim // 2, width))

    def forward(self, x):
        B, C, H, W = x.shape
        row_embed = F.interpolate(self.row_embed, size=H, mode='linear', align_corners=False)
        col_embed = F.interpolate(self.col_embed, size=W, mode='linear', align_corners=False)
        row_embed = row_embed.unsqueeze(-1).expand(1, -1, H, W)
        col_embed = col_embed.unsqueeze(-2).expand(1, -1, H, W)
        pos = torch.cat([row_embed, col_embed], dim=1)
        return pos.repeat(B, 1, 1, 1)

# ------------------------------
# 3. Transformer Encoder Module (if needed)
# ------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=4, num_heads=8, ff_dim=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                    dim_feedforward=ff_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

# ------------------------------
# 4. Simplified nnU-Net Model Definition
# ------------------------------
class ConvBlock(nn.Module):
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
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

# ------------------------------
# 5. Model Loading with TorchScript Conversion & Optimization
# ------------------------------
def load_model_native(model_path, device='cuda', quantize=False, torchscript=True):
    model = nnUNet(in_channels=1, out_channels=2, base_channels=32)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if quantize and device.type == 'cpu':
        model = quantize_model(model, device)

    if torchscript:
        example_input = torch.randn(1, 1, 240, 320).to(device)
        print("[INFO] Converting model to TorchScript...")
        traced_model = torch.jit.trace(model, example_input)
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        model = optimized_model
        print("[INFO] TorchScript model optimized for inference.")
    return model

def quantize_model(model, device):
    model.eval()
    model.to('cpu')
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    disable_qconfig_for_deconv(model)
    torch.quantization.prepare(model, inplace=True)
    for _ in range(10):
        dummy_input = torch.randn(1, 1, 240, 320)
        model(dummy_input)
    torch.quantization.convert(model, inplace=True)
    model.to(device)
    return model

def disable_qconfig_for_deconv(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose2d):
            module.qconfig = None

# ------------------------------
# 6. Preprocessing, ROI, and Overlay Functions
# ------------------------------
def preprocess_frame(frame, device='cpu', target_size=(320,240), rotate_clockwise90=False):
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("Invalid frame received for processing.")
    try:
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        raise ValueError(f"Error resizing frame: {e}")
    if rotate_clockwise90:
        resized_frame = cv2.rotate(resized_frame, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)
    image_tensor = TF.to_tensor(pil_image)
    normalize = transforms.Normalize(mean=[0.485], std=[0.229])
    image_tensor = normalize(image_tensor)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor, resized_frame

def extract_roi(frame, mask, pad=10):
    coords = cv2.findNonZero((mask > 0).astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x = max(x - pad, 0)
        y = max(y - pad, 0)
        w = min(w + 2 * pad, frame.shape[1] - x)
        h = min(h + 2 * pad, frame.shape[0] - y)
        roi_frame = frame[y:y+h, x:x+w]
        return roi_frame, (x, y, w, h)
    else:
        return frame, (0, 0, frame.shape[1], frame.shape[0])

def overlay_mask_on_frame(frame, mask, color=(0,255,0), alpha=0.5):
    mask_bool = mask.astype(bool)
    color_mask = np.zeros_like(frame)
    color_mask[mask_bool] = color
    overlayed_frame = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return overlayed_frame

# ------------------------------
# 7. Post-Processing: Filtering, Skeletonization, and Tip Extraction
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

# ------------------------------
# New Function: Inference on Patches for High-Resolution Inputs
# ------------------------------
def inference_on_patches(frame, model, device, patch_size=(512, 512), stride=(512, 512)):
    """
    Splits a high-res frame into patches, runs inference on each patch,
    and stitches the predictions back together.
    """
    H, W, _ = frame.shape
    prediction = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, stride[1]):
        for x in range(0, W, stride[0]):
            patch = frame[y: y+patch_size[1], x: x+patch_size[0]]
            h_patch, w_patch, _ = patch.shape
            if h_patch < patch_size[1] or w_patch < patch_size[0]:
                patch = cv2.copyMakeBorder(patch, 0, patch_size[1]-h_patch, 0, patch_size[0]-w_patch, cv2.BORDER_CONSTANT, value=0)
            # Process the patch using the same preprocessing (without rotation)
            tensor, _ = preprocess_frame(patch, device=device, target_size=patch_size, rotate_clockwise90=False)
            with (torch.amp.autocast("cuda") if device.type=="cuda" else nullcontext()):
                output = model(tensor)
            pred_patch = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            # Remove padding if applied
            pred_patch = pred_patch[:h_patch, :w_patch]
            prediction[y:y+h_patch, x:x+w_patch] += pred_patch
            count_map[y:y+h_patch, x:x+w_patch] += 1
    # Average overlapping predictions
    prediction = prediction / (count_map + 1e-8)
    # Threshold to create binary mask
    prediction = (prediction > 0.5).astype(np.uint8)
    return prediction

# ------------------------------
# 8. Multi-threaded Pipeline Workers and Global Queues
# ------------------------------
raw_queue = queue.Queue(maxsize=10)
preproc_queue = queue.Queue(maxsize=10)
infer_queue = queue.Queue(maxsize=10)

# Global lock for video writing
video_write_lock = threading.Lock()

def read_frames(cap):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            raw_queue.put(None)
            break
        raw_queue.put(frame)
    raw_queue.put(None)

def preprocess_worker(device, target_size, rotate_clockwise90):
    while not stop_event.is_set():
        frame = raw_queue.get()
        if frame is None:
            preproc_queue.put(None)
            break
        try:
            tensor, resized_frame = preprocess_frame(frame, device=device, target_size=target_size, rotate_clockwise90=rotate_clockwise90)
            preproc_queue.put((tensor, resized_frame))
        except Exception as e:
            print(f"Preprocessing worker error: {e}")
    preproc_queue.put(None)

def inference_worker(model, device):
    while not stop_event.is_set():
        item = preproc_queue.get()
        if item is None:
            infer_queue.put(None)
            break
        input_tensor, resized_frame = item
        H, W, _ = resized_frame.shape
        # If frame is high-res, process by patches.
        if H >= 2048 and W >= 2048:
            pred = inference_on_patches(resized_frame, model, device, patch_size=(512,512), stride=(512,512))
        else:
            with (torch.amp.autocast("cuda") if device.type=="cuda" else nullcontext()):
                output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
        infer_queue.put((resized_frame, pred))
    infer_queue.put(None)

def postprocess_worker(out, target_size, flags, pbar, rotate_clockwise90):
    post_process_flag = flags.get("post_process_flag", False)
    crf_flag = flags.get("crf_flag", False)
    smooth_flag = flags.get("smooth_flag", False)
    thin_mask_flag = flags.get("thin_mask_flag", False)
    enhance_flag = flags.get("enhance_flag", False)
    apply_filter = flags.get("apply_filter", True)
    merge_threshold = flags.get("merge_threshold", 20)
    distance_threshold = flags.get("distance_threshold", 50)
    use_roi = flags.get("use_roi", True)

    fps_counter = []
    while not stop_event.is_set():
        item = infer_queue.get()
        if item is None:
            break
        resized_frame, pred = item
        start_time = time.time()

        if enhance_flag:
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)
        if post_process_flag:
            kernel = np.ones((3,3), np.uint8)
            pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel, iterations=1)
            pred = cv2.morphologyEx(pred, cv2.MORPH_DILATE, kernel, iterations=1)
        if smooth_flag:
            pred = cv2.GaussianBlur(pred.astype(np.float32), (5,5), 0)
            _, pred = cv2.threshold(pred, 0.5, 1, cv2.THRESH_BINARY)
            pred = pred.astype(np.uint8)
        if crf_flag:
            resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            probs = np.stack([1-pred, pred], axis=0).astype(np.float32)
            unary = unary_from_softmax(probs)
            d = dcrf.DenseCRF2D(resized_frame.shape[1], resized_frame.shape[0], 2)
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=resized_frame_rgb, compat=10)
            Q = d.inference(5)
            refined_mask = np.argmax(Q, axis=0).reshape((resized_frame.shape[0], resized_frame.shape[1]))
            pred = refined_mask.astype(np.uint8)
        if thin_mask_flag:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            pred = cv2.erode(pred, kernel, iterations=1)
        if apply_filter:
            pred = merge_close_components(pred, threshold=merge_threshold)
            pred = filter_far_components(pred, distance_threshold=distance_threshold)

        if use_roi:
            roi_frame, bbox = extract_roi(resized_frame, pred)
            x, y, w, h = bbox
            cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        skel = skeletonize((pred > 0).astype(np.uint8))
        skel_uint8 = (skel.astype(np.uint8)) * 255
        skeleton_overlay = resized_frame.copy()
        skeleton_overlay[skel_uint8 > 0] = [0, 0, 255]
        skeleton_overlay = cv2.addWeighted(resized_frame, 0.7, skeleton_overlay, 0.3, 0)
        tip = get_tip_from_mask(pred)
        overlayed_frame = overlay_mask_on_frame(resized_frame, pred, color=(0,255,0), alpha=0.2)
        if tip is not None:
            cv2.circle(skeleton_overlay, tip, 1, (0, 0, 255), -1)
            cv2.putText(skeleton_overlay, "Tip", (tip[0]+10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(overlayed_frame, tip, 1, (0, 0, 255), -1)
            cv2.putText(overlayed_frame, "Tip", (tip[0]+10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Determine correct output size.
        # If rotated, swap dimensions.
        if rotate_clockwise90:
            output_size = (target_size[1], target_size[0])
        else:
            output_size = target_size

        output_frame = cv2.resize(overlayed_frame, output_size)
        with video_write_lock:
            out.write(output_frame)
            print("Frame written.")
        if pbar is not None:
            pbar.update(1)
        cv2.imshow('Display', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
        
        end_time = time.time()
        fps_counter.append(1.0 / (end_time - start_time))
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        avg_fps = sum(fps_counter) / len(fps_counter)
        print(f"Current FPS: {avg_fps:.2f}", end="\r")
    cv2.destroyAllWindows()

# ------------------------------
# 9. Main Function: Multi-threaded Pipeline with tqdm Progress Bar and Graceful Shutdown
# ------------------------------
def main():
    # Define the desired saved video dimensions (width, height)
    saved_width = 480
    saved_height = 320
    target_size = (saved_width, saved_height)  # This is used for preprocessing.
    
    # Set rotation flag. If True, the final saved output will have swapped dimensions.
    rotate_clockwise90 = True
    
    use_camera = False  # True for live camera inference; False for video file.
    model_path = "best_nnunet_model.pth"  # Update with your model checkpoint path.
    input_video_path = "/home/yanghehao/tracklearning/DATA/input_large2.mp4"  # Update with your input video path.
    
    # Adjust output size based on rotation.
    if rotate_clockwise90:
        output_size = (target_size[1], target_size[0])
    else:
        output_size = target_size
    
    output_video_path = "output.avi"  # Using .avi container with XVID codec
    fps = 30.0

    flags = {
        "post_process_flag": False,
        "crf_flag": False,
        "smooth_flag": False,
        "thin_mask_flag": False,
        "enhance_flag": False,
        "apply_filter": True,
        "merge_threshold": 20,
        "distance_threshold": 50,
        "use_roi": True,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    model = load_model_native(model_path, device=device, quantize=False, torchscript=True)
    print(f"[Main] Model loaded and optimized on {device}.")

    if use_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
        total_frames = None
    else:
        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)
    if not out.isOpened():
        print("Error: VideoWriter did not open. Check the codec and output file path.")
        return

    pbar = tqdm(total=total_frames, desc="Processing Video") if total_frames and total_frames > 0 else None

    threads = []
    threads.append(threading.Thread(target=read_frames, args=(cap,)))
    threads.append(threading.Thread(target=preprocess_worker, args=(device, target_size, rotate_clockwise90)))
    threads.append(threading.Thread(target=inference_worker, args=(model, device)))
    threads.append(threading.Thread(target=postprocess_worker, args=(out, target_size, flags, pbar, rotate_clockwise90)))

    for t in threads:
        t.daemon = True
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Interrupted! Setting stop event and waiting for threads to finish...")
        stop_event.set()
        for t in threads:
            t.join()
    finally:
        cap.release()
        out.release()
        if pbar:
            pbar.close()
        print("[Main] Processing complete. Output video saved.")

if __name__ == "__main__":
    main()
