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
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.cuda.amp as amp

# For optional post-processing (CRF)
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# For skeletonization
from skimage.morphology import skeletonize

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
# 2. Positional Encoding (2D)
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
        
        # Change first conv to single channel input
        self.encoder.conv1 = nn.Conv2d(1, self.encoder.conv1.out_channels,
                                       kernel_size=self.encoder.conv1.kernel_size,
                                       stride=self.encoder.conv1.stride,
                                       padding=self.encoder.conv1.padding,
                                       bias=False)
        if pretrained:
            with torch.no_grad():
                # Average the weights across the RGB channels
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

    # Decoder stage 1
        up1 = self.decoder['up1'](transformed)
        skip1 = skip_features[3]
        if up1.shape[2:] != skip1.shape[2:]:
            up1 = F.interpolate(up1, size=skip1.shape[2:], mode='bilinear', align_corners=True)
        conv1 = self.decoder['conv1'](torch.cat([up1, skip1], dim=1))

    # Decoder stage 2
        up2 = self.decoder['up2'](conv1)
        skip2 = skip_features[2]
        if up2.shape[2:] != skip2.shape[2:]:
            up2 = F.interpolate(up2, size=skip2.shape[2:], mode='bilinear', align_corners=True)
        conv2 = self.decoder['conv2'](torch.cat([up2, skip2], dim=1))

    # Decoder stage 3
        up3 = self.decoder['up3'](conv2)
        skip3 = skip_features[1]
        if up3.shape[2:] != skip3.shape[2:]:
            up3 = F.interpolate(up3, size=skip3.shape[2:], mode='bilinear', align_corners=True)
        conv3 = self.decoder['conv3'](torch.cat([up3, skip3], dim=1))

    # Decoder stage 4
        up4 = self.decoder['up4'](conv3)
        skip4 = skip_features[0]
        if up4.shape[2:] != skip4.shape[2:]:
            up4 = F.interpolate(up4, size=skip4.shape[2:], mode='bilinear', align_corners=True)
        conv4 = self.decoder['conv4'](torch.cat([up4, skip4], dim=1))

        out = self.out_conv(conv4)
        out = nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

# ------------------------------
# 5. Model Loading Function for SegViT
# ------------------------------
def load_model(model_path, device='cpu',quantize =False, torchscript = False):
    """
    Loads the SegViT model from the saved state dictionary.
    """
    model = SegViT(encoder_name='resnet50', pretrained=False, num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Only apply quantization if on CPU (quantized ops are not supported on CUDA)
    if quantize and device.type == 'cpu':
        model = quantize_model(model, device)

    if torchscript:
        example_input = torch.randn(1, 1, 240, 320).to(device)
        model = torch.jit.trace(model, example_input)

    return model

def disable_qconfig_for_deconv(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose2d):
            module.qconfig = None

def quantize_model(model, device):
    model.eval()
    model.to('cpu')
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    disable_qconfig_for_deconv(model)
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration loop: ideally, use representative calibration data.
    for _ in range(10):
        dummy_input = torch.randn(1, 1, 240, 320)
        model(dummy_input)
    torch.quantization.convert(model, inplace=True)
    model.to(device)
    return model

# ------------------------------
# 6. Asynchronous Video Capture Class
# ------------------------------
class VideoCaptureAsync:
    def __init__(self, src=0, width=320, height=240):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = queue.Queue(maxsize=5)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    return
                self.q.put(frame)
            else:
                time.sleep(0.005)

    def read(self):
        if not self.q.empty():
            return self.q.get()
        else:
            return None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# ------------------------------
# 7. Preprocessing and ROI Extraction
# ------------------------------
def preprocess_frame(frame, device='cpu', target_size=(320, 240), rotate_clockwise90=False):
    # Check if frame is valid
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

def overlay_mask_on_frame(frame, mask, color=(0, 255, 0), alpha=0.5):
    mask_bool = mask.astype(bool)
    color_mask = np.zeros_like(frame)
    color_mask[mask_bool] = color
    overlayed_frame = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return overlayed_frame

# ------------------------------
# 8. Post-Processing: Filtering, Skeletonization, Tip Extraction
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
# 9. Optimized Video Processing Function with tqdm and Skeletonized View
# ------------------------------
def process_video(model, cap, out, device='cpu', target_size=(320, 240),
                  rotate_clockwise90=False, use_roi=True,
                  post_process_flag=False, crf_flag=False,
                  smooth_flag=False, thin_mask_flag=False,
                  enhance_flag=False, apply_filter=True,
                  merge_threshold=20, distance_threshold=50):
    # If total frame count is available, create a tqdm progress bar.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing Video") if total_frames > 0 else None

    print("Starting video processing...")
    fps_counter = []
    while True:
        start_time = time.time()
        frame = cap.read()
        # If cap.read() returns a tuple (for video files), extract the frame.
        if isinstance(frame, tuple):
            ret, frame = frame
            if not ret:
                break
        if frame is None:
            break

        # Preprocess the frame
        try:
            input_tensor, resized_frame = preprocess_frame(frame, device=device,
                                                           target_size=target_size,
                                                           rotate_clockwise90=rotate_clockwise90)
        except ValueError as ve:
            print(f"Preprocessing error: {ve}")
            continue

        if enhance_flag:
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)

        # Run inference with mixed precision
        with torch.amp.autocast('cuda'):
            output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, (resized_frame.shape[1], resized_frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        # Optional post-processing steps
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
        if apply_filter:
            pred = merge_close_components(pred, threshold=merge_threshold)
            pred = filter_far_components(pred, distance_threshold=distance_threshold)

        # Optional ROI extraction
        if use_roi:
            roi_frame, bbox = extract_roi(resized_frame, pred)
            x, y, w, h = bbox
            cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Compute skeletonization and overlay
        skel = skeletonize((pred > 0).astype(np.uint8))
        skel_uint8 = (skel.astype(np.uint8)) * 255
        skel_bgr = cv2.cvtColor(skel_uint8, cv2.COLOR_GRAY2BGR)
        skeleton_overlay = resized_frame.copy()
        skeleton_overlay[skel_uint8 > 0] = [0, 0, 255]
        skeleton_overlay = cv2.addWeighted(resized_frame, 0.7, skeleton_overlay, 0.3, 0)
        tip = get_tip_from_mask(pred)
        overlayed_frame = overlay_mask_on_frame(resized_frame, pred, color=(0, 255, 0), alpha=0.2)
        if tip is not None:
            cv2.circle(skeleton_overlay, tip, 1, (0, 0, 255), -1)
            cv2.putText(skeleton_overlay, "Tip", (tip[0]+10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(overlayed_frame, tip, 1, (0, 0, 255), -1)
            cv2.putText(overlayed_frame, "Tip", (tip[0]+10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Combine views for display (for debugging)
        combined_view = cv2.hconcat([resized_frame, overlayed_frame, skeleton_overlay])
        cv2.imshow('Combined View', combined_view)
        
        # **Key Modification:** Resize the output frame to exactly the target size
        output_frame = cv2.resize(overlayed_frame, target_size)
        out.write(output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if pbar is not None:
            pbar.update(1)

        end_time = time.time()
        fps_counter.append(1.0 / (end_time - start_time))
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        avg_fps = sum(fps_counter) / len(fps_counter)
        print(f"Current FPS: {avg_fps:.2f}", end="\r")
    cv2.destroyAllWindows()
    if pbar is not None:
        pbar.close()

# ------------------------------
# 10. Main Function
# ------------------------------
def main():
    # Use live camera for inference
    use_camera = False  # True for real camera inference; False for video file
    model_path = "best_segvit_model10.pth"  # Update with your model checkpoint path
    output_video_path = "output_video.mp4"
    target_size = (320, 240)  # Lower resolution for faster processing
    rotate_clockwise90 = True  # Change if needed
    fps = 30.0

    # Optimization flags
    post_process_flag = False
    crf_flag = False
    smooth_flag = False
    thin_mask_flag = False
    enhance_flag = False
    apply_filter = True
    merge_threshold = 20
    distance_threshold = 50
    use_roi = True

    # Deployment options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Only quantize if running on CPU (quantized ops not supported on CUDA)
    use_quantization = (device.type == 'cpu')
    use_torchscript = True

    print(f"[Main] Using device: {device}")
    print("[Main] Loading the nnU-Net model with optimizations...")
    model = load_model(model_path, device=device, quantize=use_quantization, torchscript=use_torchscript)
    print("Model loaded successfully.")

    # Use asynchronous video capture if using a camera
    if use_camera:
        cap = VideoCaptureAsync(src=0, width=target_size[0], height=target_size[1]).start()
    else:
        cap = cv2.VideoCapture("/home/yanghehao/tracklearning/DATA/Tom.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)

    print("[Main] Starting video processing...")
    process_video(model=model, cap=cap, out=out, device=device, target_size=target_size,
                  rotate_clockwise90=rotate_clockwise90, use_roi=use_roi,
                  post_process_flag=post_process_flag, crf_flag=crf_flag,
                  smooth_flag=smooth_flag, thin_mask_flag=thin_mask_flag,
                  enhance_flag=enhance_flag, apply_filter=apply_filter,
                  merge_threshold=merge_threshold, distance_threshold=distance_threshold)
    if use_camera:
        cap.stop()
    else:
        cap.release()
    out.release()
    print("[Main] Processing complete. Output video saved.")

if __name__ == "__main__":
    main()
