import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.models as models  # For ResNet encoder
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import torchmetrics
import random

# Import pydensecrf modules
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

# ------------------------------
# 0. Reproducibility (Optional)
# ------------------------------

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# 1. Combined Loss Function
# ------------------------------

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        """
        Args:
            logits (torch.Tensor): Predicted logits with shape [B, C, H, W].
            true (torch.Tensor): Ground truth masks with shape [B, H, W].
        Returns:
            torch.Tensor: Dice loss.
        """
        probs = nn.functional.softmax(logits, dim=1)
        true_one_hot = nn.functional.one_hot(true, num_classes=probs.shape[1])
        true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_one_hot, dims)
        cardinality = torch.sum(probs + true_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy Loss and Dice Loss.
    """
    def __init__(self, weight=None, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(smooth)

    def forward(self, logits, true):
        ce_loss = self.ce(logits, true)
        dice_loss = self.dice(logits, true)
        return ce_loss + dice_loss

# ------------------------------
# 2. Dataset and Transforms
# ------------------------------

class ToTensor:
    """
    Custom transform to convert images and masks to tensors without resizing.
    """
    def __init__(self, augment=False):
        self.augment = augment
        self.normalize = transforms.Normalize(mean=[0.485],
                                             std=[0.229])

    def __call__(self, image, mask):
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(Image.fromarray(mask))
                mask = np.array(mask, dtype=np.uint8)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(Image.fromarray(mask))
                mask = np.array(mask, dtype=np.uint8)
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            if angle != 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(Image.fromarray(mask), angle)
                mask = np.array(mask, dtype=np.uint8)
        image_tensor = TF.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        mask_tensor  = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor

class BinarySegDataset(Dataset):
    """
    Custom Dataset for binary image segmentation.
    """
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.file_list  = file_list
        self.transform  = transform
        print(f"[Dataset] Initialized with {len(file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_filename = self.file_list[idx]
        base_name = os.path.splitext(img_filename)[0]
        img_path = os.path.join(self.images_dir, img_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"[Error] Image file not found: {img_path}")
        image = Image.open(img_path).convert("L")
        mask_extensions = ['.npy', '.png', '.jpg', '.jpeg']
        mask_path = None
        for ext in mask_extensions:
            potential_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.isfile(potential_path):
                mask_path = potential_path
                break
        if mask_path is None:
            raise FileNotFoundError(f"[Error] Mask file not found for image: {img_filename}")
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
            mask = Image.fromarray(mask)
        else:
            mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 1, 1, mask_np).astype(np.uint8)
        mask = Image.fromarray(mask_np)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask

# ------------------------------
# 3. SegViT Model Definition
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
    def __init__(self, encoder_name='resnet34', pretrained=True, num_classes=2):
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
        
        self.encoder.conv1 = nn.Conv2d(
            1, 
            self.encoder.conv1.out_channels, 
            kernel_size=self.encoder.conv1.kernel_size, 
            stride=self.encoder.conv1.stride, 
            padding=self.encoder.conv1.padding, 
            bias=self.encoder.conv1.bias is not None
        )
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
            num_heads=12,
            num_layers=12,
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
                skip_features.append(out)
            elif i == 4:
                skip_features.append(out)
            elif i == 5:
                skip_features.append(out)
            elif i == 6:
                skip_features.append(out)
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
# 4. Video Processing Functions
# ------------------------------

def load_model(model_path, device='cpu'):
    """
    Loads the SegViT model from the saved state dictionary.
    """
    model = SegViT(encoder_name='resnet34', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def initialize_video_io(input_path, output_path, fps=30.0, target_size=(512, 512), rotate_clockwise90=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    transform = ToTensor(augment=False)
    image_tensor, _ = transform(pil_image, np.zeros_like(gray))
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor, resized_frame

def apply_color_map(mask):
    colored_mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return colored_mask

def overlay_mask_on_frame(frame, mask, color=(0, 255, 0), alpha=0.5):
    mask_bool = mask.astype(bool)
    color_mask = np.zeros_like(frame)
    color_mask[mask_bool] = color
    overlayed_frame = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return overlayed_frame

def apply_crf(image, mask, n_iters=5):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    probs = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)
    probs[0, :, :] = 1 - mask
    probs[1, :, :] = mask
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    Q = d.inference(n_iters)
    refined_mask = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return refined_mask

# ------------------------------
# 5. Thin Mask Function (Morphological Erosion)
# ------------------------------

def thin_mask(mask, kernel_size=3, iterations=1):
    """
    Applies morphological erosion to thin the predicted mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    return eroded_mask

# ------------------------------
# 6. Video Processing Function
# ------------------------------

def process_video(model, cap, out, device='cpu', target_size=(512, 512), rotate_clockwise90=False, 
                  post_process_flag=False, crf_flag=False, smooth_flag=False, thin_mask_flag=False):
    """
    Processes each frame of the input video.
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {frame_count}")

    with torch.no_grad():
        for idx in tqdm(range(frame_count), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {idx}.")
                break

            input_tensor, resized_frame = preprocess_frame(frame, device=device, target_size=target_size, rotate_clockwise90=rotate_clockwise90)
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_original = pred.astype(np.uint8)

            if post_process_flag:
                kernel = np.ones((3, 3), np.uint8)
                pred_original = cv2.morphologyEx(pred_original, cv2.MORPH_OPEN, kernel, iterations=1)
                pred_original = cv2.morphologyEx(pred_original, cv2.MORPH_DILATE, kernel, iterations=1)
            if smooth_flag:
                pred_original = cv2.GaussianBlur(pred_original.astype(np.float32), (5, 5), 0)
                _, pred_original = cv2.threshold(pred_original, 0.5, 1, cv2.THRESH_BINARY)
                pred_original = pred_original.astype(np.uint8)
            if crf_flag:
                resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                refined_mask = apply_crf(resized_frame_rgb, pred_original, n_iters=5)
                pred_original = refined_mask
            if thin_mask_flag:
                pred_original = thin_mask(pred_original, kernel_size=3, iterations=1)

            overlayed_frame = overlay_mask_on_frame(resized_frame, pred_original, color=(0, 255, 0), alpha=0.2)
            colored_mask = apply_color_map(pred_original)

            current_fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(overlayed_frame, f'FPS: {current_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlayed_frame, f'Frame: {idx+1}/{frame_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Original Video', resized_frame)
            cv2.imshow('Segmented Mask', colored_mask)
            cv2.imshow('Overlayed Video', overlayed_frame)

            out.write(overlayed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early termination triggered by user.")
                break

    print("Video processing complete.")
    cv2.destroyAllWindows()

# ------------------------------
# 7. Main Function
# ------------------------------

def main():
    model_path = "best_segvit_model9.pth"
    input_video_path = "/home/yanghehao/tracklearning/DATA/Tom.mp4"
    output_video_path = "/home/yanghehao/tracklearning/DATA/Out.mp4"
    target_size = (480, 320)  # (width, height)
    rotate_clockwise90 = True
    fps = 60.0
    post_process_flag = False  # Set True to apply morphological open/dilate
    crf_flag = False           # Set True to apply CRF refinement
    smooth_flag = False        # Set True to apply Gaussian smoothing
    thin_mask_flag = True      # Set True to apply the thin mask (morphological erosion)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    print("[Main] Loading the model...")
    model = load_model(model_path, device=device)
    print("Model loaded successfully.")

    print("[Main] Initializing video capture and writer...")
    cap, out, frame_size = initialize_video_io(input_video_path, output_video_path, fps=fps, target_size=target_size, rotate_clockwise90=rotate_clockwise90)
    print("Video capture and writer initialized.")

    print("[Main] Starting video processing...")
    process_video(
        model=model, 
        cap=cap, 
        out=out, 
        device=device, 
        target_size=target_size, 
        rotate_clockwise90=rotate_clockwise90, 
        post_process_flag=post_process_flag, 
        crf_flag=crf_flag, 
        smooth_flag=smooth_flag,
        thin_mask_flag=thin_mask_flag
    )

    cap.release()
    out.release()
    print("[Main] Video saved successfully.")

# ------------------------------
# 8. Execute the Main Function
# ------------------------------

if __name__ == "__main__":
    main()
