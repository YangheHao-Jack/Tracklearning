import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
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
        probs = nn.functional.softmax(logits, dim=1)  # Convert logits to probabilities
        true_one_hot = nn.functional.one_hot(true, num_classes=probs.shape[1])  # [B, H, W, C]
        true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)  # Dimensions to sum over: batch, height, width
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
        """
        Args:
            logits (torch.Tensor): Predicted logits with shape [B, C, H, W].
            true (torch.Tensor): Ground truth masks with shape [B, H, W].

        Returns:
            torch.Tensor: Combined loss.
        """
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
        """
        Args:
            image (PIL Image): Input image.
            mask (np.array or PIL Image): Corresponding mask as a NumPy array or PIL Image.

        Returns:
            image_tensor (torch.Tensor): Normalized image tensor.
            mask_tensor (torch.Tensor): Mask tensor (Long type).
        """
        # Data Augmentation (if enabled)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(Image.fromarray(mask))
                mask = np.array(mask, dtype=np.uint8)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(Image.fromarray(mask))
                mask = np.array(mask, dtype=np.uint8)

            # Random rotation by 0°, 90°, 180°, or 270°
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            if angle != 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(Image.fromarray(mask), angle)
                mask = np.array(mask, dtype=np.uint8)

            # Add more augmentations as needed

        # Convert image and mask to tensors
        image_tensor = TF.to_tensor(image)             # [1, H, W], float32 in [0,1]
        image_tensor = self.normalize(image_tensor)

        mask_tensor  = torch.from_numpy(mask).long()  # [H, W], dtype=torch.int64

        return image_tensor, mask_tensor

class BinarySegDataset(Dataset):
    """
    Custom Dataset for binary image segmentation.
    """
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        """
        Args:
            images_dir (str): Directory with input images.
            masks_dir (str): Directory with corresponding mask files (.npy or image files).
            file_list (list): List of image filenames.
            transform (callable, optional): Transform to be applied on a sample.
        """
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.file_list  = file_list
        self.transform  = transform

        # Debug: Print how many files in this split
        print(f"[Dataset] Initialized with {len(file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask pair at the specified index.
        """
        img_filename = self.file_list[idx]
        base_name = os.path.splitext(img_filename)[0]

        # Load image
        img_path = os.path.join(self.images_dir, img_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"[Error] Image file not found: {img_path}")
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        # Load mask
        # Attempt to find mask with the same base name and supported extensions
        mask_extensions = ['.npy', '.png', '.jpg', '.jpeg']
        mask_path = None
        for ext in mask_extensions:
            potential_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.isfile(potential_path):
                mask_path = potential_path
                break
        if mask_path is None:
            raise FileNotFoundError(f"[Error] Mask file not found for image: {img_filename}")

        # Load mask
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
            mask = Image.fromarray(mask)
        else:
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        # Remap any label >1 to 1 to ensure binary segmentation
        mask_np = np.array(mask)
        mask_np = np.where(mask_np > 1, 1, mask_np).astype(np.uint8)
        mask = Image.fromarray(mask_np)

        # Apply transformations if any
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

# ------------------------------
# 3. SegViT Model Definition
# ------------------------------

class TransformerBridge(nn.Module):
    def __init__(self, in_channels, embed_dim=768, num_heads=12, num_layers=12, ff_dim=3072, dropout=0.1):
        """
        Transformer bridge module.

        Args:
            in_channels (int): Number of input channels from the encoder.
            embed_dim (int): Embedding dimension for the transformer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            ff_dim (int): Feedforward network dimension.
            dropout (float): Dropout rate.
        """
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
        self.norm = nn.InstanceNorm2d(embed_dim)  # Changed from LayerNorm to InstanceNorm2d

    def forward(self, x):
        """
        Forward pass for the transformer bridge.

        Args:
            x (torch.Tensor): Encoder output [B, C, H, W]

        Returns:
            torch.Tensor: Transformed features [B, C', H, W]
        """
        B, C, H, W = x.shape
        x = self.conv(x)  # [B, embed_dim, H, W]
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, embed_dim]
        x = self.transformer_encoder(x)     # [H*W, B, embed_dim]
        x = x.permute(1, 2, 0).view(B, -1, H, W)  # [B, embed_dim, H, W]
        x = self.norm(x)  # [B, embed_dim, H, W]
        return x

class DoubleConv(nn.Module):
    """
    A block consisting of two convolutional layers each followed by BatchNorm and ReLU.
    """
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
    """
    SegViT architecture integrating ResNet encoder with a Transformer bridge and U-Net-like decoder.
    """
    def __init__(self, encoder_name='resnet34', pretrained=True, num_classes=2):
        """
        Args:
            encoder_name (str): Name of the CNN encoder ('resnet18', 'resnet34', 'resnet50', etc.).
            pretrained (bool): Whether to use pretrained weights for the encoder.
            num_classes (int): Number of output segmentation classes.
        """
        super(SegViT, self).__init__()
        
        # Initialize the encoder
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
        
        # Modify the first convolutional layer to accept single-channel input
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
                # Average the weights across the RGB channels to initialize for single channel
                self.encoder.conv1.weight = nn.Parameter(
                    self.encoder.conv1.weight.mean(dim=1, keepdim=True)
                )
        
        # Remove fully connected layers and pooling layers
        self.encoder_layers = nn.Sequential(
            self.encoder.conv1,   # [B, 64, H/2, W/2]
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool, # [B, 64, H/4, W/4]
            self.encoder.layer1,  # [B, 64, H/4, W/4]
            self.encoder.layer2,  # [B, 128, H/8, W/8]
            self.encoder.layer3,  # [B, 256, H/16, W/16]
            self.encoder.layer4   # [B, 512 or 2048, H/32, W/32]
        )
        
        # Transformer Bridge
        self.transformer = TransformerBridge(
            in_channels=encoder_channels[-1],
            embed_dim=768,      # Common ViT hidden size
            num_heads=16,      # Number of attention heads
            num_layers=16,     # Number of transformer encoder layers
            ff_dim=3072,       # Feedforward network dimension
            dropout=0.1
        )
        
        # Decoder
        # Adjust the decoder channels based on encoder type
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
        
        # Final Output Layer
        self.out_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the SegViT model.

        Args:
            x (torch.Tensor): Input image tensor [B, 1, H, W]

        Returns:
            torch.Tensor: Segmentation mask logits [B, num_classes, H, W]
        """
        # Encoder
        skip_features = []
        out = x
        for i, layer in enumerate(self.encoder_layers):
            out = layer(out)
            if i == 0:
                skip_features.append(out)  # conv1: [B, 64, H/2, W/2]
            elif i == 4:
                skip_features.append(out)  # layer1: [B, 64, H/4, W/4]
            elif i == 5:
                skip_features.append(out)  # layer2: [B, 128, H/8, W/8]
            elif i == 6:
                skip_features.append(out)  # layer3: [B, 256, H/16, W/16]
        # No need to capture layer4 for skip connections

        # Transformer Bridge
        transformed = self.transformer(out)  # [B, 768, H/32, W/32]

        # Decoder Step 1
        up1 = self.decoder['up1'](transformed)    # [B, decoder_channels[0], H/16, W/16]
        skip1 = skip_features[3]                   # layer3 output: [B, 256, H/16, W/16]
        conv1 = self.decoder['conv1'](torch.cat([up1, skip1], dim=1))  # [B, decoder_channels[0], H/16, W/16]

        # Decoder Step 2
        up2 = self.decoder['up2'](conv1)          # [B, decoder_channels[1], H/8, W/8]
        skip2 = skip_features[2]                   # layer2 output: [B, 128, H/8, W/8]
        conv2 = self.decoder['conv2'](torch.cat([up2, skip2], dim=1))  # [B, decoder_channels[1], H/8, W/8]

        # Decoder Step 3
        up3 = self.decoder['up3'](conv2)          # [B, decoder_channels[2], H/4, W/4]
        skip3 = skip_features[1]                   # layer1 output: [B, 64, H/4, W/4]
        conv3 = self.decoder['conv3'](torch.cat([up3, skip3], dim=1))  # [B, decoder_channels[2], H/4, W/4]

        # Decoder Step 4
        up4 = self.decoder['up4'](conv3)          # [B, decoder_channels[3], H/2, W/2]
        skip4 = skip_features[0]                   # conv1 output: [B, 64, H/2, W/2]
        conv4 = self.decoder['conv4'](torch.cat([up4, skip4], dim=1))  # [B, decoder_channels[3], H/2, W/2]

        # Final Output
        out = self.out_conv(conv4)                # [B, num_classes, H/2, W/2]

        # Upsample to Original Size
        out = nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)  # [B, num_classes, H, W]

        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.channel_attn(x)
        out = out * self.spatial_attn(out)
        return out
class SegViT_CBAM(nn.Module):
    def __init__(self, encoder_name='resnet34', pretrained=True, num_classes=2):
        super(SegViT_CBAM, self).__init__()
        # Encoder selection
        if encoder_name == 'resnet34':
            self.encoder = models.resnet34(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError("Unsupported encoder")
        
        # Modify the first conv to accept single-channel input
        self.encoder.conv1 = nn.Conv2d(1, self.encoder.conv1.out_channels,
                                       kernel_size=self.encoder.conv1.kernel_size,
                                       stride=self.encoder.conv1.stride,
                                       padding=self.encoder.conv1.padding,
                                       bias=self.encoder.conv1.bias is not None)
        if pretrained:
            with torch.no_grad():
                self.encoder.conv1.weight = nn.Parameter(self.encoder.conv1.weight.mean(dim=1, keepdim=True))
        
        # Define the encoder layers and extract skip connections from:
        # conv1, layer1, layer2, and layer3.
        self.encoder_layers = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,  # Skip connection 1
            self.encoder.layer2,  # Skip connection 2
            self.encoder.layer3,  # Skip connection 3
            self.encoder.layer4   # Final features (not used for skip)
        )
        
        # Insert CBAM blocks on selected skip connections
        self.cbam_layer2 = CBAM(encoder_channels[2])  # After layer2
        self.cbam_layer3 = CBAM(encoder_channels[3])  # After layer3
        
        self.transformer = TransformerBridge(
            in_channels=encoder_channels[-1],
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            ff_dim=3072,
            dropout=0.1
        )
        
        decoder_channels = [256, 128, 64, 64]  # For ResNet34
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
        # Extract features and store skip connections
        for i, layer in enumerate(self.encoder_layers):
            out = layer(out)
            if i == 0:
                skip_features.append(out)  # After conv1
            elif i == 4:
                skip_features.append(out)  # After layer1
            elif i == 5:
                out = self.cbam_layer2(out)  # After layer2, with CBAM
                skip_features.append(out)
            elif i == 6:
                out = self.cbam_layer3(out)  # After layer3, with CBAM
                skip_features.append(out)
            # Do not use the final layer as a skip
        transformed = self.transformer(out)
        up1 = self.decoder['up1'](transformed)
        skip1 = skip_features[3]  # From layer3 (after CBAM)
        conv1 = self.decoder['conv1'](torch.cat([up1, skip1], dim=1))
        up2 = self.decoder['up2'](conv1)
        skip2 = skip_features[2]  # From layer2 (after CBAM)
        conv2 = self.decoder['conv2'](torch.cat([up2, skip2], dim=1))
        up3 = self.decoder['up3'](conv2)
        skip3 = skip_features[1]  # From layer1
        conv3 = self.decoder['conv3'](torch.cat([up3, skip3], dim=1))
        up4 = self.decoder['up4'](conv3)
        skip4 = skip_features[0]  # From conv1
        conv4 = self.decoder['conv4'](torch.cat([up4, skip4], dim=1))
        out = self.out_conv(conv4)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out
# ------------------------------
# 4. Video Processing Functions
# ------------------------------

def load_model(model_path, device='cpu'):
    """
    Loads the SegViT model from the saved state dictionary.

    Args:
        model_path (str): Path to the saved model state dict.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        nn.Module: Loaded SegViT model.
    """
    # Initialize the model architecture
    model = SegViT(encoder_name='resnet34', pretrained=False, num_classes=2)  # pretrained=False as we load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def initialize_video_io(input_path, output_path, fps=30.0, target_size=(512, 512), rotate_clockwise90=False):
    """
    Initializes video capture and writer objects, adjusting frame size based on resizing and rotation.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video.
        fps (float): Frames per second for the output video.
        target_size (tuple): Desired frame size (width, height).
        rotate_clockwise90 (bool): Whether to rotate frames by 90 degrees clockwise.

    Returns:
        cap (cv2.VideoCapture): Video capture object.
        out (cv2.VideoWriter): Video writer object.
        frame_size (tuple): (width, height) of the frames after resizing and rotation.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")

    # Get original frame size from input video
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize to target_size
    resized_width, resized_height = target_size

    # Determine output frame size after rotation
    if rotate_clockwise90:
        frame_size = (resized_height, resized_width)  # Width and height are swapped after rotation
    else:
        frame_size = (resized_width, resized_height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs like 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    return cap, out, frame_size

def preprocess_frame(frame, device='cpu', target_size=(512, 512), rotate_clockwise90=False):
    """
    Preprocesses a video frame: resizes and rotates it.

    Args:
        frame (np.array): Original video frame in BGR format.
        device (str): Device to move the tensor to.
        target_size (tuple): Desired frame size (width, height).
        rotate_clockwise90 (bool): Whether to rotate frame by 90 degrees clockwise.

    Returns:
        torch.Tensor: Preprocessed frame tensor [1, 1, H, W].
        np.array: Original frame after resizing and rotation [H, W, 3].
    """
    # Resize frame
    resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    # Rotate frame if needed
    if rotate_clockwise90:
        resized_frame = cv2.rotate(resized_frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert BGR to Grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Convert to PIL Image
    pil_image = Image.fromarray(gray)

    # Define the transform: to tensor and normalize
    transform = ToTensor(augment=False)

    # Apply transform
    image_tensor, _ = transform(pil_image, np.zeros_like(gray))  # Mask is dummy here

    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 1, H, W]

    return image_tensor, resized_frame  # Return resized (and rotated) frame for overlay

def apply_color_map(mask):
    """
    Applies a color map to the binary mask for better visualization.

    Args:
        mask (np.array): Binary mask [H, W].

    Returns:
        np.array: Colored mask [H, W, 3].
    """
    colored_mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return colored_mask

def overlay_mask_on_frame(frame, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlays a segmentation mask on the original frame.

    Args:
        frame (np.array): Original frame in BGR format.
        mask (np.array): Binary mask [H, W] with 0s and 1s.
        color (tuple): BGR color for the overlay.
        alpha (float): Transparency factor for the overlay.

    Returns:
        np.array: Frame with overlay.
    """
    # Ensure mask is boolean
    mask_bool = mask.astype(bool)

    # Create color mask
    color_mask = np.zeros_like(frame)
    color_mask[mask_bool] = color

    # Blend the color mask with the original frame
    overlayed_frame = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

    return overlayed_frame

def apply_crf(image, mask, n_iters=5):
    """
    Applies CRF to refine the segmentation mask.

    Args:
        image (np.array): Original image in RGB format.
        mask (np.array): Predicted mask with binary values (0 or 1).
        n_iters (int): Number of CRF iterations.

    Returns:
        np.array: Refined mask with binary values (0 or 1).
    """
    # Ensure the image is in uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Initialize DenseCRF2D
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)  # Assuming binary segmentation

    # Create unary potentials from softmax probabilities
    # Here, mask is binary: 0 or 1
    probs = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)
    probs[0, :, :] = 1 - mask  # Background probability
    probs[1, :, :] = mask      # Foreground probability
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)

    # Add pairwise Gaussian and Bilateral potentials
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    # Perform inference
    Q = d.inference(n_iters)
    refined_mask = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

    return refined_mask

def process_video(model, cap, out, device='cpu', target_size=(512, 512), rotate_clockwise90=False, post_process_flag=False, crf_flag=False, smooth_flag=False):
    """
    Processes each frame of the input video, performs segmentation, overlays the mask, displays the results, and writes to output.

    Args:
        model (nn.Module): Trained segmentation model.
        cap (cv2.VideoCapture): Video capture object.
        out (cv2.VideoWriter): Video writer object.
        device (str): Device to run on.
        target_size (tuple): Desired frame size (width, height).
        rotate_clockwise90 (bool): Whether to rotate frames by 90 degrees clockwise.
        post_process_flag (bool): Whether to apply morphological post-processing to the mask.
        crf_flag (bool): Whether to apply CRF refinement to the mask.
        smooth_flag (bool): Whether to apply Gaussian smoothing to the mask.
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {frame_count}")

    with torch.no_grad():
        for idx in tqdm(range(frame_count), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {idx}.")
                break

            # Preprocess frame and get resized (and rotated) frame
            input_tensor, resized_frame = preprocess_frame(frame, device=device, target_size=target_size, rotate_clockwise90=rotate_clockwise90)

            # Perform segmentation
            output = model(input_tensor)  # [1, num_classes, H, W]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]

            # No need to resize mask back to original frame size since output video is in resized and rotated size
            pred_original = pred.astype(np.uint8)

            # Optionally, apply post-processing to the mask
            if post_process_flag:
                # Example: Morphological operations to clean up the mask
                kernel = np.ones((3, 3), np.uint8)
                pred_original = cv2.morphologyEx(pred_original, cv2.MORPH_OPEN, kernel, iterations=1)
                pred_original = cv2.morphologyEx(pred_original, cv2.MORPH_DILATE, kernel, iterations=1)

            # Optionally, apply Gaussian smoothing
            if smooth_flag:
                # Apply Gaussian Blur
                pred_original = cv2.GaussianBlur(pred_original.astype(np.float32), (5, 5), 0)
                # Apply threshold
                _, pred_original = cv2.threshold(pred_original, 0.5, 1, cv2.THRESH_BINARY)
                pred_original = pred_original.astype(np.uint8)

            # Optionally, apply CRF refinement
            if crf_flag:
                # Convert frame to RGB for CRF
                resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                refined_mask = apply_crf(
                    resized_frame_rgb, 
                    pred_original, 
                    n_iters=5
                )
                pred_original = refined_mask

            # Overlay mask on the resized (and rotated) frame
            overlayed_frame = overlay_mask_on_frame(resized_frame, pred_original, color=(0, 255, 0), alpha=0.2)

            # Apply color map to the mask for better visualization (optional)
            colored_mask = apply_color_map(pred_original)

            # Display the results in separate windows
            cv2.imshow('Original Video', resized_frame)
            cv2.imshow('Segmented Mask', colored_mask)  # Colored mask for better visibility
            cv2.imshow('Overlayed Video', overlayed_frame)

            # Optionally, display frame rate and progress
            current_fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(overlayed_frame, f'FPS: {current_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlayed_frame, f'Frame: {idx+1}/{frame_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)

            # Write the overlayed frame to the output video
            out.write(overlayed_frame)

            # Wait for 1 ms and check if 'q' key is pressed to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early termination triggered by user.")
                break

    print("Video processing complete.")
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

# ------------------------------
# 5. Complete Main Function
# ------------------------------

def main():
    """
    Main function to perform catheter segmentation on a video using the trained SegViT model.
    """
    # ------------------------------
    # Define Your Model and Video Paths Here
    # ------------------------------
    model_path = "best_segvit_model5(improved).pth"  # Path to your trained model
    input_video_path = "/home/yanghehao/tracklearning/DATA/Tom.mp4"  # Replace with your video path
    output_video_path = "/home/yanghehao/tracklearning/DATA/TomOut.mp4"  # Replace with desired output path

    # ------------------------------
    # Define Resizing and Rotation Parameters
    # ------------------------------
    res_scale =16
    target_size = (480, 320)  # Desired frame size (width, height). Change as needed.
    rotate_clockwise90 = True  # Set to True to rotate frames by 90 degrees clockwise

    # ------------------------------
    # Hyperparameters
    # ------------------------------
    fps = 60.0  # Frames per second for the output video
    post_process_flag = False# Set to True to apply morphological post-processing to the mask
    crf_flag = False  # Set to True to apply CRF refinement to the mask
    smooth_flag = False # Set to True to apply Gaussian smoothing to the mask

    # ------------------------------
    # Device Configuration
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # ------------------------------
    # Load the Model
    # ------------------------------
    print("[Main] Loading the model...")
    model = load_model(model_path, device=device)
    print("Model loaded successfully.")

    # ------------------------------
    # Initialize Video Capture and Writer
    # ------------------------------
    print("[Main] Initializing video capture and writer...")
    cap, out, frame_size = initialize_video_io(
        input_path=input_video_path, 
        output_path=output_video_path, 
        fps=fps, 
        target_size=target_size, 
        rotate_clockwise90=rotate_clockwise90
    )
    print("Video capture and writer initialized.")

    # ------------------------------
    # Process the Video
    # ------------------------------
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
        smooth_flag=smooth_flag
    )

    # ------------------------------
    # Release Resources
    # ------------------------------
    cap.release()
    out.release()
    print("[Main] Video saved successfully.")

# ------------------------------
# 6. Execute the Main Function
# ------------------------------

if __name__ == "__main__":
    main()
