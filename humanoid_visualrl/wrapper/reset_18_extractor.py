import torch
import torch.nn as nn
from torchvision import models


class Reset18Extractor:
    def __init__(self, device: torch.device):
        """Initialize ResNet-18 as a frozen feature extractor."""
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(pretrained=True)

        # Remove the final classification layer to get feature representations
        # ResNet-18 outputs 512-dimensional features before the final FC layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Set to evaluation mode
        self.resnet.eval()

        # Move to device
        self.resnet = self.resnet.to(device)

        # # Feature dimension: ResNet-18 outputs 512-dim features
        # self.resnet_feature_dim = 512

    def extract_visual_features(self, rgb_images):
        """Extract features from RGB images using ResNet-18.

        Args:
            rgb_images: tensor of shape (batch_size, height, width, channels) or (batch_size, channels, height, width)

        Returns:
            features: tensor of shape (batch_size, 512)
        """
        if rgb_images is None:
            raise ValueError("RGB images cannot be None")

        # Ensure images are in the right format (batch_size, channels, height, width)
        if rgb_images.dim() == 4 and rgb_images.shape[-1] in [1, 3, 4]:  # HWC format
            rgb_images = rgb_images.permute(0, 3, 1, 2)  # Convert to CHW

        # Ensure we have 3 channels (RGB)
        if rgb_images.shape[1] == 4:  # RGBA
            rgb_images = rgb_images[:, :3, :, :]  # Take only RGB channels
        elif rgb_images.shape[1] == 1:  # Grayscale
            rgb_images = rgb_images.repeat(1, 3, 1, 1)  # Convert to RGB

        # Normalize to [0, 1] if needed
        if rgb_images.dtype == torch.uint8:
            rgb_images = rgb_images.float() / 255.0

        # ResNet expects images normalized with ImageNet stats
        # But for simplicity, we'll use the raw normalized images
        # You might want to apply ImageNet normalization for better features:
        # mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        # std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        # rgb_images = (rgb_images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        with torch.no_grad():
            features = self.resnet(rgb_images)
            # Remove spatial dimensions (global average pooling is already applied)
            features = features.view(features.size(0), -1)  # (batch_size, 512)

        return features
