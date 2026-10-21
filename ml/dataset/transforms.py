import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image

class SegmentationTransforms:
    """
    Synchronized transforms for image and mask pairs.
    Applies geometric transformations identically to both image and mask,
    while applying photometric augmentations strictly to the RGB image.
    Uses nearest-neighbor interpolation for masks to prevent label corruption.
    """
    def __init__(
        self,
        image_size=(256, 256),
        is_train=True,
        hflip_prob=0.5,
        vflip_prob=0.3,
        rotation_degrees=15,
        brightness_jitter=0.15,
        contrast_jitter=0.15
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_degrees = rotation_degrees
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

        # ImageNet normalization standard for transfer learning
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image, mask: Image.Image):
        # 1. Resize: Bilinear for image, Nearest for categorical mask
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            # Horizontal flip
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Rotation (small angles)
            if self.rotation_degrees > 0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # Photometric adjustments (Image only)
            if self.brightness_jitter > 0:
                factor = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
                image = TF.adjust_brightness(image, factor)

            if self.contrast_jitter > 0:
                factor = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
                image = TF.adjust_contrast(image, factor)

        # 2. Convert image to Tensor & Normalize
        image_tensor = TF.to_tensor(image)  # [3, H, W], in range [0, 1]
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)

        # 3. Convert mask to LongTensor / FloatTensor
        # Ensure mask is 0 or 1 binary values
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.float32)
        # Binarize in case of any artifact:
        mask_tensor = (mask_tensor > 0.5).float()  # [1, H, W]

        return image_tensor, mask_tensor
