from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
import numpy as np
from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode

class CIFAR10Dataset(BaseDepthDataset):
    def __init__(self, split="train", **kwargs):
        super().__init__(
            min_depth=0.0,
            max_depth=1.0,
            has_filled_depth=True,
            name_mode=DepthFileNameMode.rgb_id,
            **kwargs
        )
        
        self.cifar10 = CIFAR10(root='./data', train=(split == "train"), download=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        
        # Convert image to tensor and normalize
        image = self.transform(image)
        
        # Create a "depth" tensor representing the class
        depth = torch.full((1, 32, 32), label / 9.0)  # Normalize to [0, 1]
        
        return {
            "rgb_int": (image * 255).byte(),
            "rgb_norm": image,
            "depth_raw_linear": depth,
            "depth_filled_linear": depth,
            "valid_mask_raw": torch.ones_like(depth, dtype=torch.bool),
            "valid_mask_filled": torch.ones_like(depth, dtype=torch.bool),
        }

    def _read_depth_file(self, rel_path):
        # This method is not used in this implementation
        pass

    def _get_valid_mask(self, depth: torch.Tensor):
        return torch.ones_like(depth, dtype=torch.bool)
