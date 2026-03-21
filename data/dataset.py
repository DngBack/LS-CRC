import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F

class MedicalSegmentationDataset(Dataset):
    """
    Standard loader for binary medical image segmentation (e.g., Kvasir-SEG, CVC-ClinicDB).
    Expects directories `images/` and `masks/`.
    """
    def __init__(self, root_dir, split_file=None, img_size=(256, 256), transform=None, is_train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.is_train = is_train
        
        # Load from split file if provided
        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.filenames = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.filenames = os.listdir(os.path.join(root_dir, "images"))

        self.images_dir = os.path.join(self.root_dir, "images")
        self.masks_dir = os.path.join(self.root_dir, "masks")
        
        # Standard preprocessing
        self.img_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.images_dir, filename)
        # Handle variations like .jpg vs .png for masks
        mask_path = os.path.join(self.masks_dir, filename)
        if not os.path.exists(mask_path):
            mask_path = mask_path.replace('.jpg', '.png')
            
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float() # Binary mask
        
        # Simple data augmentation
        if self.is_train and self.transform is not None:
            # You can inject Albumentations or custom transforms here
            pass

        # Compute Subgroup Tags
        subgroup_tags = self._compute_subgroup_tags(mask)
        
        return image, mask, subgroup_tags, filename
        
    def _compute_subgroup_tags(self, mask):
        """
        Computes the subgroup properties: Size and Boundary Complexity.
        Size: >15% is 'Large', 5-15% is 'Medium', <5% is 'Small'
        Complexity: Boundary pixels / Area
        """
        area = mask.sum().item()
        total_pixels = mask.numel()
        
        # Size Tag
        ratio = area / total_pixels
        if ratio >= 0.15:
            size_tag = 2 # Large
        elif ratio >= 0.05:
            size_tag = 1 # Medium
        else:
            size_tag = 0 # Small
            
        # Boundary Complexity Tag
        # Simple finite differences to compute boundary length
        if area > 0:
            pooled = F.max_pool2d(mask.unsqueeze(0), kernel_size=3, stride=1, padding=1)
            boundary = (pooled - mask.unsqueeze(0)).sum().item()
            complexity = boundary / area
            comp_tag = 1 if complexity > 0.1 else 0 # High vs Low
        else:
            comp_tag = 0 # No object = Low complexity
            
        return {'size': size_tag, 'complexity': comp_tag}

# Fallback synthetic dataset from PoC to ensure runnable code even without downloading data
import scipy.ndimage
class SyntheticMedicalDataset(Dataset):
    """A synthetic fallback dataset if the real data is not found."""
    def __init__(self, num_samples, img_size=(256, 256), is_train=False):
        self.num_samples = num_samples
        self.img_size = img_size[0]
        self.is_train = is_train

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        cx, cy = np.random.randint(self.img_size//4, 3*self.img_size//4, size=2)
        r = np.random.randint(self.img_size//8, self.img_size//4)
        y, x = np.ogrid[-cy:self.img_size-cy, -cx:self.img_size-cx]
        mask_region = x*x + y*y <= r*r
        mask[mask_region] = 1.0
        
        img_blurred = scipy.ndimage.gaussian_filter(mask, sigma=3.0)
        noise = np.random.normal(0, 0.4, (self.img_size, self.img_size))
        img_final = np.clip(img_blurred + noise, 0, 1)
        img[..., 0] = img_final
        img[..., 1] = img_final
        img[..., 2] = img_final

        image = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()
        
        area = mask.sum().item()
        total_pixels = mask.numel()
        ratio = area / total_pixels
        size_tag = 2 if ratio >= 0.15 else (1 if ratio >= 0.05 else 0)
        
        return image, mask, {'size': size_tag, 'complexity': 0}, f"synth_{idx}"

def get_dataloader(root_dir, batch_size=8, split='train', img_size=(256,256), num_workers=4):
    """Factory function for dataloaders."""
    is_train = (split == 'train')
    
    if os.path.exists(root_dir):
        split_file = os.path.join(root_dir, f"{split}.txt")
        dataset = MedicalSegmentationDataset(root_dir, split_file=split_file, img_size=img_size, is_train=is_train)
    else:
        print(f"Warning: Directory {root_dir} not found. Returning SYNTHETIC dataset for {split} split!")
        samples = {'train': 600, 'val': 100, 'cal': 150, 'test': 150}.get(split, 100)
        dataset = SyntheticMedicalDataset(samples, img_size=img_size, is_train=is_train)
        
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
