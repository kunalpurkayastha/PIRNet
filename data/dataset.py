import os
import torch
from torch.utils.data import Dataset
from .preprocess import load_and_preprocess, augment

class MRIDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.images = self._load_image_paths()

    def _load_image_paths(self):
        image_dir = os.path.join(self.root_dir, 'train' if self.train else 'test')
        return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = load_and_preprocess(img_path)
        
        if self.train:
            image = augment(image)
        
        if self.transform:
            image = self.transform(image)
        
        # Assuming binary segmentation, target is the same as input for simplicity
        # In practice, you'd load a separate segmentation mask
        target = image
        
        return torch.from_numpy(image).float(), torch.from_numpy(target).float()