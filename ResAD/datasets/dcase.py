import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DCASE2024(Dataset):
    
    CLASS_NAMES = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']

    def __init__(self, 
                 root: str,
                 machine_types: list,
                 train: bool = True,
                 normalize: str = "logmel",
                 transform: T.Compose = None,
                 **kwargs):
        """
        Args:
            root (str): Root directory containing spectrograms.
            machine_types (list): List of machine types to include.
            train (bool): Whether to load training or test data.
            normalize (str): Normalization method ("logmel" or "none").
            transform (T.Compose): Custom transform function (optional).
            **kwargs: Additional keyword arguments (e.g., img_size, crp_size).
        """
        self.root = root
        self.machine_types = machine_types
        self.train = train
        self.cropsize = kwargs.get('crp_size', 224)
        
        # ✅ Initialize class mappings before loading data
        self.class_to_idx = {machine: idx for idx, machine in enumerate(self.machine_types)}
        self.idx_to_class = {idx: machine for machine, idx in self.class_to_idx.items()}

        # ✅ Load data paths
        self.image_paths, self.labels, self.class_names = self._load_data()

        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif normalize == "logmel":
           self.transform = T.Compose([
                T.Resize(kwargs.get('img_size', 224), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])

        else:  # No normalization
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size', 224), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.cropsize),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        image_path, label, class_name = self.image_paths[idx], self.labels[idx], self.class_names[idx]

        # ✅ Load spectrogram directly as RGB
        image = Image.open(image_path).convert('L')
        
        image = self.transform(image)
        # ✅ Use a dummy mask for compatibility
        mask = torch.ones(image.shape[1:])  # Shape: [H, W]

        return image, label, mask, class_name

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        """Loads dataset file paths and labels."""
        phase = 'train' if self.train else 'test'
        image_paths, labels, class_names = [], [], []

        for machine_type in self.machine_types:
            img_dir = os.path.join(self.root, machine_type, phase)
            
            if self.train:
                # ✅ Load only normal training data
                img_fpath_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
                image_paths.extend(img_fpath_list)
                labels.extend([0] * len(img_fpath_list))  # Training only has normal data (label=0)
                class_names.extend([machine_type] * len(img_fpath_list))
            else:
                # ✅ Load test data from both 'normal' and 'anomaly' folders
                for label, subfolder in enumerate(["normal", "anomaly"]):
                    test_dir = os.path.join(img_dir, subfolder)
                    if os.path.exists(test_dir):
                        test_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')])
                        image_paths.extend(test_files)
                        labels.extend([label] * len(test_files))  # 0 = normal, 1 = anomaly
                        class_names.extend([machine_type] * len(test_files))
        quat_size = 2
        return image_paths[:quat_size], labels[:quat_size], class_names

    def update_class_to_idx(self, class_to_idx):
        """Update class-to-index mapping dynamically."""
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
