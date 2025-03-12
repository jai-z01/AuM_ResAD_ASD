import os
import argparse
import numpy as np
from PIL import Image

import torch
import tqdm
import timm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from src.models.mamba_models import AudioMamba
import torch.nn.functional as F

#from models.imagebind import ImageBindModel

class DCASEDataset(Dataset):
    
    def __init__(self, root: str, class_name: str, train: bool = True, **kwargs) -> None:
        self.root = root
        self.class_name = class_name
        self.train = train

        self.image_paths, self.labels = self._load_data(class_name)

        self.transform = T.Compose([
            T.Resize(kwargs.get('img_size', 224), T.InterpolationMode.BICUBIC),
            T.CenterCrop(kwargs.get('crp_size', 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx], self.labels[idx]
        img = Image.open(image_path).convert('L')
        img = self.transform(img)
        return img, label, self.class_name

    def _load_data(self, class_name):
        image_paths, labels = [], []
        class_dir = os.path.join(self.root, class_name)
        
        if self.train:
            train_dir = os.path.join(class_dir, "train")
            image_paths.extend(sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".jpg")]))
            labels.extend([0] * len(image_paths))  # Training has only normal images
        
        else:
            test_normal_dir = os.path.join(class_dir, "test", "normal")
            test_anomaly_dir = os.path.join(class_dir, "test", "anomaly")

            normal_images = sorted([os.path.join(test_normal_dir, f) for f in os.listdir(test_normal_dir) if f.endswith(".jpg")])
            anomaly_images = sorted([os.path.join(test_anomaly_dir, f) for f in os.listdir(test_anomaly_dir) if f.endswith(".jpg")])

            image_paths.extend(normal_images)
            labels.extend([0] * len(normal_images))

            image_paths.extend(anomaly_images)
            labels.extend([1] * len(anomaly_images))

        return image_paths, labels


SETTINGS = {'dcase_eval': ['3DPrinter', 'AirCompressor', 'BrushlessMotor',
                    'HairDryer', 'HoveringDrone', 'RoboticArm',
                    'Scanner', 'ToothBrush', 'ToyCircuit'], 
            'dcase': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']}

def extract_features(args, encoder, dataset_class):
    device = 'cuda:0'
    root_dir = args.few_shot_dir
    image_size = 224
    conv1 = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1).to(device)  # Output: [B, 256, 196, 768]
    conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1).to(device)
    conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1).to(device)
    for class_name in SETTINGS[args.dataset]:
        dataset = dataset_class(root_dir, class_name=class_name, train=True, img_size=image_size, crp_size=image_size)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False)
        
        layer1, layer2, layer3 = [],[],[]
        torch.cuda.empty_cache()

        for batch in tqdm.tqdm(loader):
            images, _, _ = batch
            with torch.no_grad():
                feature_maps = encoder(images.to(device))
                feature_maps = feature_maps.unsqueeze(1)
                torch.cuda.empty_cache()
                feature1 = F.interpolate(feature_maps, size=(56, 56), mode="bilinear", align_corners=False)
                feature1 = conv1(feature1) 
                feature2 = F.interpolate(feature1, size=(28, 28), mode="bilinear", align_corners=False)
                feature2 = conv2(feature2)
                feature3 = F.interpolate(feature2, size=(14, 14), mode="bilinear", align_corners=False)
                feature3 = conv3(feature3)
            
            #feature_maps = feature_maps.repeat(1, 3, 1, 1)
            layer1.append(feature1)
            layer2.append(feature2)
            layer3.append(feature3)
            del feature_maps, feature1, feature2, feature3  
            torch.cuda.empty_cache()

        layer1_features = torch.cat(layer1, dim=0)
        layer2_features = torch.cat(layer2, dim=0)
        layer3_features = torch.cat(layer3, dim=0)
        print(layer1_features.shape)
        print(layer2_features.shape)
        print(layer3_features.shape)
        torch.cuda.empty_cache()
        with torch.no_grad():
            layer1_features = layer1_features.permute(0, 2, 3, 1).reshape(-1, 256)
            layer2_features = layer2_features.permute(0, 2, 3, 1).reshape(-1, 512)
            layer3_features = layer3_features.permute(0, 2, 3, 1).reshape(-1, 1024)
        
        os.makedirs(os.path.join(args.save_dir, class_name), exist_ok=True)
        
        np.save(os.path.join(args.save_dir, class_name, 'layer1.npy'), layer1_features.detach().cpu().numpy())
        np.save(os.path.join(args.save_dir, class_name, 'layer2.npy'), layer2_features.detach().cpu().numpy())
        np.save(os.path.join(args.save_dir, class_name, 'layer3.npy'), layer3_features.detach().cpu().numpy())
        del layer1_features, layer2_features, layer3_features, layer1, layer2, layer3
        torch.cuda.empty_cache()



def main(args):
    device = 'cuda:0'
    encoder = AudioMamba(aum_pretrain=True, aum_pretrain_path="base_scratch-as_20k-14.05.pth").to(device).eval()
    #encoder = AudioMamba().to(device).eval()
    extract_features(args, encoder, DCASEDataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dcase")
    parser.add_argument('--few_shot_dir', type=str, default="./ResAD/datasets/data/dev_spectrograms")
    parser.add_argument('--save_dir', type=str, default="/mnt/e/ref_features_aum/dcase")
    
    args = parser.parse_args()
    main(args)