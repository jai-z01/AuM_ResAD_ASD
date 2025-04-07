import os
import math
import random
from typing import List, Dict
from PIL import Image
import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from .datasets.mvtec import MVTEC
from .datasets.visa import VISA
import cv2
import gc
from tqdm import tqdm


class BoundaryAverager:
    def __init__(self, num_levels=3):
        self.boundaries = [0 for _ in range(num_levels)]
    
    def update_boundary(self, boundary, level, momentum=0.9):
        lvl_boundary = self.boundaries[level]
        lvl_boundary = lvl_boundary * momentum + (1 - momentum) * boundary
        self.boundaries[level] = lvl_boundary
        
    def get_boundary(self, level):
        return self.boundaries[level]
    
    
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

# def get_matched_ref_features(features: List[Tensor], ref_features: List[Tensor]) -> List[Tensor]:
#     """
#     Get matched reference features for one class.
#     """
#     torch.cuda.empty_cache()  # Free GPU memory before processing

#     matched_ref_features = []
#     chunk_size = 10000  # Reduce memory usage by processing in chunks

#     for layer_id in range(len(features)):
#         feature = features[layer_id]
#         B, C, H, W = feature.shape
#         feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
#         feature_n = F.normalize(feature, p=2, dim=1)

#         coreset = ref_features[layer_id].detach()  # Detach to prevent gradient tracking

#         # Process in chunks to avoid OOM
#         matched_features = []
#         for coreset_chunk in coreset.split(chunk_size, dim=0):
#             coreset_n = F.normalize(coreset_chunk, p=2, dim=1)
#             dist = feature_n @ coreset_n.T  # Compute similarity

#             cidx = torch.argmax(dist, dim=1)
#             index_feats = coreset_chunk[cidx]
            
#             index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
#             matched_features.append(index_feats)

#         # Concatenate processed features
#         matched_ref_features.append(torch.cat(matched_features, dim=0))

#         # Free up memory
#         del feature, feature_n, coreset, coreset_n, dist, cidx, index_feats
#         torch.cuda.empty_cache()

#     return matched_ref_features

def get_matched_ref_features(features: List[Tensor], ref_features: List[Tensor], chunk_size=1000) -> List[Tensor]:
    """
    Get matched reference features for one class with memory-efficient chunk-wise processing.
    
    Args:
        features (List[Tensor]): List of feature maps from different layers.
        ref_features (List[Tensor]): List of reference features from different layers.
        chunk_size (int): Number of rows processed at once to reduce memory usage.
    
    Returns:
        List[Tensor]: List of matched reference features.
    """
    matched_ref_features = []
    
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
        feature_n = F.normalize(feature, p=2, dim=1)

        coreset = ref_features[layer_id]  # (N2, C)
        coreset_n = F.normalize(coreset, p=2, dim=1)

        num_features = feature_n.shape[0]
        num_coreset = coreset_n.shape[0]
        
        max_values = []
        max_indices = []

        # Process feature_n in chunks
        for i in range(0, num_features, chunk_size):
            chunk = feature_n[i:i + chunk_size]  # Take a small chunk of features
            
            max_chunk_values = []
            max_chunk_indices = []

            # Process coreset_n in chunks to prevent large transpose matrix
            for j in range(0, num_coreset, chunk_size):
                coreset_chunk = coreset_n[j:j + chunk_size]  # Small chunk of coreset
                dist_chunk = torch.matmul(chunk, coreset_chunk.T)  # Compute distances

                # Find max similarity within each coreset chunk
                chunk_max_vals, chunk_max_idxs = torch.max(dist_chunk, dim=1)
                
                max_chunk_values.append(chunk_max_vals)
                max_chunk_indices.append(chunk_max_idxs + j)  # Offset index by chunk start

            # Merge max values and indices across coreset chunks
            max_chunk_values = torch.stack(max_chunk_values, dim=0)
            max_chunk_indices = torch.stack(max_chunk_indices, dim=0)
            
            # Find the best match across all chunks
            best_chunk_idx = torch.argmax(max_chunk_values, dim=0)
            best_values = max_chunk_values[best_chunk_idx, torch.arange(chunk.shape[0])]
            best_indices = max_chunk_indices[best_chunk_idx, torch.arange(chunk.shape[0])]

            max_values.append(best_values)
            max_indices.append(best_indices)

        # Merge all chunks
        max_indices = torch.cat(max_indices, dim=0)
        index_feats = coreset[max_indices]
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        matched_ref_features.append(index_feats)

    return matched_ref_features


def get_residual_features(features: List[Tensor], ref_features: List[Tensor], pos_flag: bool = False) -> List[Tensor]:
    residual_features = []
    for layer_id in range(len(features)):
        fi = features[layer_id]  # (B, dim, h, w)
        pi = ref_features[layer_id]  # (B, dim, h, w)
        
        if not pos_flag:
            ri = fi - pi
        else:
            ri = F.mse_loss(fi, pi, reduction='none')
        residual_features.append(ri)
    
    return residual_features
        

def load_reference_features(root_dir: str, class_name: str, device: torch.device) -> List[Tensor]:
    """
    Load reference features for one class.
    """
    layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
    layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
    layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
    
    layer1_refs = torch.from_numpy(layer1_refs).to(device)
    layer2_refs = torch.from_numpy(layer2_refs).to(device)
    layer3_refs = torch.from_numpy(layer3_refs).to(device)
    
    return layer1_refs, layer2_refs, layer3_refs

def get_random_normal_images(root, class_name, num_shot=4):
    # Ensure root and class_name are strings
    if not isinstance(root, str):
        raise TypeError(f"Expected 'root' to be a string, but got {type(root)}")
    
    if not isinstance(class_name, str):
        raise TypeError(f"Expected 'class_name' to be a string, but got {type(class_name)}")

    # Define correct path for DCASE dataset
    root_dir = os.path.join(root, class_name, 'train')

    # Check if directory exists
    if not os.path.exists(root_dir):
        raise ValueError(f"Directory not found: {root_dir}")

    # Get list of files
    filenames = os.listdir(root_dir)
    
    # Ensure the directory is not empty
    if len(filenames) == 0:
        raise ValueError(f"No files found in {root_dir}")

    # Select random samples
    n_idxs = np.random.randint(len(filenames), size=num_shot).tolist()
    normal_paths = [os.path.join(root_dir, filenames[n_idx]) for n_idx in n_idxs]

    return normal_paths

def load_and_transform_spectrograms(paths, device, target_size=(224, 224)):
    images = []
    
    for path in paths:
        if path.endswith(".npy"):
            spectrogram = np.load(path, allow_pickle=True)  # Load spectrogram (F, T)
            spectrogram = spectrogram[np.newaxis, :, :]  # Add channel dim: (1, F, T)

        elif path.endswith((".jpg", ".png")):  
            spectrogram = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            spectrogram = spectrogram[np.newaxis, :, :]  # Convert to (1, H, W)

        else:
            print(f"⚠️ Skipping unsupported file: {path}")
            continue

        # Resize
        spectrogram = cv2.resize(spectrogram.squeeze(0), target_size, interpolation=cv2.INTER_CUBIC)
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Back to (1, 224, 224)

        images.append(torch.tensor(spectrogram, dtype=torch.float32))

    images = torch.stack(images).to(device)  # Final shape (B, 1, 224, 224)

    return images


import torch.nn.functional as F

def get_mc_reference_features(encoder, root, class_names, device, num_shot=4):
    reference_features = {}
    class_names = np.unique(class_names)
    conv1 = nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1).to(device)   # Stride=2 downsamples to 98x384
    conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1).to(device) # 49x192
    conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1).to(device)
    for class_name in class_names:
        normal_paths = get_random_normal_images(root, class_name, num_shot)
        images = load_and_transform_spectrograms(normal_paths, device)

        with torch.no_grad():
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            features = encoder(images).unsqueeze(1)
            feature1 = conv1(features)  # [32, 256, 98, 384]
            feature2 = conv2(feature1)      # [32, 512, 49, 192]
            feature3 = conv3(feature2)      # [32, 1024, 25, 96]

            # If necessary, interpolate to exact sizes
            feature1 = F.interpolate(feature1, size=(56, 56), mode="bilinear", align_corners=False)
            feature2 = F.interpolate(feature2, size=(28, 28), mode="bilinear", align_corners=False)
            feature3 = F.interpolate(feature3, size=(14, 14), mode="bilinear", align_corners=False)
            features = [feature1, feature2, feature3]

            if isinstance(features, list):
                features = [F.adaptive_avg_pool2d(f, (1, 1)).view(f.shape[0], -1) for f in features]
            else:
                features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.shape[0], -1)

            reference_features[class_name] = features

    return reference_features

def get_mc_matched_ref_features(features: List[Tensor], class_names: List[str],
                                ref_features: Dict[str, List[Tensor]]) -> List[Tensor]:
    """
    Get matched reference features for multiple classes.
    """
    matched_ref_features = [[] for _ in range(len(features))]
    for idx, c in enumerate(class_names):  # for each image
        ref_features_c = ref_features[c]
        
        for layer_id in range(len(features)):  # for all layers of one image
            feature = features[layer_id][idx:idx+1]
            _, C, H, W = feature.shape
            
            feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
            feature_n = F.normalize(feature, p=2, dim=1)
            coreset = ref_features_c[layer_id]  # (N2, C)
            coreset_n = F.normalize(coreset, p=2, dim=1)
            dist = feature_n @ coreset_n.T  # (N1, N2)
            cidx = torch.argmax(dist, dim=1)
            index_feats = coreset[cidx]
            index_feats = index_feats.permute(1, 0).reshape(C, H, W)
            matched_ref_features[layer_id].append(index_feats)
            
    matched_ref_features = [torch.stack(item, dim=0) for item in matched_ref_features]
    
    return matched_ref_features


def calculate_metrics(scores, labels, gt_masks, pro=True, only_max_value=True):
    """
    Args:
        scores (np.ndarray): shape (N, H, W).
        labels (np.ndarray): shape (N, ), 0 for normal, 1 for abnormal.
        gt_masks (np.ndarray): shape (N, H, W).
    """
    # average precision
    pix_ap = round(average_precision_score(gt_masks.flatten(), scores.flatten()), 5)
    # f1 score, f1 score is to balance the precision and recall
    # f1 score is high means the precision and recall are both high
    precisions, recalls, _ = precision_recall_curve(gt_masks.flatten(), scores.flatten())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    pix_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
    # roc auc
    pix_auc = round(roc_auc_score(gt_masks.flatten(), scores.flatten()), 5)
    
    _, h, w = scores.shape
    size = h * w
    if only_max_value:
        topks = [1]
    else:
        topks = [int(size*p) for p in np.arange(0.01, 0.41, 0.01)]
        topks = [1, 100] + topks
    img_aps, img_aucs, img_f1_scores = [], [], []
    for topk in topks:
        img_scores = get_image_scores(scores, topk)
        img_ap = round(average_precision_score(labels, img_scores), 5)
        precisions, recalls, _ = precision_recall_curve(labels, img_scores)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        img_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
        img_auc = round(roc_auc_score(labels, img_scores), 5)
        img_aps.append(img_ap)
        img_aucs.append(img_auc)
        img_f1_scores.append(img_f1_score)
    img_ap, img_auc, img_f1_score = np.max(img_aps), np.max(img_aucs), np.max(img_f1_scores)
        
    if pro:
        pix_aupro = calculate_aupro(gt_masks, scores)
    else:
        pix_aupro = -1
    
    return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro


def get_image_scores(scores, topk=1):
    scores_ = torch.from_numpy(scores)
    img_scores = torch.topk(scores_.reshape(scores_.shape[0], -1), topk, dim=1)[0]
    img_scores = torch.mean(img_scores, dim=1)
    img_scores = img_scores.cpu().numpy()
        
    return img_scores


def calculate_aupro(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    if fprs.shape[0] <= 2:
        return 0.5
    else:
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
        pro_auc = auc(fprs, pros[idxes])
        return pro_auc

# def applying_EFDM(input_features_list, ref_features_list, alpha=0.5):
#     alpha = 1 - alpha
#     aligned_features_list = []

#     for l in range(len(input_features_list)):
#         input_features, ref_features = input_features_list[l], ref_features_list[l]
#         B, C, W, H = input_features.shape

#         # Move to CPU and convert to float16 to save memory
#         input_features_r = input_features.reshape(B, C, -1).detach().cpu().to(torch.float16)
#         ref_features_r = ref_features.reshape(B, C, -1).detach().cpu().to(torch.float16)

#         # Process in smaller chunks to avoid memory overload
#         chunk_size = 1024  # Adjust if needed
#         aligned_chunks = []

#         for i in range(0, input_features_r.shape[-1], chunk_size):
#             input_chunk = input_features_r[:, :, i:i+chunk_size]
#             ref_chunk = ref_features_r[:, :, i:i+chunk_size]

#             # Sort in smaller chunks
#             sorted_input_chunk, inds = torch.sort(input_chunk, dim=-1)
#             sorted_ref_chunk = torch.sort(ref_chunk, dim=-1)[0]

#             # Compute in-place to save memory
#             sorted_ref_chunk.mul_(alpha).add_(sorted_input_chunk.mul(1 - alpha))

#             # Get inverse indices
#             inv_inds = inds.argsort(-1)

#             # Apply inverse sorting
#             aligned_chunk = sorted_ref_chunk.gather(-1, inv_inds)

#             aligned_chunks.append(aligned_chunk)

#         # Concatenate processed chunks back together
#         aligned_features = torch.cat(aligned_chunks, dim=-1)

#         # Move back to GPU in manageable chunks
#         aligned_features = aligned_features.to("cuda").view(B, C, W, H)
#         aligned_features_list.append(aligned_features)

#         # Cleanup memory
#         del input_features_r, ref_features_r, sorted_input_chunk, sorted_ref_chunk, inv_inds, aligned_chunks
#         gc.collect()
#         torch.cuda.empty_cache()

#     return aligned_features_list

def applying_EFDM(input_features_list, ref_features_list, alpha=0.5):
    """
    Args:
        input_features (Tensor): shape of (B, C, H, W).
        ref_features (Tensor): normal reference features, (B, C, H, W).
    """
    alpha = 1 - alpha
    aligned_features_list = []
    print(len(input_features_list),input_features_list[0].shape)
    for l in tqdm(range(len(input_features_list))):
        print(l)
        input_features, ref_features = input_features_list[l], ref_features_list[l]
        B, C, W, H = input_features.shape

        input_features_r = input_features.reshape(B, C, -1).detach().cpu()
        ref_features_r = ref_features.reshape(B, C, -1).detach().cpu()

        sorted_input_features, inds = torch.sort(input_features_r)
        sorted_ref_features, _ = torch.sort(ref_features_r)
        aligned_features = sorted_input_features +  alpha * (sorted_ref_features - sorted_input_features) 
        inv_inds = inds.argsort(-1)
        aligned_features = aligned_features.cuda()
        inv_inds = inv_inds.cuda()
        aligned_features = aligned_features.gather(-1, inv_inds)
        aligned_features = aligned_features.view(B, C, W, H)
        aligned_features_list.append(aligned_features)

    return aligned_features_list


