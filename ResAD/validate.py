import warnings
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from .models.modules import get_position_encoding
from .models.utils import get_logp
from .utils import get_residual_features, get_matched_ref_features
from .utils import calculate_metrics, applying_EFDM
from .losses.utils import get_logp_a
from torch.cuda.amp import autocast

warnings.filterwarnings('ignore')


def validate(args, encoder, vq_ops, constraintor, estimators, test_loader, ref_features, device, class_name):
    vq_ops.eval()
    constraintor.eval()
    for estimator in estimators:  
        estimator.eval()
    
    label_list, gt_mask_list = [], []
    logps1_list = [list() for _ in range(args.feature_levels)]
    logps2_list = [list() for _ in range(args.feature_levels)]
    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Evaluating")
    conv1 = nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1).to(args.device)   # Stride=2 downsamples to 98x384
    conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1).to(args.device) # 49x192
    conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1).to(args.device)
    for idx, batch in enumerate(test_loader):
        progress_bar.update(1)
        
        image, label, mask, _ = batch    
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())
        
        image = image.to(device)
        print("Image Shape:", image.shape)
        size = image.shape[-1]
        
        with torch.no_grad():
            if args.backbone == 'aum':
                features = encoder(image).unsqueeze(1)
                feature1 = conv1(features)  # [32, 256, 98, 384]
                feature2 = conv2(feature1)      # [32, 512, 49, 192]
                feature3 = conv3(feature2)      # [32, 1024, 25, 96]

                # If necessary, interpolate to exact sizes
                feature1 = F.interpolate(feature1, size=(56, 56), mode="bilinear", align_corners=False)
                feature2 = F.interpolate(feature2, size=(28, 28), mode="bilinear", align_corners=False)
                feature3 = F.interpolate(feature3, size=(14, 14), mode="bilinear", align_corners=False)
                print("Feature1 Shape:", feature1.shape, feature2.shape, feature3.shape)
                features = [feature1, feature2, feature3]
                del feature1, feature2, feature3
                torch.cuda.empty_cache()
                mfeatures = get_matched_ref_features(features, ref_features)
                print("Matched Features Shape:", mfeatures[0].shape, mfeatures[1].shape, mfeatures[2].shape)
                print("fEAT Shape:", features[0].shape, features[1].shape, features[2].shape)
                rfeatures = get_residual_features(features, mfeatures, pos_flag=True)
                print("Residual Features Shape:", rfeatures[0].shape, rfeatures[1].shape, rfeatures[2].shape)
                del mfeatures
                torch.cuda.empty_cache()
            else:
                features = encoder.encode_image_from_tensors(image)
                for i in range(len(features)):
                    b, l, c = features[i].shape
                    features[i] = features[i].permute(0, 2, 1).reshape(b, c, 16, 16)
                mfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, mfeatures)
            with torch.no_grad():
                fdm_features = vq_ops(rfeatures, train=False)
            

            rfeatures = applying_EFDM(rfeatures, fdm_features, alpha=args.fdm_alpha)
            rfeatures = [f.to(torch.float32) for f in rfeatures]
            print("Residual Features before Constraintor: ", rfeatures[0].shape, rfeatures[1].shape, rfeatures[2].shape)
            rfeatures = constraintor(*rfeatures)
            print("Residual Features after Constraintor: ", rfeatures[0].shape, rfeatures[1].shape, rfeatures[2].shape)
        
            for l in range(args.feature_levels):
                e = rfeatures[l]  # BxCxHxW
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                
                # (bs, 128, h, w)
                pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(args.device)
                pos_embed = pos_embed.unsqueeze(0).expand(bs, -1, -1, -1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                estimator = estimators[l]
                torch.cuda.empty_cache()
                with torch.no_grad():
                    with autocast():
                        if args.flow_arch == 'flow_model':
                            z, log_jac_det = estimator(e)  
                        else:
                            print("Before estimator shape:", e.shape)
                            z, log_jac_det = estimator(e, [pos_embed, ])
                z = z.float()
                log_jac_det = log_jac_det.float()
                logps = get_logp(dim, z, log_jac_det)  
                logps = logps / dim  
                logps1_list[l].append(logps.reshape(bs, h, w))
                
                logps_a = get_logp_a(dim, z, log_jac_det)  # logps corresponding to abnormal distribution
                logits = torch.stack([logps, logps_a], dim=-1)  # (N, 2)
                sa = torch.softmax(logits, dim=-1)[:, 1]
                logps2_list[l].append(sa.reshape(bs, h, w))
    
    progress_bar.close()
    
    labels = np.concatenate(label_list)
    gt_masks = np.concatenate(gt_mask_list, axis=0)
    scores1 = convert_to_anomaly_scores(logps1_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
    scores2 = aggregate_anomaly_scores(logps2_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
    
    img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1 = calculate_metrics(scores1, labels, gt_masks, pro=False, only_max_value=True)
    img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2 = calculate_metrics(scores2, labels, gt_masks, pro=False, only_max_value=True)
    
    scores = (scores1 + scores2) / 2
    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(scores, labels, gt_masks, pro=False, only_max_value=True)
    
    metrics = {}
    metrics['scores1'] = [img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1]
    metrics['scores2'] = [img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2]
    metrics['scores'] = [img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro]
    
    return metrics


def convert_to_anomaly_scores(logps_list, feature_levels=3, class_name=None, size=224):
    normal_map = [list() for _ in range(feature_levels)]
    for l in range(feature_levels):
        logps = torch.cat(logps_list[l], dim=0)  
        logps-= torch.max(logps) # normalize log-likelihoods to (-Inf:0] by subtracting a constant
        probs = torch.exp(logps) # convert to probs in range [0:1]
        # upsample
        normal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score aggregation
    scores = np.zeros_like(normal_map[0])
    for l in range(feature_levels):
        scores += normal_map[l]

    # normality score to anomaly score
    scores = scores.max() - scores 
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores


def aggregate_anomaly_scores(logps_list, feature_levels=3, class_name=None, size=224):
    abnormal_map = [list() for _ in range(feature_levels)]
    for l in range(feature_levels):
        probs = torch.cat(logps_list[l], dim=0)  
        # upsample
        abnormal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score aggregation
    scores = np.zeros_like(abnormal_map[0])
    for l in range(feature_levels):
        scores += abnormal_map[l]
    scores /= feature_levels
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores
