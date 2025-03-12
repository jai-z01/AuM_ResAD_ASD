import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .focal_loss import FocalLoss

_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
log_sigmoid = torch.nn.LogSigmoid()


def get_logp_boundary(logps, mask, pos_beta=0.05, margin_tau=0.1, normalizer=10):
    """
    Find the equivalent log-likelihood decision boundaries from normal log-likelihood distribution.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, )
        pos_beta: position hyperparameter: beta
        margin_tau: margin hyperparameter: tau
    """
    normal_logps = logps[mask == 0].detach()
    n_idx = int(((mask == 0).sum() * pos_beta).item())
    sorted_indices = torch.sort(normal_logps)[1]
    
    n_idx = sorted_indices[n_idx]
    b_n = normal_logps[n_idx]  # normal boundary
    b_n = b_n / normalizer

    b_a = b_n - margin_tau  # abnormal boundary

    return b_n, b_a


def calculate_occ_loss(features, mask, target=None):
    """
    Calculate fcdd loss.
    Args:
        features: shape (N, dim)
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
    """
    A = features.norm(dim=1)
    A = torch.sqrt(A + 1) - 1
    
    loss = 0
    if torch.sum(mask == 0) != 0:
        An = A[mask == 0]
        loss_n = torch.sum(An) / An.shape[0]
        loss += loss_n

    # for anomalies, we keep the mapped features as the original features
    if torch.sum(mask == 1) != 0 and target is not None:
        ano_features = features[mask == 1]
        target_features = target[mask == 1]
        loss_a = F.mse_loss(ano_features, target_features)
        loss += loss_a
        
        # Aa = A[mask == 1]
        # loss_a = -torch.log(1 - torch.exp(-torch.sum(Aa) / Aa.shape[0]))
        # loss += loss_a

    return loss


def calculate_log_barrier_occ_loss(features, mask, target=None):
    """
    Calculate Abnormal Invariant OCC loss.
    Args:
        features: shape (N, dim)
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
    """
    A = features.norm(dim=1)
    A = torch.sqrt(A + 1) - 1
    
    Aa = A[mask == 1]
    if torch.sum(mask == 1) != 0:  # second stage, and exist anomalies
        r_max = min(0.9 * Aa.min().item(), 0.4)  # get the minimum abnormal radii as r_max
    else:  # first stage, or no anomalies in second stage
        r_max = 0.4
    
    loss, loss_n, loss_a = 0, 0, 0
    if torch.sum(mask == 0) != 0:
        An = A[mask == 0]
        An_larger = An[An > r_max]  # larger than r_max
        if An_larger.shape[0] != 0:
            weights = torch.exp(An_larger - r_max).detach()
            loss_larger = torch.mean(-log_sigmoid(-(An_larger - r_max)) * weights)
        else:
            loss_larger = 0
        
        loss_n = loss_larger
        loss += loss_n

    # for anomalies, we keep the mapped features as the original features
    if torch.sum(mask == 1) != 0 and target is not None:
        ano_features = features[mask == 1]
        target_features = target[mask == 1]
        loss_mse = F.mse_loss(ano_features, target_features)
        loss_cos = torch.mean(1 - F.cosine_similarity(ano_features, target_features))
        loss_a = loss_mse + loss_cos  # anomaly invariant loss
        
        loss += loss_a

    return loss, loss_n.item() if torch.is_tensor(loss_n) else 0, loss_a.item() if torch.is_tensor(loss_a) else 0


def calculate_orthogonal_regularizer(features, mask, alpha=1.0):
    """
    Calculate orthogonal regularization term.
    Args:
        features: shape (N, dim)
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        alpha: weighting hyperparameter
    """
    nfeatures = features[mask == 0]
    matrix = nfeatures.T @ nfeatures  # (dim, dim)
    I = torch.eye(nfeatures.shape[1], device=nfeatures.device)
    loss = alpha * torch.mean((matrix - I) ** 2) / 2
    
    return loss


def calculate_bi_occ_loss(features, mask, target=None):
    """
    Calculate Abnormal Invariant OCC loss.
    Args:
        features: shape (N, dim)
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
    """
    A = features.norm(dim=1)
    A = torch.sqrt(A + 1) - 1
    
    Aa = A[mask == 1]
    if torch.sum(mask == 1) != 0:  # second stage, and exist anomalies
        r_max = min(0.9 * Aa.min().item(), 0.4)  # get the minimum abnormal radii as r_max
        r_min = r_max * 0.99  
    else:  # first stage, or no anomalies in second stage
        r_max = 0.4
        r_min = 0.99 * 0.4
    
    loss = 0
    if torch.sum(mask == 0) != 0:
        An = A[mask == 0]
        An_larger = An[An > r_max]  # larger than r_max
        An_lower = An[An < r_min]  # lower than r_min
        if An_larger.shape[0] != 0:
            loss_larger = torch.mean(An_larger - r_min)
        else:
            loss_larger = 0
        if An_lower.shape[0] != 0:
            loss_lower = torch.mean(r_max - An_lower)
        else:
            loss_lower = 0
        loss_n = loss_larger + loss_lower
        loss += loss_n

    # for anomalies, we keep the mapped features as the original features
    if torch.sum(mask == 1) != 0 and target is not None:
        ano_features = features[mask == 1]
        target_features = target[mask == 1]
        # l2_dist = F.mse_loss(ano_features, target_features, reduction='none')
        # norm = torch.sum(target_features ** 2, dim=-1)
        # loss_mse = torch.mean(l2_dist.sum(dim=-1) / norm)
        loss_mse = F.mse_loss(ano_features, target_features)
        loss_cos = torch.mean(1 - F.cosine_similarity(ano_features, target_features))
        loss_a = loss_mse + loss_cos
        loss += loss_a

    return loss


def calculate_log_barrier_bi_occ_loss(features, mask, target=None):
    """
    Calculate Abnormal Invariant OCC loss.
    Args:
        features: shape (N, dim)
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
    """
    A = features.norm(dim=1)
    A = torch.sqrt(A + 1) - 1
    
    Aa = A[mask == 1]
    if torch.sum(mask == 1) != 0:  # second stage, and exist anomalies
        r_max = min(0.9 * Aa.min().item(), 0.4)  # get the minimum abnormal radii as r_max
        r_min = r_max * 0.99  
    else:  # first stage, or no anomalies in second stage
        r_max = 0.4
        r_min = 0.99 * 0.4
    
    loss, loss_n, loss_a = 0, 0, 0
    if torch.sum(mask == 0) != 0:
        An = A[mask == 0]
        An_larger = An[An > r_max]  # larger than r_max
        An_lower = An[An < r_min]  # lower than r_min
        if An_larger.shape[0] != 0:
            weights = torch.exp(An_larger - r_max).detach()
            loss_larger = torch.mean(-log_sigmoid(-(An_larger - r_max)) * weights)
        else:
            loss_larger = 0
        if An_lower.shape[0] != 0:
            weights = torch.exp(r_min - An_lower).detach()
            loss_lower = torch.mean(-log_sigmoid(-(r_min - An_lower)) * weights)
        else:
            loss_lower = 0
        
        # another implementation
        # loss_larger = torch.mean(log_sigmoid(-(An - r_max)))  # cooresponding to r_max, pull into r_max
        # loss_lower = torch.mean(log_sigmoid(-(r_min - An)))  # cooresponding to r_min, pull into r_min
        
        loss_n = loss_larger + loss_lower
        loss += loss_n

    # for anomalies, we keep the mapped features as the original features
    if torch.sum(mask == 1) != 0 and target is not None:
        ano_features = features[mask == 1]
        target_features = target[mask == 1]
        loss_mse = F.mse_loss(ano_features, target_features)
        loss_cos = torch.mean(1 - F.cosine_similarity(ano_features, target_features))
        loss_inv = loss_mse + loss_cos  # anomaly invariant loss
        
        boundary = r_max + 0.1
        # using log barrier loss to push ano features out the boundary
        Aa_lower = Aa[Aa < boundary]  # lower than r_min
        if Aa_lower.shape[0] != 0:
            weights = torch.exp(boundary - Aa_lower).detach()
            loss_lower = torch.mean(-log_sigmoid(-(boundary - Aa_lower)) * weights)
        else:
            loss_lower = 0
        loss_a = loss_inv + loss_lower
        loss += loss_a

    return loss, loss_n.item() if torch.is_tensor(loss_n) else 0, loss_a.item() if torch.is_tensor(loss_a) else 0


def calculate_log_barrier_bg_spp_loss(logps, mask, boundaries):
    """
    Calculate boudary guided semi-push-pull contrastive loss.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
    """
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[mask == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    if normal_logps_inter.shape[0] != 0:
        loss_n = torch.mean(-log_sigmoid(-(b_n - normal_logps_inter)))
    else:
        loss_n = 0

    b_a = boundaries[1]
    anomaly_logps = logps[mask == 1]    
    anomaly_logps_inter = anomaly_logps[anomaly_logps >= b_a]
    if anomaly_logps_inter.shape[0] != 0:
        loss_a = torch.mean(-log_sigmoid(-(anomaly_logps_inter - b_a)))
    else:
        loss_a = 0

    return loss_n, loss_a


def get_flow_loss(C, z, m, logdet_J):
    """
    This is a combined loss of ``get_focal_loss`` and ``get_logp_loss``
    """
    logp_n = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J  # to normal center
    logp_n = logp_n / C
    logp_a = C * _GCONST_ - 0.5*torch.sum((z-1)**2, 1) + logdet_J  # to abnormal center
    logp_a = logp_a / C
    
    if (m == 0).sum() != 0:  # with normal
        loss_n = (-logp_n[m == 0]).mean()
    else:
        loss_n = 0
    if m.sum() != 0:  # with anomalies
        loss_a = (-logp_a[m == 1]).mean()
    else:
        loss_a = 0
    loss_logp = loss_n + loss_a
    
    if m.sum() != 0:  # with anomalies
        # pn = torch.exp(logp_n)
        # pa = torch.exp(logp_a)
        # sn = pn / (pn + pa)
        # sa = pa / (pn + pa)
        # s = torch.stack([sn, sa], dim=-1)  # (N, 2)
        
        logits = torch.stack([logp_n, logp_a], dim=-1)  # (N, 2)
        s = torch.softmax(logits, dim=-1)
        loss_func = FocalLoss()
        loss_focal = loss_func(s, m.unsqueeze(-1))
    else:
        loss_focal = 0
    
    loss = loss_logp + loss_focal
    return loss


def get_flow_loss_with_boundary(C, z, m, logdet_J, boundaries):
    """
    This is a combined loss of ``get_focal_loss`` and ``get_logp_loss``
    """
    logp_n = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J  # to normal center
    logp_n = logp_n / C
    logp_a = C * _GCONST_ - 0.5*torch.sum((z-1)**2, 1) + logdet_J  # to abnormal center
    logp_a = logp_a / C
    
    if (m == 0).sum() != 0:  # with normal
        loss_n = (-logp_n[m == 0]).mean()
    else:
        loss_n = 0
    if m.sum() != 0:  # with anomalies
        loss_a = (-logp_a[m == 1]).mean()
    else:
        loss_a = 0
    loss_logp = loss_n + loss_a
    
    if m.sum() != 0:  # with anomalies
        # pn = torch.exp(logp_n)
        # pa = torch.exp(logp_a)
        # sn = pn / (pn + pa)
        # sa = pa / (pn + pa)
        # s = torch.stack([sn, sa], dim=-1)  # (N, 2)
        
        logits = torch.stack([logp_n, logp_a], dim=-1)  # (N, 2)
        s = torch.softmax(logits, dim=-1)
        loss_func = FocalLoss()
        loss_focal = loss_func(s, m.unsqueeze(-1))
    else:
        loss_focal = 0
    
    # boundary guided contrast loss
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logp_n[m == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    if normal_logps_inter.shape[0] != 0:
        loss_n_con = torch.mean(-log_sigmoid(-(b_n - normal_logps_inter)))
    else:
        loss_n_con = 0

    b_a = boundaries[1]
    anomaly_logps = logp_n[m == 1]    
    anomaly_logps_inter = anomaly_logps[anomaly_logps >= b_a]
    if anomaly_logps_inter.shape[0] != 0:
        loss_a_con = torch.mean(-log_sigmoid(-(anomaly_logps_inter - b_a)))
    else:
        loss_a_con = 0
    
    loss = loss_logp + loss_focal + loss_n_con + loss_a_con
    return loss