import os
import warnings
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ResAD.train import train
from ResAD.validate import validate
from ResAD.datasets.dcase import DCASE2024  # Custom dataset for spectrograms

from src.models.mamba_models import AudioMamba  # Import Audio-Mamba model
from ResAD.models.fc_flow import load_flow_model
from ResAD.models.modules import MultiScaleConv
from ResAD.models.vq import MultiScaleVQ
from ResAD.utils import init_seeds, get_residual_features, get_mc_matched_ref_features, get_mc_reference_features
from ResAD.utils import BoundaryAverager
from ResAD.losses.loss import calculate_log_barrier_bi_occ_loss

warnings.filterwarnings('ignore')

TOTAL_SHOT = 4  # total few-shot reference samples
FIRST_STAGE_EPOCH = 10

def main(args):
    # Load Spectrogram Dataset
    train_dataset = DCASE2024(root=args.train_dataset_dir, machine_types=['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'], train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Initialize Audio-Mamba as Feature Extractor
    encoder = AudioMamba(pretrained=True).to(args.device)
    encoder.eval()  # Set to evaluation mode

    # Get Feature Dimensions from Audio-Mamba
    sample_input = torch.randn(1, 1, 128, 128).to(args.device)  # Dummy spectrogram input
    with torch.no_grad():
        sample_features = encoder(sample_input)
    print("Sample features type:", type(sample_features))
    print("Model Output Shape:", sample_features.shape)
    #print("Sample features:", sample_features)
    if isinstance(sample_features, torch.Tensor):
        sample_features = [sample_features]
    
    feat_dims = [f.shape[1] for f in sample_features]*3
    print("feat_dims:", feat_dims)

    #feat_dims = [f.shape[1] for f in sample_features]  # Extract channel dimensions

    boundary_ops = BoundaryAverager(num_levels=args.feature_levels)
    vq_ops = MultiScaleVQ(num_embeddings=args.num_embeddings, channels=feat_dims).to(args.device)
    optimizer_vq = torch.optim.Adam(vq_ops.parameters(), lr=args.lr, weight_decay=0.0005)

    constraintor = MultiScaleConv(feat_dims).to(args.device)
    optimizer0 = torch.optim.Adam(constraintor.parameters(), lr=args.lr, weight_decay=0.0005)

    estimators = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    estimators = [decoder.to(args.device) for decoder in estimators]
    optimizer1 = torch.optim.Adam([p for estimator in estimators for p in estimator.parameters()], lr=args.lr, weight_decay=0.0005)

    # Training Loop
    for epoch in range(args.epochs):
        encoder.eval()
        vq_ops.train()
        constraintor.train()
        for estimator in estimators:
            estimator.train()

        train_loss_total, total_num = 0, 0
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")

        for step, batch in enumerate(train_loader):
            progress_bar.update(1)
            #print(f"Batch content: {batch}")
            spectrograms, masks, _ = batch
            spectrograms, masks = spectrograms.to(args.device), masks.to(args.device)

            with torch.no_grad():
                features = encoder(spectrograms)  # Extract features using Audio-Mamba
            print(f"🔹 Debug: Extracted feature shape -> {features.shape}")
            class_names = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
            ref_features = get_mc_reference_features(encoder, args.train_dataset_dir, class_names, spectrograms.device, args.train_ref_shot)
            reshaped_features = []
            for f in features:
                if len(f.shape) == 1:  # If it's (C,)
                    f = f.view(1, f.shape[0], 1, 1)  # Convert to (B=1, C, H=1, W=1)
                reshaped_features.append(f)
            if ref_features is not None:  # Ensure it's defined
                mfeatures = get_mc_matched_ref_features(reshaped_features, class_names, ref_features)
            else:
                raise ValueError("Error: ref_features is None. Check `get_mc_reference_features()`.")

            rfeatures = get_residual_features(features, mfeatures, pos_flag=True)

            lvl_masks = [F.interpolate(masks, size=(f.shape[2], f.shape[3]), mode='nearest').squeeze(1) for f in rfeatures]
            loss_vq = vq_ops(rfeatures, lvl_masks, train=True)

            optimizer_vq.zero_grad()
            loss_vq.backward()
            optimizer_vq.step()

            rfeatures = constraintor(*rfeatures)

            loss = 0
            for l in range(args.feature_levels):
                e, t = rfeatures[l], rfeatures[l].detach().clone()
                bs, dim, h, w = e.size()
                e, t = e.permute(0, 2, 3, 1).reshape(-1, dim), t.permute(0, 2, 3, 1).reshape(-1, dim)
                m = lvl_masks[l].reshape(-1)
                loss_i, _, _ = calculate_log_barrier_bi_occ_loss(e, m, t)
                loss += loss_i

            optimizer0.zero_grad()
            loss.backward()
            optimizer0.step()

            rfeatures = [rfeature.detach().clone() for rfeature in rfeatures]
            train_loss, num = train(args, rfeatures, estimators, optimizer1, masks, boundary_ops, epoch)
            train_loss_total += train_loss
            total_num += num

        progress_bar.close()
        print(f"Epoch[{epoch}/{args.epochs}]: train_loss: {train_loss_total / total_num}")

        if (epoch + 1) % args.eval_freq == 0:
            validate(args, encoder, vq_ops, constraintor, estimators, args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_dir', type=str, required=True)
    parser.add_argument('--test_dataset_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--feature_levels', type=int, default=3)
    parser.add_argument('--num_embeddings', type=int, default=128)
    parser.add_argument('--train_ref_shot', type=int, default=4)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/')
    parser.add_argument('--flow_arch', type=str, default='flow_model', help='Architecture of the flow model')
    parser.add_argument('--coupling_layers', type=int, default=8, help='Number of coupling layers in the flow model')
    parser.add_argument('--clamp_alpha', type=float, default=2.0, help='Affine clamping parameter for normalizing flow')


    
    args = parser.parse_args()
    main(args)
