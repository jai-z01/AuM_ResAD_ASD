import os
import warnings
import argparse
from tqdm import tqdm
import numpy as np
import torch
import timm
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import gc

from ResAD.train import train
from ResAD.validate import validate
from ResAD.datasets.mvtec import MVTEC, MVTECANO
from ResAD.datasets.visa import VISA, VISAANO
from ResAD.datasets.btad import BTAD
from ResAD.datasets.mvtec_3d import MVTEC3D
from ResAD.datasets.mpdd import MPDD
from ResAD.datasets.mvtec_loco import MVTECLOCO
from ResAD.datasets.brats import BRATS
from ResAD.datasets.dcase import DCASE2024

from ResAD.models.fc_flow import load_flow_model
from ResAD.models.modules import MultiScaleConv
from ResAD.models.vq import MultiScaleVQ
from ResAD.utils import init_seeds, get_residual_features, get_mc_matched_ref_features, get_mc_reference_features
from ResAD.utils import BoundaryAverager
from ResAD.losses.loss import calculate_log_barrier_bi_occ_loss
from ResAD.classes import VISA_TO_MVTEC, MVTEC_TO_VISA, MVTEC_TO_BTAD, MVTEC_TO_MVTEC3D
from ResAD.classes import MVTEC_TO_MPDD, MVTEC_TO_MVTECLOCO, MVTEC_TO_BRATS, DCASE_Dev_to_Eval
from src.models.mamba_models import AudioMamba

warnings.filterwarnings('ignore')

TOTAL_SHOT = 4  # total few-shot reference samples
#FIRST_STAGE_EPOCH = 10
SETTINGS = {'visa_to_mvtec': VISA_TO_MVTEC, 'mvtec_to_visa': MVTEC_TO_VISA,
            'mvtec_to_btad': MVTEC_TO_BTAD, 'mvtec_to_mvtec3d': MVTEC_TO_MVTEC3D,
            'mvtec_to_mpdd': MVTEC_TO_MPDD, 'mvtec_to_mvtecloco': MVTEC_TO_MVTECLOCO,
            'mvtec_to_brats': MVTEC_TO_BRATS, 'dcase': DCASE_Dev_to_Eval}

def train_model(args, encoder, vq_ops, constraintor, estimators, train_loader, boundary_ops):
    optimizer_vq = torch.optim.Adam(vq_ops.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler_vq = torch.optim.lr_scheduler.MultiStepLR(optimizer_vq, milestones=[70, 90], gamma=0.1)
    
    optimizer0 = torch.optim.Adam(constraintor.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, milestones=[70, 90], gamma=0.1)
    
    params = list(estimators[0].parameters())
    for l in range(1, args.feature_levels):
        params += list(estimators[l].parameters())
    optimizer1 = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0005)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[70, 90], gamma=0.1)

    N_batch = 8192  # Batch size for flow training
    start_epoch = 0  # Default to start from epoch 0

    # **Checkpoint Loading**
    checkpoint_path = f"{args.save_dir}/checkpoint_latest.pth"  # Path to latest checkpoint
    if os.path.exists(checkpoint_path):  # Only load if a checkpoint exists
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        vq_ops.load_state_dict(checkpoint['vq_ops'])
        constraintor.load_state_dict(checkpoint['constraintor'])
        for est, state_dict in zip(estimators, checkpoint['estimators']):
            est.load_state_dict(state_dict)

        optimizer_vq.load_state_dict(checkpoint['optimizer_vq'])
        optimizer0.load_state_dict(checkpoint['optimizer0'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        scheduler_vq.load_state_dict(checkpoint['scheduler_vq'])
        scheduler0.load_state_dict(checkpoint['scheduler0'])
        scheduler1.load_state_dict(checkpoint['scheduler1'])

        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        print(f"Resumed training from epoch {start_epoch}")
    conv1 = nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1).to(args.device)   # Stride=2 downsamples to 98x384
    conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1).to(args.device) # 49x192
    conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1).to(args.device)
    for epoch in range(start_epoch, args.epochs):
        vq_ops.train()
        constraintor.train()
        for estimator in estimators:
            estimator.train()

        train_loss_total, total_num = 0, 0
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")

        for step, batch in enumerate(train_loader):
            progress_bar.update(1)
            images, _, masks, class_names = batch
            images = images.to(args.device)
            masks = masks.to(args.device).unsqueeze(1) 

            with torch.no_grad():
                features = encoder(images).unsqueeze(1)
                feature1 = conv1(features)  # [32, 256, 98, 384]
                feature2 = conv2(feature1)      # [32, 512, 49, 192]
                feature3 = conv3(feature2)      # [32, 1024, 25, 96]

                # If necessary, interpolate to exact sizes
                feature1 = F.interpolate(feature1, size=(56, 56), mode="bilinear", align_corners=False)
                feature2 = F.interpolate(feature2, size=(28, 28), mode="bilinear", align_corners=False)
                feature3 = F.interpolate(feature3, size=(14, 14), mode="bilinear", align_corners=False)
                features = [feature1, feature2, feature3]
                #print(f"Feature1: {feature1.shape}, Feature2: {feature2.shape}, Feature3: {feature3.shape}")
            
            ref_features = get_mc_reference_features(encoder, args.train_dataset_dir, class_names, images.device, args.train_ref_shot)
            mfeatures = get_mc_matched_ref_features(features, class_names, ref_features)
            rfeatures = get_residual_features(features, mfeatures, pos_flag=True)

            lvl_masks = [F.interpolate(masks, size=(r.shape[2], r.shape[3]), mode='nearest').squeeze(1) for r in rfeatures]
            rfeatures_t = [rfeature.detach().clone() for rfeature in rfeatures]

            loss_vq = vq_ops(rfeatures, lvl_masks, train=True)
            optimizer_vq.zero_grad()
            loss_vq.backward()
            optimizer_vq.step()

            rfeatures = constraintor(*rfeatures)
            loss = sum(calculate_log_barrier_bi_occ_loss(rfeatures[l].permute(0, 2, 3, 1).reshape(-1, rfeatures[l].shape[1]), 
                                                         lvl_masks[l].reshape(-1),
                                                         rfeatures_t[l].permute(0, 2, 3, 1).reshape(-1, rfeatures_t[l].shape[1]))[0]
                        for l in range(args.feature_levels))
            optimizer0.zero_grad()
            loss.backward()
            optimizer0.step()

            train_loss_total += loss.item()
            total_num += 1

            # Train normalizing flow
            rfeatures = [rfeature.detach().clone() for rfeature in rfeatures]
            loss, num = train(args, rfeatures, estimators, optimizer1, masks, boundary_ops, epoch, N_batch=N_batch)
            train_loss_total += loss
            total_num += num

        scheduler_vq.step()
        scheduler0.step()
        scheduler1.step()
        progress_bar.close()
        print(f"Epoch[{epoch}/{args.epochs}]: train_loss: {train_loss_total / total_num}")
        if (epoch + 1) % 10 == 0:  # Save model every 10 epochs
            #save_path = f"{args.save_dir}/checkpoint_latest.pth"  # Always overwrite the latest checkpoint
            save_path = f"{args.save_dir}/continue/checkpoint_latest.pth"
            torch.save({
                'vq_ops': vq_ops.state_dict(),
                'constraintor': constraintor.state_dict(),
                'estimators': [estimator.state_dict() for estimator in estimators],
                'optimizer_vq': optimizer_vq.state_dict(),
                'optimizer0': optimizer0.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'scheduler_vq': scheduler_vq.state_dict(),
                'scheduler0': scheduler0.state_dict(),
                'scheduler1': scheduler1.state_dict(),
                'epoch': epoch
            }, save_path)
            print(f"Checkpoint saved at {save_path}")  

def test_model(args, encoder, vq_ops, constraintor, estimators):
    # Load checkpoint if it exists
    checkpoint_path = f"{args.save_dir}/checkpoint_latest.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)  # Load to CPU first
        print(f"Loading checkpoint from {checkpoint_path}")
        # Load models safely to avoid memory spikes
        vq_ops.load_state_dict(checkpoint['vq_ops'], strict=False)
        constraintor.load_state_dict(checkpoint['constraintor'], strict=False)

        for est, state_dict in zip(estimators, checkpoint['estimators']):
            est.load_state_dict(state_dict, strict=False)

        del checkpoint  # Free RAM after loading
        torch.cuda.empty_cache()  # Force garbage collection
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found. Running inference with current model weights.")

    # Set models to evaluation mode
    encoder.eval()
    vq_ops.eval()
    constraintor.eval()
    for estimator in estimators:
        estimator.eval()

    #unseen_classes = ['3DPrinter', 'AirCompressor', 'BrushlessMotor', 'HairDryer', 'HoveringDrone', 'RoboticArm', 'Scanner', 'ToothBrush', 'ToyCircuit']
    #seen_classes = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    seen_classes = ['bearing']
    gc.collect()
    torch.cuda.empty_cache()
    print("Freed up GPU memory before testing...")

    # Load test reference features
    test_ref_features = load_mc_reference_features(args.test_ref_feature_dir, seen_classes, args.device , args.num_ref_shot)

    s1_res, s2_res, s_res = [], [], []

    output_dir = "test_results"
    base_filename = f"results_{args.num_ref_shot}shot_{args.backbone}"
    counter = 1
    while os.path.exists(os.path.join(output_dir, f"{base_filename}_{counter}.txt")):
        counter += 1
    result_file = os.path.join(output_dir, f"{base_filename}_{counter}.txt")

    for class_name in seen_classes:
        #print(f"Loading test dataset from: {args.test_dataset_dir}")
        test_dataset = DCASE2024(args.test_dataset_dir, class_name=class_name, train=False, normalize='logmel',
                                 img_size=224, crp_size=224, msk_size=224, msk_crp_size=224, machine_types=seen_classes)
        #print(f"Dataset size: {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
        #print(f"Test Loader Length: {len(test_loader)}")

        # Run validation
        metrics = validate(args, encoder, vq_ops, constraintor, estimators, test_loader, test_ref_features[class_name], args.device, class_name)
        save_result(class_name, metrics, result_file)
        print(f"Class: {class_name}, AUC: {metrics['scores'][0]}, pAUC: {metrics['scores'][1]}")
        s1_res.append(metrics['scores1'])
        s2_res.append(metrics['scores2'])
        s_res.append(metrics['scores'])

    print("Final Testing Results:")
    print_and_save_results(np.array(s1_res), np.array(s2_res), np.array(s_res), result_file)

def save_result(class_name, metrics, result_file):
    img_auc1, img_pauc1, img_ap1, img_f1_score1 = metrics['scores1']
    img_auc2, img_pauc2, img_ap2, img_f1_score2 = metrics['scores2']
    img_auc, img_pauc, img_ap, img_f1_score = metrics['scores']
    results = [
        f"Class: {class_name}\n"
        f"(Logps) Avg Image AUC: {img_auc1:.3f}\t Avg Image pAUC: {img_pauc1:.3f}\t Avg Precision: {img_ap1:.3f}\t Avg F1 Score: {img_f1_score1:.3f}",
        f"(BScores) Avg Image AUC: {img_auc2:.3f}\t Avg Image pAUC: {img_pauc2:.3f}\t Avg Precision: {img_ap2:.3f}\t Avg F1 Score: {img_f1_score2:.3f}",
        f"(Merged) Avg Image AUC: {img_auc:.3f}\t Avg Image AUC: {img_pauc:.3f}\t Avg Precision: {img_ap:.3f}\t Avg F1 Score: {img_f1_score:.3f}"
    ]
    with open(result_file, "a") as file:
        for line in results:
            file.write(line + "\n")
        file.write("\n")

def print_and_save_results(s1_res, s2_res, s_res, result_file):
    img_auc1, img_pauc1, img_ap1, img_f1_score1 = np.mean(s1_res, axis=0)
    img_auc2, img_pauc2, img_ap2, img_f1_score2 = np.mean(s2_res, axis=0)
    img_auc, img_pauc, img_ap, img_f1_score = np.mean(s_res, axis=0)
    results = [
        f"Overall\n"
        f"(Logps) Avg Image AUC: {img_auc1:.3f}\t Avg Image pAUC: {img_pauc1:.3f}\t Avg Precision: {img_ap1:.3f}\t Avg F1 Score: {img_f1_score1:.3f}",
        f"(BScores) Avg Image AUC: {img_auc2:.3f}\t Avg Image pAUC: {img_pauc2:.3f}\t Avg Precision: {img_ap2:.3f}\t Avg F1 Score: {img_f1_score2:.3f}",
        f"(Merged) Avg Image AUC: {img_auc:.3f}\t Avg Image AUC: {img_pauc:.3f}\t Avg Precision: {img_ap:.3f}\t Avg F1 Score: {img_f1_score:.3f}"
    ]
    for i in results[1:]:
        print(i)
    with open(result_file, "a") as file:
        for line in results:
            file.write(line + "\n")

    print(f"Results saved to {result_file}")

def main(args):
    if args.setting not in SETTINGS.keys():
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")
    
    train_dataset = DCASE2024(args.train_dataset_dir, class_name=SETTINGS[args.setting]['seen'], train=True, normalize="logmel",
                              img_size=224, crp_size=224, msk_size=224, msk_crp_size=224, machine_types=['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    encoder = AudioMamba(pretrained=True).eval().to(args.device)
    boundary_ops = BoundaryAverager(num_levels=args.feature_levels)
    vq_ops = MultiScaleVQ(num_embeddings=args.num_embeddings, channels=(256,512,1024)).to(args.device)
    constraintor = MultiScaleConv((256,512,1024)).to(args.device)
    estimators = [load_flow_model(args, feat_dim).to(args.device) for feat_dim in (256,512,1024)]

    if args.mode == "train":
        train_model(args, encoder, vq_ops, constraintor, estimators, train_loader, boundary_ops)
    elif args.mode == "test":
        del train_dataset, train_loader
        gc.collect()
        torch.cuda.empty_cache()
        test_model(args, encoder, vq_ops, constraintor, estimators)
    else:
        raise ValueError("Invalid mode. Use --mode train or --mode test")
    
def load_mc_reference_features(root_dir: str, class_names, device: torch.device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
        layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
        layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
        
        layer1_refs = torch.from_numpy(layer1_refs).to(device)
        layer2_refs = torch.from_numpy(layer2_refs).to(device)
        layer3_refs = torch.from_numpy(layer3_refs).to(device)
        
        K1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer1_refs = layer1_refs[:K1, :]
        K2 = (layer2_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer2_refs = layer2_refs[:K2, :]
        K3 = (layer3_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer3_refs = layer3_refs[:K3, :]
        
        refs[class_name] = (layer1_refs, layer2_refs, layer3_refs)
    
    return refs
                    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="dcase")
    parser.add_argument('--train_dataset_dir', type=str, default="./ResAD/datasets/data/dev_spectrograms")
    parser.add_argument('--test_dataset_dir', type=str, default="./ResAD/datasets/data/dev_spectrograms")
    parser.add_argument('--test_ref_feature_dir', type=str, default="/mnt/e/ref_features_aum/dcase")
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="aum")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--save_dir', type=str, default="/mnt/e/resad_aum_model")
    
    # flow parameters
    parser.add_argument('--flow_arch', type=str, default='conditional_flow_model')
    parser.add_argument('--feature_levels', default=3, type=int)
    parser.add_argument('--coupling_layers', type=int, default=10)
    parser.add_argument('--clamp_alpha', type=float, default=1.9)
    parser.add_argument('--pos_embed_dim', type=int, default=256)
    parser.add_argument('--pos_beta', type=float, default=0.05)
    parser.add_argument('--margin_tau', type=float, default=0.1)
    parser.add_argument('--bgspp_lambda', type=float, default=1)
    
    parser.add_argument('--fdm_alpha', type=float, default=0.4)  # low value, more training distribution
    parser.add_argument('--num_embeddings', type=int, default=1536)  # VQ embeddings
    parser.add_argument("--train_ref_shot", type=int, default=4)
    parser.add_argument("--num_ref_shot", type=int, default=4)
    
    args = parser.parse_args()
    init_seeds(42)
    
    main(args)
