import os
import sys
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from albumentations import Compose, PadIfNeeded, LongestMaxSize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isegm.model.is_trimap_plaintvit_model_noposembed import NoPosEmbedTrimapPlainVitModel
from isegm.data.datasets.adobe_alpha import AdobeAlphamatteDataset
from isegm.model.alpha_metrics import AlphaMAE, AlphaMSE, AlphaPSNR, AlphaGradientError

#from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference


def build_model(device):
    backbone_params = dict(
        img_size=(448, 448),
        patch_size=(14, 14),
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim=1280,
        out_dims=[240, 480, 960, 1920],
    )

    head_params = dict(
        in_channels=[240, 480, 960, 1920],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=torch.nn.BCEWithLogitsLoss(),
        align_corners=False,
        upsample='x4',
        channels=64,
    )

    model = NoPosEmbedTrimapPlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=False,
    )

    model.to(device)
    return model

def evaluate_model(model, dataset, dataset_name, device, log_file, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mae = AlphaMAE()
    mse = AlphaMSE()
    psnr = AlphaPSNR()
    grad = AlphaGradientError()

    model.eval()
    with torch.no_grad():
        for sample in tqdm(loader, desc=f"\nEvaluating {dataset_name}"):
            images = sample['images'].to(device)
            seg_masks = sample['seg_mask'].to(device)
            alpha_gt = sample['instances'].to(device)

            output = model(images, seg_masks)
            alpha_pred = output['instances']
            alpha_pred = F.interpolate(alpha_pred, size=alpha_gt.shape[-2:], mode='bilinear', align_corners=False)
            alpha_pred = torch.sigmoid(alpha_pred).squeeze(1)  # [B, H, W]

            for b in range(alpha_pred.shape[0]):
                pred = alpha_pred[b]
                gt = alpha_gt[b]
                mae.update(pred, gt)
                mse.update(pred, gt)
                psnr.update(pred, gt)
                grad.update(pred, gt)

    mae_val = mae.get_epoch_value()
    mse_val = mse.get_epoch_value()
    psnr_val = psnr.get_epoch_value()
    grad_val = grad.get_epoch_value()

    print(f"\n=== {dataset_name} ===")
    print("Dataset Size:", len(dataset))
    print(f"MAE: {mae_val:.4f} | MSE: {mse_val:.4f} | PSNR: {psnr_val:.2f} | Grad Error: {grad_val:.4f}")

    with open(log_file, 'a') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Samples: {len(dataset)}\n")
        f.write(f"MAE\tMSE\tPSNR\tGradientError\n")
        f.write(f"{mae_val:.4f}\t{mse_val:.4f}\t{psnr_val:.2f}\t{grad_val:.4f}\n\n")

    mae.reset_epoch_stats()
    mse.reset_epoch_stats()
    psnr.reset_epoch_stats()
    grad.reset_epoch_stats()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use (cpu or cuda)')
    parser.add_argument('--batch_size', default=4, type=int, help='Evaluation batch size')
    parser.add_argument('--log_dir', default='./evaluation/eval_logs', type=str, help='Directory to save evaluation logs')
    parser.add_argument('--alpha_dataset_path', required=True, type=str, help='Path to alpha matte dataset')
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d_%H%M%S_eval_log.txt'))

    with open(log_file, 'a') as f:
        f.write(f"\nEval_weight_path: {args.weight_path}\n")

    model = build_model(device=args.device)
    checkpoint = torch.load(args.weight_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    # interpolate_pos_embed_inference(model.backbone, infer_img_size=(448, 448), device=args.device)
    model.eval()

    eval_augmentator = Compose([
        LongestMaxSize(max_size=448),
        PadIfNeeded(min_height=448, min_width=448, border_mode=0),
    ])

    eval_dataset = AdobeAlphamatteDataset(dataset_path=args.alpha_dataset_path, augmentator=eval_augmentator)
    evaluate_model(model, eval_dataset, 'Adobe-Alpha', args.device, log_file, batch_size=args.batch_size)

if __name__ == '__main__':
    main()