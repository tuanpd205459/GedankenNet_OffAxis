############################
#
#  Adapted for Off-Axis Holography (Self-supervised)
#  Based on GedankenNet-Phase
#
#  train_OffAxis.py: Training script for GedankenNet-OffAxis
#
#  Key changes vs train_GedankenP.py:
#   - Physics forward model: I = |O + U_ref|^2  (no FSP)
#   - (kx, ky) sampled from auto-detected +1 order mask
#   - Loss: MSE + Fourier MAE + TV on FULL hologram intensity
#
############################

# %% init
import os
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # SPECIFY YOUR GPU ID

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities import *
import glob
from skimage.metrics import structural_similarity as ssim

import operator
from functools import reduce
from functools import partial

from my_tools import (
    auto_detect_carrier,
    RealOffAxisDataset,
    batch_offaxis_interference,
    count_params,
    min_max_norm,
    pad_and_crop
)
from networks.fno import FNO2d
import np_transforms
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
# TV Loss
################################################################

def tv_loss(inputs):
    """Total Variation Loss - làm mượt ảnh pha."""
    n, c, h, w = inputs.shape
    grad_x = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
    grad_y = inputs[:, :, :, 1:] - inputs[:, :, :, :-1]
    tv = (grad_x.abs().sum() + grad_y.abs().sum()) / (n * c * h * w)
    return tv


################################################################
# Configs  -- THAY ĐỔI CÁC THÔNG SỐ NÀY THEO DỮ LIỆU CỦA ANH
################################################################

DATA_RAW_PATH = 'data_raw'       # Đường dẫn tới thư mục ảnh thực tế (e.g. s1 (1).bmp)
SAMPLE_HOLOGRAM = 'data_raw/s1 (1).bmp'  # Đường dẫn tới 1 ảnh hologram thực tế để auto-detect mask

M = 2              # Số hologram per sample (ví dụ random 2 góc tham chiếu)
modes = 256        # Số Fourier modes của FNO
width = 4          # Độ rộng kênh của FNO
batch_size = 1
batch_per_ep = 250
epochs = 5000
learning_rate = 0.0001

S = 512            # Kích thước patch

# Thông số vật lý (đơn vị: um)
params = {
    'wavelength': 0.530,
    'pixel_size': 0.3733,
    'patch_size': S,
    'ref_ind': 1.00,
    'ph': 1.0,           # Dải pha: [0, pi]
    'noise_level': 0.005 * np.sqrt(2),  # 40 dB
}


################################################################
# Main
################################################################

def main():

    # ── 1. Auto-detect vùng phổ +1 từ hologram thực tế ──────────────────
    print("=" * 60)
    print("[Step 1] Auto-detecting +1 order spectrum mask...")
    sample_holo = plt.imread(SAMPLE_HOLOGRAM)
    if sample_holo.ndim == 3:
        sample_holo = sample_holo[:, :, 0]  # Lấy kênh Gray
    
    sample_holo = pad_and_crop(sample_holo, S)

    mask_plus1, kx0, ky0 = auto_detect_carrier(sample_holo, dc_mask_ratio=0.15)
    print(f"  -> Found +1 order at approx. kx={kx0:.1f}, ky={ky0:.1f} (pixel units from center)")
    print(f"  -> Mask +1 area: {mask_plus1.sum()} pixels")

    # Lưu mask để kiểm tra (tuỳ chọn)
    plt.imsave('mask_plus1_detected.png', mask_plus1.astype(float), cmap='gray')
    print("  -> Saved mask to 'mask_plus1_detected.png'")
    print("=" * 60)

    # ── 2. Load dataset ──────────────────────────────────────────────────
    print("[Step 2] Scanning real data and splitting Train/Valid...")
    sample_dict = {}
    for fname in os.listdir(DATA_RAW_PATH):
        if fname.endswith('.bmp'):
            prefix = fname.rsplit(' (', 1)[0]
            if prefix not in sample_dict:
                sample_dict[prefix] = []
            sample_dict[prefix].append(os.path.join(DATA_RAW_PATH, fname))
            
    valid_samples = [k for k, v in sample_dict.items() if len(v) >= M]
    random.shuffle(valid_samples)
    
    split_idx = int(len(valid_samples) * 0.8)
    train_samples = valid_samples[:split_idx]
    valid_samples_list = valid_samples[split_idx:]
    
    train_dataset = RealOffAxisDataset(train_samples, sample_dict, mask_plus1, M, S)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    valid_dataset = RealOffAxisDataset(valid_samples_list, sample_dict, mask_plus1, M, S)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"  -> Train: {len(train_samples)} samples | Valid: {len(valid_samples_list)} samples")

    # ── 3. Khởi tạo Model ────────────────────────────────────────────────
    # FNO2d(modes, width, in_dim=M, out_dim=1)
    # in_dim = M: nhận M kênh hologram đầu vào
    # out_dim = 1: xuất ra 1 kênh pha
    model = FNO2d(modes, width, M, 1).cuda()
    print(f"[Step 3] Model parameters: {count_params(model):,}")

    path = 'OffAxis_ep=%d_m=%d_w=%d_M=%d_ph%.1f' % (epochs, modes, width, M, params['ph'])
    path_model = 'Models/' + path
    os.makedirs(path_model, exist_ok=True)
    writer = SummaryWriter(os.path.join("runs", path))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

    maeloss = nn.L1Loss(reduction='mean')
    mseloss = nn.MSELoss()

    # Fourier domain Hann window để ưu tiên tần số thấp
    hann_window = torch.outer(torch.hann_window(S), torch.hann_window(S))
    hann_window = torch.fft.ifftshift(hann_window).unsqueeze(0).unsqueeze(0).cuda()

    # ── 4. Resume từ checkpoint nếu có ──────────────────────────────────
    start_ep = -1
    min_valid_mse = 1e4
    ckpt_path = os.path.join(path_model, "checkpoint.pth")
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        start_ep = checkpoint['epoch']
        print(f"[Resume] From checkpoint: epoch {start_ep + 1}")
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])

    # ── 5. Training Loop ─────────────────────────────────────────────────
    print("[Step 4] Starting training...")
    for ep in range(start_ep + 1, epochs):
        print(f"  Epoch {ep} | lr={optimizer.param_groups[0]['lr']:.6f}")

        # ── Train ──
        model.train()
        t1 = default_timer()
        train_loss_sum = 0

        for i, (xx, yy, kk) in enumerate(train_loader):
            if i >= batch_per_ep:
                break

            # xx: [N, M, H, W]  hologram đầu vào (mô phỏng hoặc thực tế)
            # yy: [N, 2, H, W]  ground truth trường phức (chỉ để monitor)
            # kk: [N, M, 2]     các cặp (kx, ky) dùng để tạo hologram

            xx = xx.to(device)
            yy = yy.to(device)
            kk = kk.to(device)

            # ── Forward qua FNO ──
            # model output: im [N, 1, H, W] - predicted phase
            im, _ = model(xx)

            # ── Physics Forward Model: I_sim = |O + U_ref|^2 ──
            # batch_offaxis_interference nhận pha [N,1,H,W] và k_batch [N,M,2]
            im_x = batch_offaxis_interference(im, kk, params)  # [N, M, H, W]

            # ── Tính Loss ──
            loss = 0
            for m_idx in range(M):
                sim_m = im_x[:, m_idx:m_idx+1, :, :]   # [N, 1, H, W]
                raw_m = xx[:, m_idx:m_idx+1, :, :]      # [N, 1, H, W]

                # 1. Fourier MAE (ưu tiên cấu trúc toàn cục)
                loss += maeloss(
                    torch.fft.fft2(sim_m) * hann_window,
                    torch.fft.fft2(raw_m) * hann_window
                ) * 0.1

                # 2. Spatial MAE (khớp chi tiết từng pixel)
                loss += maeloss(sim_m, raw_m) * 10.0

            # 3. TV Loss trên pha dự đoán (làm mượt)
            loss += tv_loss(im) * 5.0

            train_loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        # ── Validation ──
        model.eval()
        valid_mse_sum = 0
        xx_list, yy_list, im_list = [], [], []

        with torch.no_grad():
            for i, (xx, yy, kk) in enumerate(valid_loader):
                if i >= 5:
                    break
                xx = xx.to(device)
                yy = yy.to(device)

                im, _ = model(xx)

                # Mô phỏng hologram từ pha dự đoán để đánh giá (Evaluation)
                im_x = batch_offaxis_interference(im, kk, params)
                
                # Tính MSE giữa hologram mô phỏng và hologram thực tế
                valid_mse_sum += mseloss(im_x, xx).item()

                xx_list.append(xx[:, 0:1, ...].cpu().numpy())   # kênh hologram đầu tiên
                im_list.append((im - im.mean()).cpu().numpy())

        valid_mse = valid_mse_sum / max(i + 1, 1)

        # TensorBoard logging
        xx_np = np.vstack(xx_list).reshape((-1,) + xx_list[0].shape[1:])
        im_np = np.vstack(im_list).reshape((-1,) + im_list[0].shape[1:])

        writer.add_images('output_ph',
            np.clip((im_np - im_np.min()) / (im_np.max() - im_np.min() + 1e-8), 0, 1),
            ep, dataformats='NCHW')
        writer.add_scalar('train_loss', train_loss_sum / batch_per_ep, ep)
        writer.add_scalar('valid_mse', valid_mse, ep)

        # Lưu model tốt nhất
        if valid_mse < min_valid_mse and ep > 50:
            torch.save(model, os.path.join(path_model, f"ep_{ep}.pth"))
            min_valid_mse = valid_mse

        # Checkpoint mỗi 50 epoch
        if (ep + 1) % 50 == 0:
            torch.save({
                'epoch': ep,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'np_rand_state': np.random.get_state(),
                'scheduler': scheduler.state_dict(),
                'torch_rand_state': torch.get_rng_state(),
            }, ckpt_path)

        scheduler.step()
        t2 = default_timer()
        print(f"    train_loss={train_loss_sum / batch_per_ep:.4f} | valid_mse={valid_mse:.4f} | time={t2 - t1:.1f}s")

    # ── 6. Save final model ──────────────────────────────────────────────
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
    torch.save(model, os.path.join(path_model, f"ep_{epochs}_final.pth"))
    print("[Done] Training complete!")


if __name__ == '__main__':
    main()
