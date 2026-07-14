############################
#
#  Adapted for Off-Axis Holography (Self-supervised)
#  Based on GedankenNet-Phase
#
#  test_OffAxis.py: Testing script for a trained GedankenNet-OffAxis model
#
#  Usage:
#   1. Set MODEL_PATH to your trained .pth file
#   2. Set TEST_HOLOGRAM_1 và TEST_HOLOGRAM_2 tới 2 ảnh hologram thực tế
#   3. Set SAMPLE_HOLOGRAM (dùng để auto-detect mask +1, thường = TEST_HOLOGRAM_1)
#   4. Run: python test_OffAxis.py
#
############################

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.nn.functional as F
from timeit import default_timer

from utilities import *
from my_tools import auto_detect_carrier, min_max_norm


################################################################
# Configs -- THAY ĐỔI THEO DỮ LIỆU THỰC TẾ CỦA ANH
################################################################

MODEL_PATH = 'Models/OffAxis_ep=5000_m=256_w=4_M=2_ph1.0/ep_5000_final.pth'

SAMPLE_HOLOGRAM  = ''   # Đường dẫn ảnh hologram thực để auto-detect mask
TEST_HOLOGRAM_1  = ''   # Hologram góc tham chiếu 1
TEST_HOLOGRAM_2  = ''   # Hologram góc tham chiếu 2

OUTPUT_DIR = 'outputs/OffAxis'

S = 512   # Kích thước patch (phải khớp với lúc train)
params = {
    'ph': 1.0,
    'patch_size': S,
}


################################################################
# Testing
################################################################

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Auto-detect mask +1
print("[Step 1] Auto-detecting +1 order spectrum mask...")
sample_holo = plt.imread(SAMPLE_HOLOGRAM)
if sample_holo.ndim == 3:
    sample_holo = sample_holo[:, :, 0]
mask_plus1, kx0, ky0 = auto_detect_carrier(sample_holo, n_steps=500, dc_mask_ratio=0.15)
print(f"  -> +1 order at kx={kx0:.1f}, ky={ky0:.1f}")
plt.imsave(os.path.join(OUTPUT_DIR, 'mask_plus1.png'), mask_plus1.astype(float), cmap='gray')

# 2. Đọc 2 ảnh hologram thực tế
def load_hologram(path, S):
    img = plt.imread(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    # Center crop về S x S
    h, w = img.shape
    ch, cw = h // 2, w // 2
    img = img[ch - S // 2:ch + S // 2, cw - S // 2:cw + S // 2]
    img = img.astype(np.float32)
    img /= img.mean()  # Chuẩn hóa
    return img

print("[Step 2] Loading test holograms...")
h1 = load_hologram(TEST_HOLOGRAM_1, S)
h2 = load_hologram(TEST_HOLOGRAM_2, S)

# Stack thành tensor [1, 2, H, W] cho mạng
xx = torch.Tensor(np.stack([h1, h2], axis=0)).unsqueeze(0).to(device)
print(f"  -> Input tensor shape: {xx.shape}")

# 3. Load model & Inference
print("[Step 3] Loading model and running inference...")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

t1 = default_timer()
with torch.no_grad():
    im, _ = model(xx)
t2 = default_timer()
print(f"  -> Inference time: {(t2 - t1) * 1000:.1f} ms")

# im: [1, 1, H, W] - predicted phase
im_np = im.cpu().numpy().squeeze()  # [H, W]

# 4. Chuẩn hóa và lưu kết quả
im_ph = im_np - im_np.mean()
im_ph_norm = min_max_norm(im_ph, vmin=np.percentile(im_ph, 0.5), vmax=np.percentile(im_ph, 99.5))

plt.imsave(os.path.join(OUTPUT_DIR, 'input_h1.png'), min_max_norm(h1), cmap='gray')
plt.imsave(os.path.join(OUTPUT_DIR, 'input_h2.png'), min_max_norm(h2), cmap='gray')
plt.imsave(os.path.join(OUTPUT_DIR, 'output_phase.png'), im_ph_norm, cmap='viridis')

# Lưu dạng .mat để xử lý tiếp bằng MATLAB
sio.savemat(os.path.join(OUTPUT_DIR, 'result.mat'), {
    'input_h1': h1,
    'input_h2': h2,
    'output_phase': im_np,
    'mask_plus1': mask_plus1.astype(np.uint8),
    'kx0': kx0,
    'ky0': ky0,
})

print(f"[Done] Results saved to {OUTPUT_DIR}/")
