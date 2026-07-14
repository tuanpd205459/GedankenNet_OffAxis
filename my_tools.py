############################
#
#  Adapted for Off-Axis Holography (Self-supervised)
#  Based on: "Self-supervised learning of hologram reconstruction using physics consistency"
#  Original Authors: Luzhe Huang, Hanlong Chen, Tairan Liu and Aydogan Ozcan
#
#  my_tools.py: essential functions & datasets for GedankenNet-OffAxis
#
#  Key changes vs GedankenNet_Phase:
#   - Replaced FSP (Free Space Propagation) with off-axis interference model: I = |O + Ref|^2
#   - Added auto_detect_carrier(): finds +1 order spectrum region using adaptive
#     threshold expansion until exactly 3 connected regions appear (0, +1, -1)
#   - Training data uses random (kx, ky) sampled STRICTLY inside the detected +1 mask
#
############################

import random
from typing import List
import torch
import numpy as np
import scipy
import scipy.io
import scipy.signal
import scipy.ndimage
import skimage.measure
from skimage.measure import label
import PIL
import torch.nn as nn
import operator
from functools import reduce
from functools import partial


# ─────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────

def min_max_norm(img, vmin=None, vmax=None):
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()
    img = np.clip(img, vmin, vmax)
    return (img - vmin) / (vmax - vmin)


def comp_field_norm(comp_field):
    """Normalize a complex field [N, C=2, H, W] or [C, H, W]."""
    if isinstance(comp_field, np.ndarray):
        if comp_field.ndim == 3:
            comp_field = comp_field[0, ...] + 1j * comp_field[1, ...]
            comp_field /= (np.mean(np.abs(comp_field), axis=(-2, -1), keepdims=True) *
                           np.exp(1j * np.mean(np.angle(comp_field), axis=(-2, -1), keepdims=True)))
            return np.stack((np.real(comp_field), np.imag(comp_field)), axis=0)
        elif comp_field.ndim == 4:
            comp_field = comp_field[:, 0, ...] + 1j * comp_field[:, 1, ...]
            comp_field /= (np.mean(np.abs(comp_field), axis=(-2, -1), keepdims=True) *
                           np.exp(1j * np.mean(np.angle(comp_field), axis=(-2, -1), keepdims=True)))
            return np.stack((np.real(comp_field), np.imag(comp_field)), axis=1)
    elif isinstance(comp_field, torch.Tensor):
        comp_field = comp_field[:, 0, ...] + 1j * comp_field[:, 1, ...]
        comp_field /= (torch.mean(torch.abs(comp_field), dim=(-2, -1), keepdim=True) *
                       torch.exp(1j * torch.mean(torch.angle(comp_field), dim=(-2, -1), keepdim=True)))
        return torch.stack([torch.real(comp_field), torch.imag(comp_field)], dim=1)


# ─────────────────────────────────────────────
# OFF-AXIS SPECTRUM MASK AUTO-DETECTION
# Thuật toán: giảm ngưỡng dần đến khi có đúng 3 vùng liên thông
# Vùng +1 được xác định là vùng PHẢI của trung tâm
# ─────────────────────────────────────────────

def auto_detect_carrier(hologram, n_steps=500, dc_mask_ratio=0.15):
    """
    Tự động phát hiện vùng phổ bậc +1 bằng cách giảm dần ngưỡng cường độ
    cho đến khi xuất hiện đúng 3 vùng liên thông (bậc 0, +1, -1).
    Vùng bậc +1 được chọn là vùng nằm bên PHẢI tâm ảnh.

    Args:
        hologram   : ảnh hologram thực tế, numpy 2D float [H, W]
        n_steps    : số bước giảm ngưỡng
        dc_mask_ratio: tỷ lệ vùng DC bị che (0.15 = 15% cạnh ngắn nhất)

    Returns:
        mask_plus1  : numpy 2D bool [H, W] - mask vùng phổ +1
        kx0, ky0    : tọa độ trung tâm (cột, hàng) của đỉnh phổ +1
                      trong hệ tọa độ đã fftshift (tâm ảnh = (H/2, W/2))
    """
    H, W = hologram.shape[:2]
    cx, cy = W // 2, H // 2

    # 1. Tính phổ biên độ (đã fftshift về trung tâm)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(hologram.astype(np.float64))))

    # 2. Tạo DC mask để che vùng bậc 0 ở trung tâm
    dc_r = int(min(H, W) * dc_mask_ratio)
    yy, xx = np.ogrid[:H, :W]
    dc_mask = (xx - cx) ** 2 + (yy - cy) ** 2 > dc_r ** 2
    spectrum_no_dc = spectrum * dc_mask

    peak_val = spectrum_no_dc.max()

    # 3. Giảm dần ngưỡng cho đến khi xuất hiện đúng 3 vùng liên thông
    best_mask = None
    best_n_regions = 0
    for step in range(n_steps):
        # Ngưỡng từ peak_val xuống tới ~5% peak
        threshold = peak_val * (1.0 - step / n_steps * 0.95)
        binary = (spectrum_no_dc >= threshold).astype(np.uint8)
        labeled, n_regions = label(binary, return_num=True)
        if n_regions == 3:
            best_mask = binary
            best_n_regions = 3
            break
        elif n_regions < 3:
            # Lưu lại kết quả gần nhất có ít hơn 3 vùng để dự phòng
            best_mask = binary
            best_n_regions = n_regions

    # Nếu không tìm thấy đúng 3 vùng, dùng kết quả tốt nhất
    if best_n_regions != 3:
        print(f"[WARN] auto_detect_carrier: found {best_n_regions} regions, expected 3. Using best result.")

    labeled, _ = label(best_mask, return_num=True)

    # 4. Chọn vùng +1: vùng có tọa độ trung tâm nằm bên PHẢI của tâm ảnh (cx)
    regions = []
    for region_id in range(1, _.max() + 1 if hasattr(_, 'max') else 4):
        pass
    # Lấy tất cả vùng và chọn vùng bên phải
    from skimage.measure import regionprops
    props = regionprops(labeled)
    right_regions = [r for r in props if r.centroid[1] > cx]  # centroid[1] là cột (x)

    if len(right_regions) == 0:
        raise ValueError("Không tìm thấy vùng phổ +1 ở phía phải! Kiểm tra lại hologram đầu vào.")

    # Chọn vùng phải có biên độ lớn nhất
    plus1_region = max(right_regions, key=lambda r: spectrum[labeled == r.label].sum())
    mask_plus1 = (labeled == plus1_region.label).astype(bool)

    # 5. Tọa độ tâm của vùng +1 (hàng, cột)
    # Trả về kx = cột - cx (lệch so với tâm), ky = hàng - cy
    ky0 = plus1_region.centroid[0] - cy   # lệch theo chiều hàng
    kx0 = plus1_region.centroid[1] - cx   # lệch theo chiều cột

    return mask_plus1, kx0, ky0


def sample_k_from_mask(mask_plus1, cx, cy, n_samples=2):
    """
    Lấy ngẫu nhiên n_samples tọa độ (kx, ky) nằm trong vùng mask +1.
    kx, ky là tọa độ chuẩn hóa tính từ tâm ảnh (đơn vị: pixel lệch).

    Args:
        mask_plus1 : 2D bool mask [H, W]
        cx, cy     : tọa độ tâm ảnh (W//2, H//2)
        n_samples  : số lượng cặp (kx, ky) cần lấy

    Returns:
        k_list : list of (kx, ky) tuples
    """
    # Lấy tất cả pixel thuộc vùng mask
    rows, cols = np.where(mask_plus1)
    if len(rows) == 0:
        raise ValueError("Mask +1 rỗng!")

    k_list = []
    for _ in range(n_samples):
        idx = np.random.randint(len(rows))
        ky = float(rows[idx] - cy)
        kx = float(cols[idx] - cx)
        k_list.append((kx, ky))
    return k_list


# ─────────────────────────────────────────────
# MÔ PHỎNG HOLOGRAM OFF-AXIS
# I = |O + U_ref|^2
# U_ref = exp(i * 2pi * (kx * x + ky * y) / N)
# ─────────────────────────────────────────────

def simulate_offaxis_hologram(comp_obj, kx, ky):
    """
    Mô phỏng hologram off-axis theo công thức: I = |O + U_ref|^2
    
    Args:
        comp_obj : numpy 2D complex [H, W] - trường phức của vật thể
        kx, ky   : tọa độ tần số sóng mang (pixel units, tính từ tâm phổ)

    Returns:
        intensity : numpy 2D float [H, W] - cường độ hologram mô phỏng
    """
    H, W = comp_obj.shape[:2]
    # Tạo lưới tọa độ không gian [0, H) x [0, W)
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    # Sóng tham chiếu: U_ref = exp(i * 2pi * (kx*x/W + ky*y/H))
    U_ref = np.exp(1j * 2 * np.pi * (kx * x_grid / W + ky * y_grid / H))
    # Giao thoa: I = |O + U_ref|^2
    interference = comp_obj + U_ref
    intensity = np.abs(interference) ** 2
    return intensity.astype(np.float32)


def simulate_offaxis_hologram_torch(comp_obj, kx, ky):
    """
    Phiên bản PyTorch của simulate_offaxis_hologram - dùng trong training loop.
    comp_obj : torch complex [H, W]
    kx, ky   : torch scalar (learnable parameters)
    Returns  : torch float [H, W]
    """
    H, W = comp_obj.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=comp_obj.device),
        torch.arange(W, dtype=torch.float32, device=comp_obj.device),
        indexing='ij'
    )
    U_ref = torch.exp(1j * 2 * np.pi * (kx * x_grid / W + ky * y_grid / H))
    interference = comp_obj + U_ref
    intensity = torch.abs(interference) ** 2
    return intensity


def batch_offaxis_interference(batch_phase, k_batch, params):
    """
    Mô phỏng batch hologram off-axis từ dự đoán pha của mạng.
    batch_phase : torch [N, 1, H, W] - pha dự đoán
    k_batch     : torch [N, M, 2] - M cặp (kx, ky) cho mỗi mẫu trong batch
                  (M = số hologram per sample, thường = 2)
    Returns     : torch [N, M, H, W] - hologram mô phỏng
    """
    N, _, H, W = batch_phase.shape
    M = k_batch.shape[1]
    sim_holograms = []

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=batch_phase.device),
        torch.arange(W, dtype=torch.float32, device=batch_phase.device),
        indexing='ij'
    )

    for n in range(N):
        # Trường phức vật thể (Phase-only, Amplitude = 1)
        comp_obj = torch.exp(1j * params['ph'] * np.pi * batch_phase[n, 0, ...])
        holo_list = []
        for m in range(M):
            kx = k_batch[n, m, 0]
            ky = k_batch[n, m, 1]
            U_ref = torch.exp(1j * 2 * np.pi * (kx * x_grid / W + ky * y_grid / H))
            interference = comp_obj + U_ref
            intensity = torch.abs(interference) ** 2
            holo_list.append(intensity)
        sim_holograms.append(torch.stack(holo_list, dim=0))  # [M, H, W]

    return torch.stack(sim_holograms, dim=0)  # [N, M, H, W]


# ─────────────────────────────────────────────
# DATASET CHO OFF-AXIS HOLOGRAPHY
# ─────────────────────────────────────────────

class GedankenOffAxisDataset(torch.utils.data.Dataset):
    """
    Dataset tạo ra các cặp hologram off-axis mô phỏng từ ảnh nhân tạo.

    Quy trình mỗi sample:
      1. Đọc ảnh nhân tạo PNG -> tạo trường phức Phase-only: O = exp(i*ph*pi*ang)
      2. Lấy ngẫu nhiên M cặp (kx, ky) từ vùng mask +1 đã auto-detect
      3. Mô phỏng M hologram: I_m = |O + U_ref_m|^2
      4. Trả về: (inp [M, H, W], tag [2, H, W], k_vecs [M, 2])
    """

    def __init__(self, file_paths, mask_plus1, M, trans, params):
        """
        Args:
            file_paths  : list đường dẫn ảnh PNG nhân tạo
            mask_plus1  : 2D bool mask [H, W] - vùng phổ +1 auto-detect được
            M           : số hologram per sample (thường = 2)
            trans       : torchvision transforms
            params      : dict chứa 'ph', 'noise_level', 'patch_size', ...
        """
        self.file_paths = file_paths
        self.mask_plus1 = mask_plus1
        self.m = M
        self.trans = trans
        self.params = params
        self.S = params['patch_size']
        self.cx = self.S // 2
        self.cy = self.S // 2

        # Lưu trước tọa độ các pixel trong mask để sample nhanh
        rows, cols = np.where(mask_plus1)
        self.mask_rows = rows
        self.mask_cols = cols
        print(f"[INFO] GedankenOffAxisDataset: {len(file_paths)} files, "
              f"+1 mask has {len(rows)} pixels, M={M} holograms per sample")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # 1. Đọc ảnh nhân tạo và tạo trường phức Phase-only
        ang_img = np.array(PIL.Image.open(self.file_paths[index])).astype('float32') / 255
        ang = self.trans(ang_img[:, :, 0]).numpy().squeeze()
        s = ang.shape[-1]
        comp_field = np.exp(1j * self.params['ph'] * np.pi * ang)

        # 2. Downsampling + upsampling (giống GedankenNet_Phase để add low-freq prior)
        comp_field = skimage.measure.block_reduce(comp_field, block_size=2, func=np.mean)
        comp_field = scipy.ndimage.zoom(comp_field, 2, order=1)
        comp_field = scipy.ndimage.gaussian_filter(comp_field, sigma=1.0, mode='constant', cval=0)

        # 3. Thêm nhiễu trắng nếu cần
        if self.params.get('noise_level', 0) != 0:
            noise = np.random.normal(0, self.params['noise_level'], (s, s, 2)).view(np.complex128).reshape(s, s).astype(np.complex64)
            comp_field += noise

        # 4. Lấy ngẫu nhiên M cặp (kx, ky) TRONG vùng mask +1
        k_list = []
        for _ in range(self.m):
            idx = np.random.randint(len(self.mask_rows))
            ky = float(self.mask_rows[idx] - self.cy)
            kx = float(self.mask_cols[idx] - self.cx)
            k_list.append((kx, ky))

        # 5. Mô phỏng M hologram off-axis
        holo_list = []
        for (kx, ky) in k_list:
            holo = simulate_offaxis_hologram(comp_field, kx, ky)
            holo_list.append(holo)

        inp = np.stack(holo_list, axis=0).astype('float32')  # [M, H, W]

        # 6. Ground truth (chỉ dùng để monitor, không dùng trong loss)
        re, im = np.real(comp_field), np.imag(comp_field)
        tag = np.stack((re, im), axis=0)  # [2, H, W]
        k_vecs = np.array(k_list, dtype=np.float32)  # [M, 2]

        return torch.Tensor(inp), torch.Tensor(tag), torch.Tensor(k_vecs)


# ─────────────────────────────────────────────
# Utility: đếm params model
# ─────────────────────────────────────────────

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size() + (2,) if p.is_complex() else p.size()))
    return c
