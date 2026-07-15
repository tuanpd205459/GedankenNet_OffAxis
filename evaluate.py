import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from my_tools import pad_and_crop, auto_detect_carrier, RealOffAxisDataset, batch_offaxis_interference

def main():
    S = 512
    M = 2
    params = {
        'wavelength': 0.530,
        'pixel_size': 0.3733,
        'patch_size': S,
        'ref_ind': 1.00,
        'ph': 1.0,           
        'noise_level': 0.005 * np.sqrt(2), 
    }
    
    print("="*60)
    print("🔍 [Bước 1] Đang tìm mô hình (Não) xịn nhất...")
    model_dirs = glob.glob('Models/OffAxis*')
    if not model_dirs:
        print("❌ Không tìm thấy thư mục Models. Bạn đã train chưa?")
        return
    
    latest_dir = max(model_dirs, key=os.path.getmtime)
    pth_files = glob.glob(os.path.join(latest_dir, '*.pth'))
    if not pth_files:
        print("❌ Không tìm thấy file .pth nào.")
        return
        
    best_model_path = max(pth_files, key=os.path.getmtime)
    print(f"✅ Đã tải Model: {best_model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(best_model_path, map_location=device)
    model.eval()

    print("\n🔍 [Bước 2] Đang chuẩn bị dữ liệu đánh giá...")
    raw_images = glob.glob('data_raw/*.bmp')
    if len(raw_images) < M:
        print(f"❌ Cần ít nhất {M} ảnh .bmp trong thư mục data_raw")
        return
        
    # Auto-detect mask từ ảnh đầu tiên
    sample_holo = plt.imread(raw_images[0])
    if sample_holo.ndim == 3: sample_holo = sample_holo[:, :, 0]
    sample_holo = pad_and_crop(sample_holo, S)
    mask_plus1, kx0, ky0 = auto_detect_carrier(sample_holo, dc_mask_ratio=0.15)
    
    # Tạo dict chứa sample để cho vào Dataset
    sample_names = ['eval_sample']
    sample_dict = {'eval_sample': raw_images[:M]} 
    
    dataset = RealOffAxisDataset(sample_names, sample_dict, mask_plus1, M, S)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("\n🚀 [Bước 3] Đang tiến hành đánh giá (Evaluation)...")
    for xx, yy, kk in loader:
        xx = xx.to(device)
        kk = kk.to(device)
        
        with torch.no_grad():
            im, _ = model(xx)
            # Tái tạo lại Hologram từ Pha dự đoán (Kiểm chứng tính đúng đắn vật lý)
            im_x = batch_offaxis_interference(im, kk, params)
            
            # Tính sai số vật lý (MSE)
            mse = torch.nn.functional.mse_loss(im_x, xx).item()
            print(f"🎯 Sai số vật lý (Physics MSE): {mse:.6f}")
            
        # Lấy dữ liệu để vẽ ảnh
        holo_real = xx[0, 0].cpu().numpy()
        holo_recon = im_x[0, 0].cpu().numpy()
        phase_pred = im[0, 0].cpu().numpy()
        
        # Chuẩn hoá pha để hiển thị đẹp nhất
        phase_pred = np.clip((phase_pred - phase_pred.min()) / (phase_pred.max() - phase_pred.min() + 1e-8), 0, 1)
        
        # Vẽ biểu đồ so sánh
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(holo_real, cmap='gray')
        axes[0].set_title("1. Hologram gốc (Thực tế)", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(phase_pred, cmap='gray')
        axes[1].set_title("2. Pha siêu nét (AI Phục hồi)", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(holo_recon, cmap='gray')
        axes[2].set_title(f"3. Hologram tái tạo ngược\n(Để kiểm chứng MSE: {mse:.4f})", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        os.makedirs('outputs/OffAxis', exist_ok=True)
        out_path = 'outputs/OffAxis/evaluation_result.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"\n🎉 [Hoàn tất] Đã lưu ảnh phân tích kết quả tại: {out_path}")
        print("="*60)
        break

if __name__ == '__main__':
    main()
