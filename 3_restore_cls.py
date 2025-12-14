import cv2
import numpy as np
import os
from skimage import img_as_float

# --- 設定路徑 ---
INPUT_DIR = "dataset_mild_blur"
OUTPUT_DIR = "dataset_cls_restored"

# --- 參數調整 ---
KERNEL_SIZE = 20
ANGLE = 0

# [關鍵策略]
# 我們把 Lambda 設得非常大，強迫模型選擇「平滑」。
# 這會犧牲一點銳利度，但保證能消除那個恐怖的網格。
LAMBDA_PARAM = 0.5  # 之前是 0.1，現在加到 0.5 甚至 1.0

def get_motion_psf(shape, kernel_size, angle):
    psf = np.zeros(shape)
    center = (shape[1]//2, shape[0]//2)
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    small_kernel = np.diag(np.ones(kernel_size))
    small_kernel = cv2.warpAffine(small_kernel, M, (kernel_size, kernel_size))
    small_kernel = small_kernel / kernel_size
    k_h, k_w = small_kernel.shape
    y_off = center[1] - k_h // 2
    x_off = center[0] - k_w // 2
    psf[y_off:y_off+k_h, x_off:x_off+k_w] = small_kernel
    return psf

def cls_filter_ultimate(img, kernel_size, angle, lambda_param):
    img_float = img_as_float(img)
    
    # 1. Padding: 加大邊界，防止 FFT 邊界效應
    pad_h, pad_w = 50, 50
    img_padded = cv2.copyMakeBorder(img_float, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
    rows, cols = img_padded.shape[:2]
    
    # 2. 準備 PSF (H)
    psf = get_motion_psf((rows, cols), kernel_size, angle)
    psf = np.fft.fftshift(psf)
    H = np.fft.fft2(psf)
    
    # 3. 準備 Regularization Operator (L)
    # 這次我們使用簡單的平滑算子
    laplacian = np.array([[0, -1, 0], 
                          [-1, 4, -1], 
                          [0, -1, 0]])
    
    padded_lap = np.zeros((rows, cols))
    l_h, l_w = laplacian.shape
    padded_lap[rows//2 - l_h//2 : rows//2 + l_h//2 + 1, 
               cols//2 - l_w//2 : cols//2 + l_w//2 + 1] = laplacian
    padded_lap = np.fft.fftshift(padded_lap)
    L = np.fft.fft2(padded_lap)
    
    # 4. 執行 CLS
    restored_padded = np.zeros_like(img_padded)
    for i in range(3):
        G = np.fft.fft2(img_padded[:,:,i])
        numerator = np.conj(H) * G
        
        # [核心修正] 強力壓制
        # 我們把 Lambda 乘上一個較大的權重，強迫分母變大，消除高頻震盪
        denominator = np.abs(H)**2 + lambda_param * np.abs(L)**2
        
        # 避免分母太小
        denominator = np.maximum(denominator, 1e-5)
        
        F_hat = numerator / denominator
        restored_padded[:,:,i] = np.real(np.fft.ifft2(F_hat))
    
    # 5. Un-padding
    restored = restored_padded[pad_h:-pad_h, pad_w:-pad_w]
    return np.clip(restored, 0, 1)

# --- 主程式 ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"正在執行 CLS 終極平滑版 (Lambda={LAMBDA_PARAM})...")

for filename in files:
    img = cv2.imread(os.path.join(INPUT_DIR, filename))
    if img is None: continue
    
    out = cls_filter_ultimate(img, KERNEL_SIZE, ANGLE, LAMBDA_PARAM)
    out_uint8 = (out * 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), out_uint8)

print("CLS 處理完畢！這次應該會得到平滑且沒有網格的結果。")