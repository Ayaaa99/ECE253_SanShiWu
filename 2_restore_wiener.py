import cv2
import numpy as np
import os
from skimage import restoration, img_as_float

# --- 設定路徑 ---
INPUT_DIR = "dataset_mild_blur"
OUTPUT_DIR = "dataset_mild_restored"

# --- 參數 (必須跟生成時一樣) ---
KERNEL_SIZE = 20
ANGLE = 0
BALANCE = 0.01  # [修改] 稍微調大一點點，消除網格雜訊

def get_psf(kernel_size, angle):
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    psf = np.diag(np.ones(kernel_size))
    psf = cv2.warpAffine(psf, M, (kernel_size, kernel_size))
    psf = psf / kernel_size
    return psf

def restore_with_padding(img, psf, balance):
    img_f = img_as_float(img)
    
    # 1. [關鍵步驟] Padding: 先把圖片擴大，用鏡像填補邊緣
    # 這能有效防止 FFT 造成的「邊緣振鈴效應」(那種網格紋)
    pad_h, pad_w = 30, 30 # 上下左右各補 30 像素
    img_padded = cv2.copyMakeBorder(img_f, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
    
    res_padded = np.zeros_like(img_padded)
    
    # 2. 執行修復
    for i in range(3):
        res_padded[:,:,i] = restoration.wiener(img_padded[:,:,i], psf, balance=balance)
        
    # 3. [關鍵步驟] Un-Padding: 把剛剛擴大的部分切掉，只留中間
    res = res_padded[pad_h:-pad_h, pad_w:-pad_w]
    
    return np.clip(res, 0, 1)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
psf = get_psf(KERNEL_SIZE, ANGLE)

print(f"開始修復 {len(files)} 張圖片 (使用 Padding 技術消除波紋)...")

for filename in files:
    img = cv2.imread(os.path.join(INPUT_DIR, filename))
    if img is None: continue
    
    out = restore_with_padding(img, psf, BALANCE)
    out_uint8 = (out * 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), out_uint8)

print("修復完成！這次網格紋應該會消失了。")