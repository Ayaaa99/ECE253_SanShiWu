import cv2
import numpy as np
import os
from skimage import restoration, img_as_float

# --- 设定路徑 ---
INPUT_DIR = "dataset_mild_blur"        # 用同一组模糊图
OUTPUT_DIR = "dataset_smooth_restored" # 输出到新文件夹

# --- 设定参数 ---
KERNEL_SIZE = 20
ANGLE = 0

# [关键策略] 
# 之前设 0.001 -> 锐利但有网格 (适合机器)
# 现在设 0.1   -> 平滑但稍软 (适合人类)
# 这就是最完美的 "Visual Comparison"！
BALANCE = 0.1 

def get_psf(kernel_size, angle):
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    psf = np.diag(np.ones(kernel_size))
    psf = cv2.warpAffine(psf, M, (kernel_size, kernel_size))
    psf = psf / kernel_size
    return psf

def restore(img, psf, balance):
    img_f = img_as_float(img)
    res = np.zeros_like(img_f)
    for i in range(3):
        # Wiener Filter 在高 Balance 下会自动压制网格
        res[:,:,i] = restoration.wiener(img_f[:,:,i], psf, balance=balance)
    return np.clip(res, 0, 1)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
psf = get_psf(KERNEL_SIZE, ANGLE)

print(f"正在生成「人类视觉友善版」(平滑无网格)...")

for filename in files:
    img = cv2.imread(os.path.join(INPUT_DIR, filename))
    if img is None: continue
    
    out = restore(img, psf, BALANCE)
    out_uint8 = (out * 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), out_uint8)

print("生成完毕！请去 dataset_smooth_restored 看看，这次绝对没有网格了。")