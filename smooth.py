import cv2
import numpy as np
import os
from skimage import restoration, img_as_float


INPUT_DIR = "dataset_mild_blur"        
OUTPUT_DIR = "dataset_smooth_restored"

#Parameters
KERNEL_SIZE = 20
ANGLE = 0

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
        res[:,:,i] = restoration.wiener(img_f[:,:,i], psf, balance=balance)
    return np.clip(res, 0, 1)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
psf = get_psf(KERNEL_SIZE, ANGLE)

print(f"Executing...")

for filename in files:
    img = cv2.imread(os.path.join(INPUT_DIR, filename))
    if img is None: continue
    
    out = restore(img, psf, BALANCE)
    out_uint8 = (out * 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), out_uint8)

print("Done!")
