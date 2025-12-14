import cv2
import os
import numpy as np

INPUT_DIR = "dataset_mild_restored"  
OUTPUT_DIR = "dataset_final_polished" 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(INPUT_DIR):
    print(f"error：cant find input folder '{INPUT_DIR}'！")
    exit()

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"checking folder：{os.path.abspath(INPUT_DIR)}")
print(f"found {len(files)} photos")

if len(files) == 0:
    print("warning：empty file，file name is incorrect！")
else:
    print("start restoring...")
    count = 0
    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"cant read：{filename}")
            continue
        
        polished = cv2.medianBlur(img, 3)
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), polished)
        count += 1

    print(f"Done！ {count} photos。")
