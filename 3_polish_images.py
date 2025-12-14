import cv2
import os
import numpy as np

# --- 設定 ---
INPUT_DIR = "dataset_mild_restored"  # 讀取原本長痘痘的修復圖
OUTPUT_DIR = "dataset_final_polished" # 存放磨皮後的漂亮圖

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(INPUT_DIR):
    print(f"錯誤：找不到輸入資料夾 '{INPUT_DIR}'！")
    exit()

# [修正點] 加入 .jpeg 支援
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"正在檢查資料夾：{os.path.abspath(INPUT_DIR)}")
print(f"找到 {len(files)} 張圖片")

if len(files) == 0:
    print("警告：資料夾是空的，或是副檔名不對！")
else:
    print("開始進行後處理 (磨皮)...")
    count = 0
    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"無法讀取：{filename}")
            continue
        
        # --- 使用中值濾波 (Median Blur) 去除雜訊痘痘 ---
        # ksize=3 代表 3x3 區域，數值越大越糊，3 是最適合保留文字的
        polished = cv2.medianBlur(img, 3)
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), polished)
        count += 1

    print(f"處理完成！成功處理 {count} 張圖片。")
    print(f"請去 {OUTPUT_DIR} 看看成果。")