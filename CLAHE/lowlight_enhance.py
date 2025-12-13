#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch CLAHE enhancement on color images (Y channel only)

Example:
    python lowlight_enhance.py --test_dir ./data/lowlight_small/ --save_dir ./lowlight_test/
"""

import os
import argparse
import cv2


def enhance_plate_color_clahe(img_bgr, clipLimit=3.0, tileGridSize=(8, 8)):
    """Apply CLAHE on the Y channel of a BGR image."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    y_clahe = clahe.apply(y)

    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    return cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch CLAHE enhancement on color images"
    )
    parser.add_argument(
        "--test_dir", required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--save_dir", required=True,
        help="Directory to save enhanced images"
    )
    parser.add_argument(
        "--clip-limit", type=float, default=3.0, dest="clip_limit",
        help="CLAHE clipLimit (default: 3.0)"
    )
    parser.add_argument(
        "--tile-size", nargs=2, type=int, default=[8, 8],
        help="CLAHE tileGridSize, e.g., 8 8"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    test_dir = args.test_dir
    save_dir = args.save_dir

    if not os.path.isdir(test_dir):
        raise NotADirectoryError(f"Invalid input directory: {test_dir}")

    os.makedirs(save_dir, exist_ok=True)

    tile = tuple(args.tile_size)
    img_names = sorted([f for f in os.listdir(test_dir) if is_image_file(f)])

    if not img_names:
        print(f"No image files found in: {test_dir}")
        return

    print(f"Found {len(img_names)} images.")
    print(f"Saving results to: {save_dir}")
    print(f"CLAHE settings: clipLimit={args.clip_limit}, tileGridSize={tile}")
    print("-" * 50)

    for name in img_names:
        in_path = os.path.join(test_dir, name)
        img = cv2.imread(in_path)

        if img is None:
            print(f"[Skipped] Cannot read: {in_path}")
            continue

        enhanced = enhance_plate_color_clahe(
            img,
            clipLimit=args.clip_limit,
            tileGridSize=tile
        )

        base, ext = os.path.splitext(name)
        out_path = os.path.join(save_dir, base + "_clahe" + ext)

        if cv2.imwrite(out_path, enhanced):
            print(f"[OK] {in_path} -> {out_path}")
        else:
            print(f"[Failed] Cannot save: {out_path}")

    print("-" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
