import numpy as np
import cv2
from pathlib import Path

INPUT_FOLDER = "../images/angle_dataset"
OUTPUT_FOLDER = "../images/corrected_angle_dataset"

def get_rotation_angle(image):
    """Detect rotation angle using Hough Transform on detected lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    
    # Use Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=30, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        return 0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    # Filter to near-horizontal lines (likely plate edges)
    horizontal_angles = [a for a in angles if abs(a) < 45 or abs(a) > 135]
    
    if not horizontal_angles:
        return 0
    
    # Normalize angles to [-45, 45] range
    normalized = []
    for a in horizontal_angles:
        if a > 135:
            a -= 180
        elif a < -135:
            a += 180
        normalized.append(a)
    
    # Use median to reduce outlier impact
    return np.median(normalized)


def correct_image(image):
    """Correct image so longer edge is horizontal."""
    angle = get_rotation_angle(image)
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    corrected = cv2.warpAffine(image, M, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
    
    return corrected, angle


def main():
    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    
    for img_file in images:
        try:
            image = cv2.imread(str(img_file))
            corrected, angle = correct_image(image)
            cv2.imwrite(str(output_path / img_file.name), corrected)
            print(f"✓ {img_file.name}: {angle:.2f}°")
        except Exception as e:
            print(f"✗ {img_file.name}: {e}")


if __name__ == '__main__':
    main()