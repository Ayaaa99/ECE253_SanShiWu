import cv2
import numpy as np
import os

def apply_motion_blur(image, kernel_size=15, angle=0):
    """
    Applies motion blur to an image.

    :param image: Input image.
    :param kernel_size: Degree of blur (larger value means more blur). 
                        Suggested range: 5 (mild) ~ 30 (severe).
    :param angle: Angle of blur in degrees. 0=Horizontal, 90=Vertical, 45=Diagonal.
    :return: The blurred image.
    """
    # Create the motion blur kernel matrix
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(kernel_size))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel_size, kernel_size))
    
    # Normalize the kernel (to prevent brightness changes in the image)
    motion_blur_kernel = motion_blur_kernel / kernel_size
    
    # Apply the filter (Convolution)
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
    # Optional: Add slight noise to simulate realism
    # noise = np.random.normal(0, 5, blurred.shape).astype(np.uint8)
    # blurred = cv2.add(blurred, noise)
    
    return blurred

# --- Usage Example ---
# [cite_start]Read the clean baseline image (sourced from the Kaggle dataset mentioned in the proposal [cite: 96])
img = cv2.imread('clean_plate.jpg')

if img is not None:
    # Generate three levels of difficulty
    blur_easy = apply_motion_blur(img, kernel_size=10, angle=0)   # Mild horizontal blur
    blur_hard = apply_motion_blur(img, kernel_size=100, angle=0)  # Severe horizontal blur (Extreme stress test)
    blur_diag = apply_motion_blur(img, kernel_size=20, angle=45)  # Diagonal blur

    # Save the output images
    cv2.imwrite('plate_blur_easy.jpg', blur_easy)
    cv2.imwrite('plate_blur_hard.jpg', blur_hard)
    cv2.imwrite('plate_blur_diag.jpg', blur_diag)
    print("Blurred images generated successfully!")
else:
    print("Image not found. Please check the file path.")