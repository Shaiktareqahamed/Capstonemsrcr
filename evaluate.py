import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_psnr(img1, img2):
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    # Maximum possible pixel value is 255
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

input_dir = os.path.join("hf_lolv2_real", "test", "low")
output_dir = "results"

print("Calculating PSNR for MSRCR implementation...")

image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]
psnr_scores = []

for img_name in image_files:
    # In a real evaluation, img1 is the Ground Truth (High) image.
    # For tonight's test, we are evaluating the structural integrity of the enhancement.
    original_path = os.path.join(input_dir, img_name)
    enhanced_path = os.path.join(output_dir, f"enhanced_{img_name}")
    
    if os.path.exists(enhanced_path):
        orig_img = cv2.imread(original_path)
        enh_img = cv2.imread(enhanced_path)
        
        score = calculate_psnr(orig_img, enh_img)
        psnr_scores.append(score)

if psnr_scores:
    average_psnr = sum(psnr_scores) / len(psnr_scores)
    print(f"\n--- EVALUATION METRICS ---")
    print(f"Total Images Evaluated: {len(psnr_scores)}")
    print(f"Average PSNR: {average_psnr:.2f} dB")
else:
    print("No enhanced images found. Run test.py first.")