import os
import cv2
from models.msrcr import msrcr_enhance

# Windows-safe path handling
input_dir = os.path.join("hf_lolv2_real", "test", "low")
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

print("Starting MSRCR enhancement pipeline on dataset...")

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total = len(image_files)
print(f"Found {total} images to process.")

for idx, img_name in enumerate(image_files, 1):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    
    if img is not None:
        enhanced = msrcr_enhance(img)
        save_path = os.path.join(output_dir, f"enhanced_{img_name}")
        cv2.imwrite(save_path, enhanced)
        if idx % 25 == 0 or idx == total:
            print(f"[{idx}/{total}] Processed and saved: enhanced_{img_name}")

print("Processing complete. Enhanced images are saved in 'results/'.")