import os
from datasets import load_dataset

print("Downloading the ENTIRE LOL-v2-Real dataset from Hugging Face...")
# Loads the complete dataset
dataset = load_dataset("okhater/lolv2-real", split="train")

output_dir = "hf_lolv2_real/test/low"
os.makedirs(output_dir, exist_ok=True)

print(f"Total images found: {len(dataset)}. Downloading all of them now (this may take a minute)...")

for i, item in enumerate(dataset):
    img = item["image"]
    # Saves every single image without stopping
    img.save(os.path.join(output_dir, f"dark_image_{i}.png"))
    
print(f"Success! The full dataset has been saved into the '{output_dir}' folder.")