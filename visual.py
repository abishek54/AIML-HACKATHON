import cv2
import numpy as np
import os

# --- 1. CONFIGURATION ---
MASK_PATH = r'D:\Hackathon_Data\Final_Submissions\Live_Demo\ANAITPURA_FATEHGARH SAHIB_32705_ORTHO\ANAITPURA_FATEHGARH SAHIB_32705_ORTHO.png'
ORIGINAL_IMG_PATH = r'D:\Hackathon_Data\live_demo_2\ANAITPURA_FATEHGARH SAHIB_32705_ORTHO\ANAITPURA_FATEHGARH SAHIB_32705_ORTHO.tif'
SAVE_PATH = r'D:\Hackathon_Data\Final_Submissions\Live_Demo\Visual_Comparison.png'

# Color Map (BGR format for OpenCV)
COLORS = {
    1: [255, 255, 255], # Building: White
    2: [128, 128, 128], # Road Network: Grey
    4: [255, 0, 0],     # Water: Blue
    8: [0, 255, 255]    # Infrastructure: Yellow
}

def colorize_and_compare():
    # Load the "blank" mask and original image
    mask = cv2.imread(MASK_PATH, 0)
    original = cv2.imread(ORIGINAL_IMG_PATH)
    
    if mask is None or original is None:
        print("❌ Error: Files not found. Check your paths.")
        return

    # Create a colored version of the mask
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for val, color in COLORS.items():
        colored_mask[mask == val] = color

    # Resize for easy viewing if the image is massive
    display_size = (1024, 1024)
    original_small = cv2.resize(original, display_size)
    mask_small = cv2.resize(colored_mask, display_size)

    # Combine side-by-side for comparison
    comparison = np.hstack((original_small, mask_small))
    
    cv2.imwrite(SAVE_PATH, comparison)
    print(f"✅ Success! View your result at: {SAVE_PATH}")

if __name__ == "__main__":
    colorize_and_compare()