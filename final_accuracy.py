import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

# Suppress warnings for a clean terminal output
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
DATA_DIR = r'D:\Hackathon_Data\Hard_Tiles' 
MODEL_PATH = r'D:\Hackathon_Data\svamitwa_95plus_model.pth' 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The 5 Functional Categories that satisfy the SVAMITVA scheme
FINAL_CLASSES = {
    0: "Background",
    1: "Building Footprints",
    2: "Road Networks",
    4: "Water Infrastructure",
    8: "Critical Infrastructure"
}

def evaluate_accuracy():
    print("🚀 Initializing SVAMITVA Official Accuracy Evaluation...")
    
    # Load your elite 15-class architecture
    model = smp.DeepLabV3Plus(encoder_name="efficientnet-b0", classes=15).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    img_dir = os.path.join(DATA_DIR, 'images')
    mask_dir = os.path.join(DATA_DIR, 'masks')
    images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
    
    total_inter = {c: 0 for c in FINAL_CLASSES}
    total_union = {c: 0 for c in FINAL_CLASSES}
    
    # --- NEW: Pixel Accuracy Trackers ---
    total_correct_pixels = 0
    total_evaluated_pixels = 0
    
    transform = A.Compose([A.Normalize(), ToTensorV2()])

    for f in tqdm(images, desc="Evaluating Hard Tiles"):
        # Load Image and Ground Truth
        img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f)), cv2.COLOR_BGR2RGB)
        target = cv2.imread(os.path.join(mask_dir, f), 0)
        
        # --- APPLY GROUND TRUTH AGGREGATION LOGIC ---
        target[np.isin(target, [3])] = 2          # Road Center -> Road
        target[np.isin(target, [5, 6])] = 4       # Water Line/Point -> Water Body
        target[np.isin(target, [7, 9])] = 8       # Railway/Utility -> Critical Infra
        
        # Predict
        tensor = transform(image=img)['image'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = torch.argmax(model(tensor), dim=1).cpu().numpy()[0].astype(np.uint8)

        # --- APPLY PREDICTION AGGREGATION LOGIC ---
        pred[np.isin(pred, [1, 11, 12, 13, 14])] = 1 # All Building Types -> Building
        pred[np.isin(pred, [3])] = 2                 # Road Center -> Road
        pred[np.isin(pred, [5, 6])] = 4              # Water Line/Point -> Water Body
        pred[np.isin(pred, [7, 9])] = 8              # Railway/Utility -> Critical Infra

        # --- NEW: Calculate Pixel Accuracy ---
        # Compares every single pixel in the 512x512 tile
        total_correct_pixels += np.sum(pred == target)
        total_evaluated_pixels += target.size

        # --- CALCULATE IoU (Intersection over Union) ---
        for cls in FINAL_CLASSES:
            gt_mask = (target == cls)
            if not np.any(gt_mask): continue # Skip if this class isn't in this specific tile
            
            pred_mask = (pred == cls)
            inter = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            total_inter[cls] += inter
            total_union[cls] += union

    # --- PRINT FINAL REPORT ---
    print("\n" + "="*50)
    print("🏆 FINAL AGGREGATED IoU & PIXEL ACCURACY REPORT 🏆")
    print("="*50)
    
    ious = []
    for cls, name in FINAL_CLASSES.items():
        if total_union[cls] > 0:
            iou = (total_inter[cls] / total_union[cls]) * 100
            print(f"{name:<25} : {iou:.2f}% Accuracy (IoU)")
            ious.append(iou)
    
    mean_iou = np.mean(ious)
    overall_pixel_accuracy = (total_correct_pixels / total_evaluated_pixels) * 100
    
    print("-" * 50)
    print(f"🌟 MEAN IoU (mIoU)        : {mean_iou:.2f}%")
    print(f"🎯 PIXEL ACCURACY         : {overall_pixel_accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate_accuracy()