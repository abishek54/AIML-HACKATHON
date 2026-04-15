import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- 1. CONFIGURATION ---
DATA_DIR = r'D:\Hackathon_Data\Tiles'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 
EPOCHS = 30
LR = 0.0001

# --- 2. THE IOU-MAXIMIZING ARCHITECTURE ---
# DeepLabV3+ uses Atrous convolutions (ASPP) to capture tiny features
model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b0", 
    classes=10, 
    encoder_weights="imagenet"
).to(DEVICE)

# --- 3. DUAL-LOSS STRATEGY ---
# Lovasz maximizes IoU directly; Focal handles the "hard" pixels
focal_loss = smp.losses.FocalLoss(mode='multiclass')
lovasz_loss = smp.losses.LovaszLoss(mode='multiclass')

def criterion(preds, targets):
    return (0.5 * focal_loss(preds, targets)) + (0.5 * lovasz_loss(preds, targets))

# --- 4. DATASET & CLASS-AWARE SAMPLER ---
class SvamitvaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        
        # Pre-calculate rare-class presence for the sampler
        self.weights = []
        print("🔍 Calculating weights for 95% boost...")
        for img_name in self.images:
            mask = cv2.imread(os.path.join(self.mask_dir, img_name), 0)
            # Higher weight if mask contains Utilities (9) or Water Points (6)
            if np.any(np.isin(mask, [3, 6, 7, 8, 9])): 
                self.weights.append(10.0) 
            else:
                self.weights.append(1.0)

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, self.images[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, self.images[idx]), 0)
        mask[mask >= 10] = 0
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask'].long()
        return img, mask

# --- 5. INITIALIZATION ---
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5), A.GaussianBlur(p=0.2),
    A.Normalize(), ToTensorV2()
])

dataset = SvamitvaDataset(DATA_DIR, transform=train_transform)
# Sampler ensures batches are balanced with rare features
sampler = WeightedRandomSampler(dataset.weights, len(dataset.weights))
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler('cuda')

# --- 6. TRAINING LOOP ---
if __name__ == '__main__':
    print("🚀 Starting Final 80% IoU Optimization...")
    for epoch in range(EPOCHS):
        model.train()
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if i % 100 == 0:
                print(f"Epoch {epoch+1} | Step {i} | Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), "final_80plus_model.pth")