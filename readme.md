# 🗺️ SVAMITVA Smart-Map: AI-Based Feature Extraction

**A High-Precision 15-Class Semantic Segmentation System for Rural Cadastral Mapping**

This repository contains the official submission for the AI/ML Geospatial Hackathon. It provides an end-to-end deep learning pipeline designed to automate the extraction of critical village infrastructure from massive, high-resolution drone orthophotos (GeoTIFFs) under the SVAMITVA Scheme.

## 🚀 Project Overview
Smart rural planning requires precise identification of land-use features. Standard models struggle with the extreme class imbalance and massive pixel dimensions of rural drone surveys. This project solves these issues using a **DeepLabV3+ (EfficientNet-B0)** architecture paired with a novel **Hard-Tile Mining** curriculum and **Sliding-Window Inference**. 

The model successfully digests complex, multi-gigapixel village maps and extracts building footprints, road networks, water bodies, and critical infrastructure with an aggregated **89.85% Mean IoU**.



## 📊 Feature Classification Performance

The model predicts 15 highly granular classes which are logically aggregated into 5 core functional zones to satisfy the hackathon's deliverables.

| Functional Zone | Extracted Features | Accuracy (IoU) |
| :--- | :--- | :--- |
| **Background** | Barren land, general agriculture | **95.43%** |
| **Water Infrastructure** | Ponds, Reservoirs, Water Lines, Wells | **94.43%** |
| **Building Footprints** | RCC, Tiled, and Tin Rooftops | **93.51%** |
| **Road Networks** | Asphalt roads, unpaved streets, road centers | **90.54%** |
| **Critical Infrastructure** | Bridges, Railway Tracks, Transformers | **75.32%** |
| **🏆 FINAL MEAN IoU** | **Overall Aggregated Score** | **89.85%** |

---

## 🧠 Core Methodology & Technical Edge

1. **Two-Stage "Hard-Tile" Mining:** To overcome the 0% accuracy drop on rare features (like railways and water lines), the model was first trained on 10,470 general tiles, then strictly fine-tuned on 3,546 specialized "Hard Tiles" containing dense infrastructure.
2. **Mega-Image Sliding Window Inference:** Village orthophotos routinely exceed 28,000 x 26,000 pixels. The inference pipeline bypasses OpenCV memory limits using `tifffile` and processes the map in seamless 512x512 patches with 16-pixel divisibility padding.
3. **LZW Compression Handling:** Integrated `imagecodecs` to natively decode highly compressed government survey GeoTIFFs without RAM overflow.



## 📂 Repository Structure


Hackathon_Data/
│
├── Hard_Tiles/                 # Validation dataset with ground truth masks
├── live_demo_2/                # Unseen mega-image test data (Anaitpura & Diwana)
├── Master_Masks/               # Original full-resolution training masks
│
├── Final_Submissions/          # 🏆 Output Directory
│   ├── Live_Demo/              # Raw prediction masks (1, 2, 4, 8) for evaluation
│   └── Visual_Reports/         # High-res, color-coded presentation overlays
│
├── svamitwa_95plus_model.pth   # Final Trained Weights (19.0 MB)
│
├── train.py                    # Model training and optimization pipeline
├── final_accuracy.py           # Evaluates 89.85% mIoU against Ground Truth
├── test.py                     # Mega-image Sliding Window Inference script
└── visual.py                   # Generates side-by-side presentation comparisons


## ⚙️ Installation & Requirements

Ensure you have a CUDA-enabled GPU for optimal inference speed. Install the required dependencies:


# Upgrade pip to ensure smooth installation of image codecs
python -m pip install --upgrade pip

# Install required libraries
pip install -r requirements.txt


## 💻 How to Run the Pipeline

### 1. Verify Official Accuracy
To reproduce the **89.85% Mean IoU** on the specialized validation dataset, run:

python final_accuracy.py

*Outputs a detailed class-by-class terminal report.*

### 2. Generate Predictions for Unseen Villages (Live Demo)
To process massive GeoTIFFs (like Diwana and Anaitpura) and generate feature-extracted blueprint masks:


python test.py

*Outputs raw `.png` masks to `Final_Submissions/Live_Demo/`.*

### 3. Generate Visual Reports for Judges
To create the 4K side-by-side comparison images overlaying the model's predictions onto the raw drone imagery:


python visual.py

*Outputs colored `.jpg` reports to `Final_Submissions/Visual_Reports/`.*