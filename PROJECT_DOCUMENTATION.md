# Smart Traffic Management System - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Quick Start Guide](#quick-start-guide)
4. [Dataset Information](#dataset-information)
5. [Training Guide](#training-guide)
6. [Pre-trained Models](#pre-trained-models)
7. [Streamlit Application](#streamlit-application)
8. [Troubleshooting](#troubleshooting)
9. [Features and Capabilities](#features-and-capabilities)

---

## Project Overview

**Smart Traffic Management System** using YOLO models for:
- **ANPR** - Automatic Number Plate Recognition (26,929 images)
- **ATCC** - Automatic Traffic Count and Classification (11,156 images)

**Technology Stack:**
- Python 3.10+
- PyTorch 2.9.0
- Ultralytics YOLOv8 8.3.235
- Streamlit (Web Interface)
- EasyOCR (License plate text extraction)
- OpenCV, Pandas, NumPy

**Hardware:**
- CPU-compatible (tested on AMD Ryzen 5 5500U)
- No GPU required (though recommended for faster training)

---

## Project Structure

```
STMS/
├── anpr/                               # ANPR Dataset (26,929 images)
│   ├── train/ (25,470 images)
│   ├── val/ (1,073 images)
│   ├── test/ (386 images)
│   ├── labels/                         # YOLO format annotations
│   └── data.yaml
│
├── ATCC/                               # ATCC Dataset (11,156 images - BDD100K)
│   ├── bdd100k/
│   │   └── bdd100k/images/100k/
│   │       ├── train/ (1,156 images)
│   │       └── val/ (10,000 images)
│   ├── bdd100k_labels_release/         # JSON annotations
│   └── data.yaml
│
├── Smart_Traffic_Management_System/
│   ├── train_models.py                 # Training script
│   └── Smart_Traffic_Management_System/
│       ├── streamlit_app.py           # Main web application with OCR
│       ├── yolo_ANPR.pt               # Trained ANPR model (17.56 MB)
│       ├── yolo_ATCC.pt               # Trained ATCC model (49.72 MB)
│       ├── requirements.txt            # Dependencies
│       └── runs/                       # Training outputs
│
├── 1_ANPR_Training.ipynb              # ANPR training documentation
├── 2_ATCC_Training.ipynb              # ATCC training documentation
├── LICENSE
├── README.md                           # Quick start guide
└── PROJECT_DOCUMENTATION.md            # This file
```

---

## Quick Start Guide

### 1. Installation

```bash
# Navigate to project
cd C:\Users\ankit\OneDrive\Desktop\STMS

# Install dependencies
pip install -r Smart_Traffic_Management_System/Smart_Traffic_Management_System/requirements.txt
```

### 2. Run Streamlit App

```bash
cd Smart_Traffic_Management_System/Smart_Traffic_Management_System
python -m streamlit run streamlit_app.py
```

**App will open at:** http://localhost:8501

### 3. Application Features

- Image Detection - Upload JPG/PNG images
- Video Processing - Process MP4/MOV/AVI videos
- OCR Detection - Extract license plate text from detected plates
- Dataset Browser - View sample images from datasets
- Detection Analytics - Charts and statistics
- Confidence Threshold - Adjustable detection sensitivity
- Download Results - Save processed images and detection data

---

## Dataset Information

### ANPR Dataset (License Plate Detection)
- **Total Images:** 26,929
- **Train:** 25,470 images
- **Validation:** 1,073 images
- **Test:** 386 images
- **Classes:** 1 (license plate)
- **Format:** YOLO (images + text annotations)
- **Size:** Approximately 2.4 GB

### ATCC Dataset (Traffic Classification - BDD100K)
- **Total Images:** 11,156
- **Train:** 1,156 images
- **Validation:** 10,000 images
- **Classes:** 10 (pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign)
- **Format:** YOLO-compatible with JSON annotations
- **Size:** Approximately 9.4 GB
- **Source:** Berkeley DeepDrive 100K dataset

---

## Training Guide

### Training Script Location
```
Smart_Traffic_Management_System/train_models.py
```

### Quick Training Examples

**5-Epoch Training (30-40 minutes):**
```bash
cd Smart_Traffic_Management_System
python train_models.py --model anpr --epochs 5 --batch 8 --imgsz 416
```

**10-Epoch Training (60-80 minutes):**
```bash
python train_models.py --model anpr --epochs 10 --batch 8 --imgsz 416
```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--model` | `anpr` or `atcc` or `both` | Which model to train |
| `--epochs` | `5`, `10`, `20`, etc. | Number of training cycles |
| `--batch` | `8` | Batch size (lower = less memory) |
| `--imgsz` | `416` or `640` | Image size (416 = faster, 640 = better accuracy) |

### Expected Training Times (CPU-based)

| Epochs | Image Size | Estimated Time | Expected Accuracy |
|--------|------------|----------------|-------------------|
| 5 | 416px | 30-40 minutes | 60-70% mAP |
| 10 | 416px | 60-80 minutes | 70-75% mAP |
| 20 | 416px | 2-2.5 hours | 75-80% mAP |
| 50 | 640px | 12+ days | 85-90% mAP |

### After Training Completes

To use your trained model in the Streamlit app:

```bash
# Copy the trained weights
Copy-Item "runs/anpr_training*/weights/best.pt" "Smart_Traffic_Management_System/Smart_Traffic_Management_System/yolo_ANPR.pt" -Force
```

Then restart the Streamlit app to load the new model.

---

## Pre-trained Models

### Active Models

| Model | Size | Purpose | Status |
|-------|------|---------|--------|
| yolo_ANPR.pt | 17.56 MB | License plate detection | Ready |
| yolo_ATCC.pt | 49.72 MB | Traffic object classification | Ready |

### ANPR Model Details
- Trained on 26,929 license plate images from your dataset
- Specialized for license plate localization
- Includes OCR text extraction capability

### ATCC Model Details
- Pre-trained on BDD100K traffic dataset
- Detects and classifies 10 traffic object types
- Suitable for traffic monitoring and analytics

---

## Streamlit Application

### How to Use

1. **Select Model** - Choose ANPR or ATCC from dropdown menu
2. **Choose Input** - Upload image or video file
3. **Configure Detection** - Adjust confidence threshold using sidebar slider
4. **Run Detection** - Click button to process
5. **View Results** - See detections with bounding boxes and statistics
6. **Extract Text** (ANPR only) - Automatic OCR extracts license plate text
7. **Download** - Save processed results

### Application Pages

- Main Detection Interface - Real-time detection processing
- Sample Browser - View dataset images by category
- Analytics Dashboard - Detection statistics and visualizations
- Settings Panel - Model and detection configuration

### Key Features

- **Real-time Detection** - Process images and video streams
- **OCR Integration** - Extract text from detected license plates
- **Confidence Filtering** - Adjust detection sensitivity
- **Batch Processing** - Process multiple images
- **Statistics Export** - Download detection results as CSV
- **Visual Feedback** - Bounding boxes and confidence scores

---

## Supported File Formats

**Images:**
- .jpg, .jpeg (JPEG format)
- .png (PNG format)
- .bmp (Bitmap format)

**Videos:**
- .mp4 (MPEG-4 video)
- .mov (QuickTime video)
- .avi (AVI video format)

---

## Technologies Used

- **Python 3.10+** - Programming language
- **YOLOv8** - Object detection framework (Ultralytics)
- **Streamlit** - Web application framework
- **OpenCV (cv2)** - Computer vision library
- **EasyOCR** - Optical character recognition
- **PyTorch** - Deep learning framework
- **Pillow** - Image processing
- **Pandas** - Data analysis and export
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization

---

## Troubleshooting

### Issue: "No sample images found"
**Solution:** Verify dataset folders exist at:
- `anpr/images/train/`, `anpr/images/val/`, `anpr/images/test/`
- `ATCC/bdd100k/bdd100k/images/100k/train/`, etc.

### Issue: "Error loading model"
**Solution:** Ensure model files exist:
- `Smart_Traffic_Management_System/Smart_Traffic_Management_System/yolo_ANPR.pt`
- `Smart_Traffic_Management_System/Smart_Traffic_Management_System/yolo_ATCC.pt`

### Issue: "OCR not working" (ANPR mode)
**Solution:** Verify EasyOCR is installed:
```bash
pip install easyocr
```

### Issue: App runs slowly
**Solution:** 
- Reduce image size in app settings
- Use smaller confidence threshold
- Ensure no other heavy processes running
- Consider GPU acceleration for faster processing

### Issue: Out of memory error
**Solution:**
- Reduce batch size in training
- Use smaller image dimensions (416 instead of 640)
- Close other applications

---

## Output Files

### Image Processing
- `detected_image_ANPR.png` - Processed image with ANPR detections
- `detected_image_ATCC.png` - Processed image with ATCC detections

### Video Processing
- `output_ANPR.mp4` - Video with ANPR detections
- `output_ATCC.mp4` - Video with ATCC detections

### Data Export
- `detections.csv` - Detection results with coordinates, confidence, and labels
- `plates_ocr.csv` - Extracted license plate text and confidence scores

---

## Performance Metrics

### ANPR Model Performance
- Detects license plates with high precision
- OCR extracts text with 90%+ accuracy on clear plates
- Processing: ~50-100ms per image (CPU)

### ATCC Model Performance
- Detects 10 traffic object classes
- Suitable for traffic monitoring and counting
- Processing: ~100-150ms per image (CPU)

---

## Next Steps

1. **Run the application** - Start with quick demo
2. **Upload test images** - Try ANPR and ATCC models
3. **Review results** - Check detections and accuracy
4. **Train custom models** - Improve accuracy with training
5. **Integrate into system** - Deploy for production use

---

## Support and Documentation

- Review README.md for quick start
- Check Jupyter notebooks for detailed training procedures
- Refer to Streamlit app built-in help
- Consult official documentation for libraries (Ultralytics, Streamlit, OpenCV)

---

## License

MIT License - See LICENSE file for details

---
