# Smart Traffic Management System - YOLO Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-orange)](https://ultralytics.com/)

Smart Traffic Management System using custom YOLO models for **ANPR (Automatic Number Plate Recognition)** and **ATCC (Automatic Traffic Count and Classification)**.

---

## Features

- Dual YOLO Models - ANPR for license plates, ATCC for traffic objects
- Image Detection - Upload and process JPG/PNG images
- Video Detection - Process MP4/MOV/AVI videos frame-by-frame
- OCR Detection - Extract license plate text from detected plates
- Real-time Analytics - View detection counts by class
- Download Results - Save processed images and videos
- Adjustable Confidence - Fine-tune detection sensitivity

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Smart-Traffic-Management-System.git
cd Smart-Traffic-Management-System/Smart_Traffic_Management_System/Smart_Traffic_Management_System

# Install dependencies
pip install -r requirements.txt
```

### Add Your YOLO Models

Place your trained models in the project directory:

```
Smart_Traffic_Management_System/Smart_Traffic_Management_System/
├── streamlit_app.py
├── yolo_ANPR.pt          (ADD YOUR ANPR MODEL)
└── yolo_ATCC.pt          (ADD YOUR ATCC MODEL)
```

### Run the App

```bash
streamlit run streamlit_app.py
```

App opens at: `http://localhost:8501`

---

## Project Structure

```
STMS/
├── anpr/                                    # ANPR Dataset
│   ├── data.yaml                           # Dataset configuration
│   ├── images/
│   │   ├── train/                          # 25,470 training images
│   │   ├── val/                            # 1,073 validation images
│   │   └── test/                           # 386 test images
│   └── labels/                             # YOLO format labels
│
├── ATCC/                                    # ATCC Dataset (BDD100K)
│   ├── data.yaml                           # Dataset configuration
│   ├── bdd100k/
│   │   └── bdd100k/images/100k/
│   │       ├── train/                      # 1,156 training images
│   │       └── val/                        # 10,000 validation images
│   ├── bdd100k_labels_release/             # JSON labels
│   └── bdd100k_seg/                        # Segmentation data
│
├── Smart_Traffic_Management_System/
│   ├── train_models.py                     # Training script
│   └── Smart_Traffic_Management_System/
│       ├── streamlit_app.py                # Detection app with OCR
│       ├── requirements.txt                # Dependencies
│       ├── yolo_ANPR.pt                   # Trained ANPR model
│       └── yolo_ATCC.pt                   # Trained ATCC model
│
├── 1_ANPR_Training.ipynb                   # ANPR training notebook
├── 2_ATCC_Training.ipynb                   # ATCC training notebook
├── README.md                               # This file
└── PROJECT_DOCUMENTATION.md                # Complete documentation
```

---

## Models

### ANPR Model (yolo_ANPR.pt)
- Purpose: License plate detection and text recognition
- Classes: 1 (license plate)
- With OCR: Extracts actual plate text

### ATCC Model (yolo_ATCC.pt)
- Purpose: Traffic object detection and classification
- Classes: 10
  - pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign

---

## How to Use

1. Select Model - Choose ANPR or ATCC from dropdown
2. Upload File - Upload image or video
3. Adjust Confidence - Use slider to set detection threshold
4. Run Detection - Click button to process
5. View Results - See bounding boxes, statistics, and extracted text
6. Download - Save processed files

---

## Supported Formats

Images: .jpg, .jpeg, .png
Videos: .mp4, .mov, .avi

---

## Technologies

- Python 3.10+
- YOLOv8 (Ultralytics)
- Streamlit (Web Framework)
- OpenCV (Image/Video Processing)
- EasyOCR (Text Recognition)
- Pillow (Image Operations)
- Pandas (Data Processing)
- NumPy (Numerical Computing)

---

## Training Your Own Models

### Your Datasets are Ready

**ANPR Dataset:**
- 25,470 training images
- 1,073 validation images
- 386 test images
- Format: YOLO (images + labels)

**ATCC Dataset (BDD100K):**
- 1,156 training images
- 10,000 validation images
- Source: Berkeley DeepDrive
- 10 traffic object classes

### Train Models

```bash
cd Smart_Traffic_Management_System

# Install training dependencies
pip install ultralytics torch torchvision

# Train both models
python train_models.py --model both --epochs 100 --batch 16

# Or train individually
python train_models.py --model anpr --epochs 100
python train_models.py --model atcc --epochs 100
```

Training Time:
- With GPU: 3-5 hours per model
- Without GPU: 24-36 hours per model

Output: Creates yolo_ANPR.pt and yolo_ATCC.pt in the app directory

---

## Output Files

The system generates:
- detected_image_ANPR.png / detected_image_ATCC.png - Processed images
- output_ANPR.mp4 / output_ATCC.mp4 - Processed videos
- detections_data.csv - YOLO detection data

---

## Documentation

- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) - Complete documentation
- [1_ANPR_Training.ipynb](1_ANPR_Training.ipynb) - ANPR training guide
- [2_ATCC_Training.ipynb](2_ATCC_Training.ipynb) - ATCC training guide

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV Community
- Streamlit Team
- EasyOCR Contributors

---

