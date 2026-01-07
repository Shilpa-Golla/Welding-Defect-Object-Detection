# Project 1 — Welding Defect Object Detection (YOLOv8)

**Author:** Shilpa Golla  
**Course:** Representation Learning: From Neural Networks to Transformers  
**Project:** Project 1 – Welding Defect Object Detection  

This project develops, trains, and evaluates a CNN-based object detection model to identify and localize welding defects in images and videos using YOLOv8.

---

## Dataset

- **Welding Defect Object Detection**
- Source: Kaggle  
  https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection

**Classes**
- Bad Weld  
- Good Weld  
- Defect  

**Annotation format:** YOLO

---

## step 1: Environment Setup

Install the required dependencies:

```bash
pip install ultralytics opencv-python matplotlib scikit-learn pyyaml

```


## Dataset Folder Structure

After downloading and extracting the dataset, the folder structure must be:

```text
The Welding Defect Dataset/
│
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```
## Project Structure


```text
Project1_Shilpa/
│
├── notebooks/
│   ├── train.ipynb
│   ├── evaluate.ipynb
│   └── video_inference.ipynb
│
├── slides.pdf (or slides.pptx)
└── README.md
```
## Configuration

In all notebooks, update the dataset path:

from pathlib import Path

DATA_ROOT = Path(
    r"C:\Users\golla\Downloads\Welding\The Welding Defect Dataset\The Welding Defect Dataset"
)

## How to Run the Project (Notebook Workflow)
#### Training

Open and run:
### notebooks/train.ipynb
This notebook performs:
- Dataset inspection and sample visualization  
- Bounding box verification  
- Training YOLOv8s using transfer learning  
- Tracking training and validation loss and mAP@0.5  

**Training configuration**
- Model: YOLOv8s (pretrained)  
- Image size: 640 × 640  
- Epochs: 50  
- Batch size: 16  
- Random seed: 42  

**Outputs generated**
```text
runs/weld_yolov8s_640/
├── weights/
│   ├── best.pt
│   └── last.pt
└── results.png
```
## Evaluation

Open and run:
```text
notebooks/evaluate.ipynb
```
This notebook performs:
- Evaluation on the test dataset  
- Computes Precision, Recall, mAP@0.5, and mAP@0.5:0.95  
- Generates Precision–Recall curve, F1–confidence curve, and normalized confusion matrix  
- Saves good and failed prediction examples  
- Compares YOLOv8s with YOLOv8n  

**Outputs generated**
```text
runs/detect/val*/
pred_samples/yolov8s_test_preds/
```
## Video Inference

Open and run:
```text
notebooks/video_inference.ipynb
```

This notebook performs:
- Object detection on a short welding video  
- Saves the annotated output video  
- Computes average inference speed (FPS)  

**Outputs generated**
```text
video_out/weld_yolov8s_video/
```
## Hardware and Framework

- **Framework:** PyTorch (Ultralytics YOLOv8)  
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU  

---

## References

- Kaggle Welding Defect Object Detection Dataset  
- Ultralytics YOLOv8 Documentation  



