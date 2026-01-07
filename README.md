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

