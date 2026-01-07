{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2340d11c-5c51-4841-8b23-d02ee87b4929",
   "metadata": {},
   "source": [
    "# Project 1 — Welding Defect Object Detection (YOLOv8)\n",
    "\n",
    "**Author:** Shilpa Golla  \n",
    "**Course:** Representation Learning: From Neural Networks to Transformers  \n",
    "**Project:** Project 1 – Welding Defect Object Detection  \n",
    "\n",
    "This project develops, trains, and evaluates a CNN-based object detection model to identify and localize welding defects in images and videos using YOLOv8.\n",
    "\n",
    "---\n",
    "\n",
    "## Dataset\n",
    "\n",
    "- **Welding Defect Object Detection**\n",
    "- Source: Kaggle  \n",
    "  https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection\n",
    "\n",
    "**Classes**\n",
    "- Bad Weld  \n",
    "- Good Weld  \n",
    "- Defect  \n",
    "\n",
    "**Annotation format:** YOLO\n",
    "\n",
    "---\n",
    "\n",
    "## step 1: Environment Setup\n",
    "\n",
    "Install the required dependencies:\n",
    "\n",
    "```bash\n",
    "pip install ultralytics opencv-python matplotlib scikit-learn pyyaml\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dd687efa-9fea-4b14-b8ce-3b48d2cdf66c",
   "metadata": {},
   "source": [
    "## Dataset Folder Structure\n",
    "\n",
    "After downloading and extracting the dataset, the folder structure must be:\n",
    "\n",
    "```text\n",
    "The Welding Defect Dataset/\n",
    "│\n",
    "├── train/\n",
    "│   ├── images/\n",
    "│   └── labels/\n",
    "├── valid/\n",
    "│   ├── images/\n",
    "│   └── labels/\n",
    "├── test/\n",
    "│   ├── images/\n",
    "│   └── labels/\n",
    "└── data.yaml\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "235c1fdd-bcbc-43ed-b622-28a8e63d58a3",
   "metadata": {},
   "source": [
    "## Project Structure"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c073ce5-a6b6-480f-8f83-2b02aac93bac",
   "metadata": {},
   "source": [
    "```text\n",
    "Project1_Shilpa/\n",
    "│\n",
    "├── notebooks/\n",
    "│   ├── train.ipynb\n",
    "│   ├── evaluate.ipynb\n",
    "│   └── video_inference.ipynb\n",
    "│\n",
    "├── slides.pdf (or slides.pptx)\n",
    "└── README.md\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6fac0ef1-b6c3-4ac4-9002-e127500f4a0a",
   "metadata": {},
   "source": [
    "## Configuration\n",
    "\n",
    "In all notebooks, update the dataset path:\n",
    "\n",
    "from pathlib import Path\n",
    "\n",
    "DATA_ROOT = Path(\n",
    "    r\"C:\\Users\\golla\\Downloads\\Welding\\The Welding Defect Dataset\\The Welding Defect Dataset\"\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "456b1a1c-6545-40fa-904d-e81753f4d6d4",
   "metadata": {},
   "source": [
    "## How to Run the Project (Notebook Workflow)\n",
    "#### Training\n",
    "\n",
    "Open and run:\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ec03032c-eddd-4a3d-894b-da21e04d12eb",
   "metadata": {},
   "source": [
    "### notebooks/train.ipynb\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e362e1b4-d898-4a0e-bf3c-1b96b6920ffe",
   "metadata": {},
   "source": [
    "This notebook performs:\n",
    "- Dataset inspection and sample visualization  \n",
    "- Bounding box verification  \n",
    "- Training YOLOv8s using transfer learning  \n",
    "- Tracking training and validation loss and mAP@0.5  \n",
    "\n",
    "**Training configuration**\n",
    "- Model: YOLOv8s (pretrained)  \n",
    "- Image size: 640 × 640  \n",
    "- Epochs: 50  \n",
    "- Batch size: 16  \n",
    "- Random seed: 42  \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dcaf04ff-5c69-4238-b414-e123c39dad3a",
   "metadata": {},
   "source": [
    "**Outputs generated**\n",
    "```text\n",
    "runs/weld_yolov8s_640/\n",
    "├── weights/\n",
    "│   ├── best.pt\n",
    "│   └── last.pt\n",
    "└── results.png\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6930c1c2-24d2-40df-b8e1-dc0c44242971",
   "metadata": {},
   "source": [
    "## Evaluation\n",
    "\n",
    "Open and run:\n",
    "```text\n",
    "notebooks/evaluate.ipynb\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "38f80fdb-e62b-47d1-a4a7-f07f8327028d",
   "metadata": {},
   "source": [
    "This notebook performs:\n",
    "- Evaluation on the test dataset  \n",
    "- Computes Precision, Recall, mAP@0.5, and mAP@0.5:0.95  \n",
    "- Generates Precision–Recall curve, F1–confidence curve, and normalized confusion matrix  \n",
    "- Saves good and failed prediction examples  \n",
    "- Compares YOLOv8s with YOLOv8n  \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35e936ef-e906-4e32-8e6e-87b83b4ac1dc",
   "metadata": {},
   "source": [
    "**Outputs generated**\n",
    "```text\n",
    "runs/detect/val*/\n",
    "pred_samples/yolov8s_test_preds/\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cde25b75-a185-4e1b-834f-d8a0865e827c",
   "metadata": {},
   "source": [
    "## Video Inference\n",
    "\n",
    "Open and run:\n",
    "```text\n",
    "notebooks/video_inference.ipynb\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db4019d6-9fff-4d47-88ee-bd44dbb59db6",
   "metadata": {},
   "source": [
    "This notebook performs:\n",
    "- Object detection on a short welding video  \n",
    "- Saves the annotated output video  \n",
    "- Computes average inference speed (FPS)  \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ad297a90-9d25-4d5d-8237-c394f1d18352",
   "metadata": {},
   "source": [
    "**Outputs generated**\n",
    "```text\n",
    "video_out/weld_yolov8s_video/\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c3cd051-88a9-4604-9165-4afa274a6228",
   "metadata": {},
   "source": [
    "## Hardware and Framework\n",
    "\n",
    "- **Framework:** PyTorch (Ultralytics YOLOv8)  \n",
    "- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU  \n",
    "\n",
    "---\n",
    "\n",
    "## References\n",
    "\n",
    "- Kaggle Welding Defect Object Detection Dataset  \n",
    "- Ultralytics YOLOv8 Documentation  \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f1fdaf0b-8cb2-459e-89b2-db3f0254dd7f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (shilpa GPU)",
   "language": "python",
   "name": "shilpa"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
