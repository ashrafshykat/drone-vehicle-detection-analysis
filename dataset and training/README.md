***

# Vehicle Detection System: YOLOv8 & RT-DETR Pipeline

This repository contains a comprehensive pipeline for detecting **Cars, Trucks, and Bikes**. The project leverages a semi-automated labeling workflow via **Label Studio ML Backend** and evaluates two distinct state-of-the-art architectures: a CNN-based YOLOv8 and a Transformer-based RT-DETR.

## 📊 Project Statistics
- **Total Labeled Images:** 313
- **Classes:** 3 (Car, Truck, Bike)
- **Data Split:** 80% Training | 20% Validation

---

## 🚀 1. Semi-Automated Labeling Workflow
To accelerate the annotation process, we implemented a custom ML Backend. This setup reduced manual labeling time by providing pre-annotations for the 313-image dataset.

### ML Backend Architecture
- **Environment:** Conda `yolo-env1` (Python 3.12)
- **Framework:** `label-studio-ml-backend`
- **Inference Engine:** `ultralytics`
- **Connection:** Port `9090`

### Implementation Logic
1. **Model Deployment:** The `model.py` script loads weights and serves predictions via a Flask-based WSGI server.
2. **Dynamic Mapping:** Internal YOLO classes (`motorcycle`, `bicycle`) are dynamically remapped to the Label Studio XML tag `Bike` to ensure UI compatibility.
3. **Assisted Labeling:** Annotators used the "Retrieve Predictions" feature to pull model-generated boxes, performing only "quality assurance" corrections rather than manual drawing.

---

## 🏗 2. Model Pipelines

We evaluated two distinct architectures to compare traditional Convolutional Neural Networks (CNN) against modern Real-Time Transformers (DETR).

### Pipeline A: YOLOv26 (CNN-Based)
**Objective:** High-speed real-time performance and rapid convergence.
- **Model Variant:** `yolo26m.pt` (Medium)
- **Training Duration:** 60 Epochs
- **Optimization:** Focused on local feature extraction. This model converged quickly, reaching high accuracy within the first third of the training cycle.

### Pipeline B: RT-DETR (Transformer-Based)
**Objective:** Superior precision in complex scenes involving occlusion and overlapping vehicles.
- **Model Variant:** `rtdetr-l.pt` (Large)
- **Training Duration:** 150+ Epochs
- **Optimization:** Utilizes a Hybrid Encoder and Global Attention mechanisms. Unlike YOLO, this pipeline requires a significantly longer training duration to allow Transformer attention heads to stabilize spatial relationships.

---

## ⚙️ 3. Hyperparameter Configurations

The following hyperparameters were meticulously tuned to optimize performance for the 313 labeled images.

| Hyperparameter | YOLOv8 (60 Epochs) | RT-DETR (150 Epochs) | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Optimizer** | `Auto (SGD)` | `AdamW` | RT-DETR requires decoupled weight decay (AdamW) for transformer stability. |
| **Learning Rate (lr0)** | `0.01` | `0.0001` | DETR models are sensitive; a lower LR prevents gradient explosion (NaN loss). |
| **Image Size** | `640` | `640` | Standardized resolution for consistent metric comparison. |
| **Batch Size** | `16` | `16` | Optimized for Tesla T4 (16GB VRAM) memory constraints. |
| **Warmup Epochs** | `3.0` | `3.0` | Gradual learning rate increase to prevent early divergence. |
| **Weight Decay** | `0.0005` | `0.0001` | Optimized to prevent overfitting on the relatively small dataset. |

---

## 📊 4. Model Matrix & Performance Analysis

### YOLOv8 Matrix Performance
- **mAP50:** ~0.80 (80%)
- **Analysis:** Highly stable Precision/Recall balance. The model effectively identified standard vehicle silhouettes.
- **Convergence:** Fully converged by epoch 50.

### RT-DETR Matrix Performance
- **mAP50:** ~0.41 (41%)
- **Analysis:** While currently lower in mAP than YOLO, the GIoU (Generalized Intersection over Union) loss is steadily decreasing.
- **Convergence:** The model is still in the "learning" phase at epoch 150; the upward trend suggests potential to surpass YOLO with extended training (200+ epochs).

---

## 🛠 5. Installation & Execution

### Start ML Backend for Assisted Labeling
```bash
# Activate environment
conda activate yolo-env1

# Set path and run WSGI server
set PYTHONPATH=%CD%;%CD%\my_ml_backend
python my_ml_backend\_wsgi.py --port 9090
```

### Reproduce RT-DETR Training
```bash
yolo detect train \
  data=/content/data.yaml \
  model=rtdetr-l.pt \
  epochs=150 \
  imgsz=1280 \
  optimizer=AdamW \
  lr0=0.00005 \
  name=vehicle_rtdetr
```

---

## 📝 6. Data Management
Data was managed via a standard YOLO directory structure, with labels exported from Label Studio and split via `train_val_split.py`.
```text
/content/custom_data
  /train
    /images (313 images)
    /labels (.txt)
  /validation
    /images (62 images)
    /labels (.txt)
  classes.txt
```

---
**Project Maintenance:** Deployment ready for YOLOv26. Fine-tuning ongoing for RT-DETR.