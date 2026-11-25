# 🧠 Parkinson's Multimodal AI Diagnostic System - Setup & Execution Guide

This repository contains the complete implementation for the **Multimodal Parkinson's AI Diagnostic System** utilizing **3D Swin Transformers**, **Bidirectional Cross-Attention Fusion**, and **DER++ Continual Learning**.

The application supports real-time multi-format inference on **Structural MRI** and **DaTscan SPECT** data (`.nii`, `.nii.gz`, `.pt`, `.dcm`, `.zip` containing DICOM slice hierarchies, and 2D images).

---

## 📋 System Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: Version 3.10 or 3.11 (3.11 recommended)
- **Hardware**:
  - **CUDA GPU** (Recommended): NVIDIA GPU with CUDA 11.8 or 12.1+ for fast 3D tensor processing.
  - **CPU Only**: Supported (inference automatically falls back to CPU if CUDA is unavailable).

---

## 🛠️ Step-by-Step Installation

### Step 1: Navigate to Project Workspace
Open your terminal or PowerShell and navigate to the project directory:
```bash
cd d:/project_mri
```

### Step 2: Install PyTorch

#### For NVIDIA GPU (CUDA 11.8 / 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU Only:
```bash
pip install torch torchvision torchaudio
```

---

### Step 3: Install Required Dependencies
Install the remaining medical imaging, deep learning, and web dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure & Model Weights Setup

Ensure your project folder contains the following directories and model weights:

```
d:/project_mri/
├── checkpoints/                        # Original Baseline Model Weights (Pre-trained)
│   ├── proposed_cross_attn_fusion.pt   # Proposed Multimodal Model (3D Swin + Cross-Attn + DER++)
│   ├── mri_swin3d.pt                   # Unimodal MRI Model
│   ├── spect_swin3d.pt                 # Unimodal SPECT Model
│   ├── conv_fusion.pt                  # Conventional Concat Baseline
│   └── no_der_cross_attn.pt            # Ablation Study Model (Without DER++)
├── checkpoints_finetuned/              # DER++ Fine-Tuned Model Weights (Adapted on LONI)
│   ├── proposed_cross_attn_fusion.pt   # Fine-Tuned Multimodal Model (98.0% Accuracy)
│   ├── mri_swin3d.pt                   # Fine-Tuned Unimodal MRI Model
│   ├── conv_fusion.pt                  # Fine-Tuned Concat Fusion Model
│   └── no_der_cross_attn.pt            # Fine-Tuned Ablation Model
├── src/                                # Deep Learning Core Architecture
│   ├── models/                         # 3D Swin Transformer & Cross-Attention Modules
│   ├── continual/                      # DER++ Memory Buffer & Replay Loss
│   ├── data/                           # Data Loaders & Subject Registry
│   └── evaluation/                     # Metric Evaluation Engine
├── templates/
│   └── index.html                      # Glassmorphism Dark-Mode Web UI
├── web_app.py                          # Flask Web Server (REST API & UI Routing)
├── predict.py                          # Universal Multi-Format Inference Engine
├── gui_app.py                          # Desktop Tkinter GUI Application
├── fine_tune_loni_der.py              # DER++ Continual Fine-Tuning Script
├── eval_unseen_loni.py                 # LONI Unseen Cohort Benchmark Evaluator
├── convert_dicom_to_nii.py             # DICOM Stacking & Extraction Engine
├── PROJECT_DOCUMENTATION.md            # Complete Submission-Ready Technical Documentation
├── loni_benchmark_report.md            # Empirical Benchmark Results Table
├── loni_200_paired_subjects.txt        # Reference List of 200 Paired PPMI Subject IDs
└── requirements.txt                    # Dependency Specification
```

---

## 🚀 How to Run the Application

### Option A: Launch Web Application (Recommended)

Run the Flask web server:
```bash
python web_app.py
```

Output:
```text
Starting Parkinson's AI Web Diagnostic Application on http://127.0.0.1:5000 ...
 * Running on http://127.0.0.1:5000
```

1. Open your web browser and navigate to: **`http://127.0.0.1:5000`**
2. **Features**:
   - Select **Model Architecture Variant** (*Proposed Multimodal*, *Unimodal MRI-Only*, etc.).
   - Toggle **Weights Version**: Choose **✨ DER++ Fine-Tuned Model** (98% Accuracy on LONI) or **🏛️ Original Baseline Model**.
   - Select **Pre-loaded Scan Files** or upload **Custom Scans** (`.nii`, `.pt`, `.dcm`, `.zip` of DICOM slices, `.png`/`.jpg`).
   - Click **⚡ Execute Diagnostic Inference Engine** to view 3D axial slice renders, 3-class probability breakdown (*Control*, *Prodromal*, *PD*), confidence scores, and decision entropy.

---

### Option B: Launch Desktop GUI Application

To run the interactive Tkinter desktop GUI app:
```bash
python gui_app.py
```

---

### Option C: Run DER++ Continual Fine-Tuning on New Scans

To fine-tune all 4 architecture variants on new domain cohorts (like LONI PPMI) with DER++ active replay:
```bash
python fine_tune_loni_der.py
```
*Fine-tuned weights are saved strictly inside `checkpoints_finetuned/`, leaving baseline models untouched.*

---

### Option D: Run Full Benchmark Evaluation

To compute side-by-side benchmark evaluation metrics across all 200 paired LONI subjects:
```bash
python eval_unseen_loni.py
```

---

## 📦 Supported Input File Formats

| Format | Description | Preprocessing Workflow |
| :--- | :--- | :--- |
| **`.nii` / `.nii.gz`** | NIfTI 3D Medical Volume | Loaded via `nibabel`, intensity $Z$-score normalized, spatial interpolation to $96 \times 96 \times 96$. |
| **`.pt`** | PyTorch Saved Tensor | Loaded via `torch.load()`, normalized, interpolated to $96 \times 96 \times 96$. |
| **`.zip`** | DICOM Series Archive | Extracted automatically, sorted by instance number, stacked via `pydicom` into 3D volume. |
| **`.dcm`** | Single DICOM Slice / Stack | Stored pixel arrays resampled into 3D tensor volume automatically. |
| **`.png` / `.jpg`** | 2D Slice Image | Interpolated across depth dimension into 3D volume tensor. |

---

## ❓ Troubleshooting

1. **Port 5000 Already in Use**:
   - Edit `web_app.py` line 119 to change the port (e.g., `port=5001`).

2. **Browser UI Not Updating / Cached Version**:
   - Perform a hard refresh in your browser: `Ctrl + Shift + R` (Windows) or `Cmd + Shift + R` (Mac).

3. **DICOM Processing Errors**:
   - Ensure `pydicom` and `nibabel` are installed: `pip install pydicom nibabel`.
