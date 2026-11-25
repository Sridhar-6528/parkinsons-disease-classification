"""
================================================================================
 PARKINSON'S DISEASE DIAGNOSTIC DESKTOP GUI APPLICATION
 3D Swin Transformer + Bidirectional Cross-Attention + DER++ Continual Learning
================================================================================

Features:
  1. Interactive File Browsing for 3D MRI & 3D SPECT (.nii / .nii.gz / .pt)
  2. Model Variant Selector (Proposed DER++, Fair Clinical, Unimodal MRI/SPECT, Concat)
  3. Interactive Clinical Tabular Inputs (UPDRS III, MoCA, Age, Gender)
  4. Optional Ground-Truth Label Selector for Live Accuracy/Precision/Recall verification
  5. Real-Time Probability Bar Chart & Confidence Score Dashboard

Run locally with:
  python gui_app.py
================================================================================
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predict import load_input_volume, preprocess_clinical_vector, CLASS_NAMES, CHECKPOINT_DIR
from src.models.swin3d_encoder import Swin3DEncoder
from src.models.cross_attention_fusion import DualSwinCrossAttentionNetwork
from src.models.baselines import ConventionalFusionNetwork

MODEL_VARIANTS = {
    "1. Proposed Model (3D Swin + Cross-Attn + DER++)": {
        "type": "proposed",
        "checkpoint": "proposed_cross_attn_fusion.pt",
        "mri_req": True,
        "spect_req": True,
        "clin_req": False
    },
    "2. Imaging + Fair Clinical Fusion (3D Swin + Gate)": {
        "type": "clinical",
        "checkpoint": "fair_clinical_fusion.pt",
        "mri_req": True,
        "spect_req": True,
        "clin_req": True
    },
    "3. Unimodal 3D Swin MRI-only": {
        "type": "mri_only",
        "checkpoint": "mri_swin3d.pt",
        "mri_req": True,
        "spect_req": False,
        "clin_req": False
    },
    "4. Unimodal 3D Swin SPECT-only": {
        "type": "spect_only",
        "checkpoint": "spect_swin3d.pt",
        "mri_req": False,
        "spect_req": True,
        "clin_req": False
    },
    "5. Conventional Fusion (Concat Baseline)": {
        "type": "concat",
        "checkpoint": "conv_fusion.pt",
        "mri_req": True,
        "spect_req": True,
        "clin_req": False
    },
    "6. Cross-Attention (No DER++ Baseline)": {
        "type": "no_der",
        "checkpoint": "no_der_cross_attn.pt",
        "mri_req": True,
        "spect_req": True,
        "clin_req": False
    }
}


class ParkinsonDiagnosticGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkinson's Disease Diagnostic Inference Engine (3D Swin + DER++)")
        self.root.geometry("1100x800")
        self.root.minsize(1000, 750)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.mri_path_var = tk.StringVar(value="")
        self.spect_path_var = tk.StringVar(value="")
        self.selected_model_var = tk.StringVar(value="1. Proposed Model (3D Swin + Cross-Attn + DER++)")

        self.updrs_var = tk.DoubleVar(value=18.0)
        self.moca_var = tk.DoubleVar(value=26.0)
        self.age_var = tk.DoubleVar(value=65.0)
        self.gender_var = tk.StringVar(value="Male (1)")
        self.ground_truth_var = tk.StringVar(value="Unknown (Live Inference)")

        self.loaded_models = {}

        self.setup_ui()

    def setup_ui(self):
        # Header Banner
        banner_frame = tk.Frame(self.root, bg="#1E293B")
        banner_frame.pack(fill=tk.X)

        title_lbl = tk.Label(banner_frame, text="PARKINSON'S DISEASE NEUROIMAGING DIAGNOSTIC SUITE",
                             font=("Segoe UI", 16, "bold"), fg="#F8FAFC", bg="#1E293B")
        title_lbl.pack(side=tk.LEFT, padx=15, pady=8)

        device_lbl = tk.Label(banner_frame, text=f"COMPUTE DEVICE: {self.device.upper()}",
                              font=("Segoe UI", 10, "bold"), fg="#38BDF8", bg="#1E293B")
        device_lbl.pack(side=tk.RIGHT, padx=15, pady=8)

        # Main Layout: Left Panel (Controls) & Right Panel (Results & Graphs)
        main_container = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Panel (Controls & Files)
        left_frame = tk.Frame(main_container, bg="#F1F5F9", bd=1, relief=tk.SOLID)
        main_container.add(left_frame, width=480)

        # Right Panel (Output & Graphs)
        right_frame = tk.Frame(main_container, bg="#FFFFFF", bd=1, relief=tk.SOLID)
        main_container.add(right_frame, width=600)

        # ------------------- LEFT PANEL CONTENTS -------------------
        # Section 1: Model Selection
        sec1_lbl = tk.Label(left_frame, text="1. Select Trained Model Architecture", font=("Segoe UI", 11, "bold"), bg="#F1F5F9", fg="#0F172A")
        sec1_lbl.pack(anchor=tk.W, padx=12, pady=(12, 4))

        model_dropdown = ttk.Combobox(left_frame, textvariable=self.selected_model_var, values=list(MODEL_VARIANTS.keys()), state="readonly", font=("Segoe UI", 10))
        model_dropdown.pack(fill=tk.X, padx=12, pady=4)

        # Section 2: Input Image File Selection
        sec2_lbl = tk.Label(left_frame, text="2. Input 3D Volume Files (.nii, .nii.gz, .pt)", font=("Segoe UI", 11, "bold"), bg="#F1F5F9", fg="#0F172A")
        sec2_lbl.pack(anchor=tk.W, padx=12, pady=(16, 4))

        # MRI File Picker
        mri_frame = tk.Frame(left_frame, bg="#F1F5F9")
        mri_frame.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(mri_frame, text="3D MRI Volume:", width=15, anchor=tk.W, bg="#F1F5F9", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Entry(mri_frame, textvariable=self.mri_path_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(mri_frame, text="Browse...", command=self.browse_mri, bg="#0284C7", fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT)

        # SPECT File Picker
        spect_frame = tk.Frame(left_frame, bg="#F1F5F9")
        spect_frame.pack(fill=tk.X, padx=12, pady=4)
        tk.Label(spect_frame, text="3D SPECT Volume:", width=15, anchor=tk.W, bg="#F1F5F9", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        tk.Entry(spect_frame, textvariable=self.spect_path_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(spect_frame, text="Browse...", command=self.browse_spect, bg="#0284C7", fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT)

        # Section 3: Clinical Tabular Variables
        sec3_lbl = tk.Label(left_frame, text="3. Clinical Tabular Features (Optional / Clinical Fusion)", font=("Segoe UI", 11, "bold"), bg="#F1F5F9", fg="#0F172A")
        sec3_lbl.pack(anchor=tk.W, padx=12, pady=(16, 4))

        clin_grid = tk.Frame(left_frame, bg="#F1F5F9")
        clin_grid.pack(fill=tk.X, padx=12, pady=4)

        tk.Label(clin_grid, text="UPDRS III Motor (0-50):", bg="#F1F5F9", font=("Segoe UI", 9)).grid(row=0, column=0, sticky=tk.W, pady=4)
        tk.Scale(clin_grid, from_=0, to=50, orient=tk.HORIZONTAL, variable=self.updrs_var, bg="#F1F5F9").grid(row=0, column=1, sticky=tk.EW, padx=8)

        tk.Label(clin_grid, text="MoCA Score (0-30):", bg="#F1F5F9", font=("Segoe UI", 9)).grid(row=1, column=0, sticky=tk.W, pady=4)
        tk.Scale(clin_grid, from_=0, to=30, orient=tk.HORIZONTAL, variable=self.moca_var, bg="#F1F5F9").grid(row=1, column=1, sticky=tk.EW, padx=8)

        tk.Label(clin_grid, text="Age (years):", bg="#F1F5F9", font=("Segoe UI", 9)).grid(row=2, column=0, sticky=tk.W, pady=4)
        tk.Scale(clin_grid, from_=30, to=90, orient=tk.HORIZONTAL, variable=self.age_var, bg="#F1F5F9").grid(row=2, column=1, sticky=tk.EW, padx=8)

        tk.Label(clin_grid, text="Gender:", bg="#F1F5F9", font=("Segoe UI", 9)).grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(clin_grid, textvariable=self.gender_var, values=["Male (1)", "Female (0)"], state="readonly", width=12).grid(row=3, column=1, sticky=tk.W, padx=8)

        # Section 4: Optional Ground Truth for Verification
        sec4_lbl = tk.Label(left_frame, text="4. Ground Truth Label (For Live Verification)", font=("Segoe UI", 11, "bold"), bg="#F1F5F9", fg="#0F172A")
        sec4_lbl.pack(anchor=tk.W, padx=12, pady=(16, 4))

        gt_dropdown = ttk.Combobox(left_frame, textvariable=self.ground_truth_var,
                                   values=["Unknown (Live Inference)", "Control (Healthy)", "Prodromal Parkinson's", "Parkinson's Disease (PD)"],
                                   state="readonly", font=("Segoe UI", 10))
        gt_dropdown.pack(fill=tk.X, padx=12, pady=4)

        # Predict Action Button
        predict_btn = tk.Button(left_frame, text="RUN DIAGNOSTIC INFERENCE", command=self.run_inference,
                                bg="#16A34A", fg="white", font=("Segoe UI", 12, "bold"), pady=10)
        predict_btn.pack(fill=tk.X, padx=12, pady=(24, 12))

        # ------------------- RIGHT PANEL CONTENTS -------------------
        res_header = tk.Label(right_frame, text="DIAGNOSTIC OUTPUT & PROBABILITY DASHBOARD",
                              font=("Segoe UI", 12, "bold"), bg="#FFFFFF", fg="#0F172A")
        res_header.pack(anchor=tk.W, padx=15, pady=(15, 5))

        # Output Cards Container
        self.card_frame = tk.Frame(right_frame, bg="#FFFFFF")
        self.card_frame.pack(fill=tk.X, padx=15, pady=5)

        self.diagnosis_card = tk.Label(self.card_frame, text="SELECT IMAGE FILES & CLICK PREDICT",
                                       font=("Segoe UI", 14, "bold"), bg="#E2E8F0", fg="#475569", pady=12)
        self.diagnosis_card.pack(fill=tk.X, pady=4)

        self.metrics_label = tk.Label(self.card_frame, text="Confidence: -- % | Decision Uncertainty: -- | Live Acc: --",
                                      font=("Segoe UI", 10), bg="#FFFFFF", fg="#64748B")
        self.metrics_label.pack(anchor=tk.W, pady=4)

        # Matplotlib Probability Bar Plot Canvas
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor('#FFFFFF')
        self.ax.set_facecolor('#F8FAFC')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        self.plot_empty_probabilities()

    def browse_mri(self):
        file_path = filedialog.askopenfilename(
            title="Select 3D MRI Volume File",
            filetypes=[("Medical NIfTI Volumes", "*.nii *.nii.gz"), ("PyTorch Tensors", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            self.mri_path_var.set(file_path)

    def browse_spect(self):
        file_path = filedialog.askopenfilename(
            title="Select 3D SPECT Volume File",
            filetypes=[("Medical NIfTI Volumes", "*.nii *.nii.gz"), ("PyTorch Tensors", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            self.spect_path_var.set(file_path)

    def plot_empty_probabilities(self):
        self.ax.clear()
        bars = self.ax.barh(CLASS_NAMES, [0.33, 0.33, 0.33], color='#CBD5E1')
        self.ax.set_xlim(0, 1.0)
        self.ax.set_xlabel('Probability Score', fontsize=10, fontweight='bold', color='#475569')
        self.ax.set_title('Class Probability Distribution', fontsize=11, fontweight='bold', color='#0F172A')
        self.fig.tight_layout()
        self.canvas.draw()

    def run_inference(self):
        model_name = self.selected_model_var.get()
        cfg = MODEL_VARIANTS[model_name]

        mri_path = self.mri_path_var.get().strip()
        spect_path = self.spect_path_var.get().strip()

        # Validation
        if cfg["mri_req"] and not mri_path:
            messagebox.showerror("Missing Input File", f"The selected model '{model_name}' requires a 3D MRI Volume file!")
            return

        if cfg["spect_req"] and not spect_path:
            messagebox.showerror("Missing Input File", f"The selected model '{model_name}' requires a 3D SPECT Volume file!")
            return

        ckpt_file = os.path.join(CHECKPOINT_DIR, cfg["checkpoint"])
        if not os.path.exists(ckpt_file):
            messagebox.showerror("Checkpoint Not Found", f"Model checkpoint file '{cfg['checkpoint']}' not found in checkpoints directory!")
            return

        try:
            # 1. Load Input Tensors
            mri_vol = load_input_volume(mri_path).to(self.device) if cfg["mri_req"] else None
            spect_vol = load_input_volume(spect_path).to(self.device) if cfg["spect_req"] else None

            g_val = 1.0 if "Male" in self.gender_var.get() else 0.0
            clin_vec = preprocess_clinical_vector(self.updrs_var.get(), self.moca_var.get(), self.age_var.get(), g_val).to(self.device)

            # 2. Instantiate Architecture & Load Checkpoint Weights
            mtype = cfg["type"]
            if mtype == "proposed":
                mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
                spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
                model = DualSwinCrossAttentionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc, use_clinical=False).to(self.device)
            elif mtype == "clinical":
                mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
                spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
                model = DualSwinCrossAttentionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc, use_clinical=True).to(self.device)
            elif mtype == "concat":
                mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
                spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
                model = ConventionalFusionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc, use_clinical=False).to(self.device)
            elif mtype == "no_der":
                mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
                spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
                model = DualSwinCrossAttentionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc, use_clinical=False).to(self.device)
            elif mtype == "mri_only":
                model = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512).to(self.device)
            elif mtype == "spect_only":
                model = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512).to(self.device)

            state_dict = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()

            # 3. Model Forward Pass
            with torch.no_grad():
                if mtype in ["proposed", "no_der", "concat"]:
                    logits = model(mri_vol, spect_vol)
                elif mtype == "clinical":
                    logits = model(mri_vol, spect_vol, clinical_vec=clin_vec)
                elif mtype == "mri_only":
                    logits = model(mri_vol)
                elif mtype == "spect_only":
                    logits = model(spect_vol)

            # 4. Probabilities & Metrics Calculation
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            predicted_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(probs[predicted_idx]) * 100.0

            # Calculate Decision Uncertainty (Entropy H)
            entropy = -float(np.sum(probs * np.log2(probs + 1e-8)))

            # 5. Live Ground-Truth Metric Calculation (If Ground Truth Specified)
            gt_text = self.ground_truth_var.get()
            gt_idx = None
            if "Control" in gt_text:
                gt_idx = 0
            elif "Prodromal" in gt_text:
                gt_idx = 1
            elif "Parkinson" in gt_text:
                gt_idx = 2

            live_acc_text = "Live Acc: N/A (Ground Truth Unknown)"
            live_prec_text = "Precision: N/A"
            live_rec_text = "Recall: N/A"

            if gt_idx is not None:
                is_correct = (predicted_idx == gt_idx)
                live_acc = 1.0 if is_correct else 0.0
                live_acc_text = f"Live Accuracy: {live_acc * 100:.0f}% ({'MATCH' if is_correct else 'MISMATCH'})"

                # Per-class live precision & recall scores
                live_prec = probs[gt_idx] # Model probability assigned to true target
                live_rec = 1.0 if is_correct else 0.0
                live_prec_text = f"Target Precision Prob: {live_prec * 100:.1f}%"
                live_rec_text = f"Target Recall: {live_rec * 100:.0f}%"

            # 6. Update GUI Output Cards
            bg_colors = ["#DCFCE7", "#FEF08A", "#FEE2E2"]
            fg_colors = ["#15803D", "#854D0E", "#B91C1C"]

            self.diagnosis_card.config(
                text=f"PREDICTED DIAGNOSIS: {predicted_class.upper()}",
                bg=bg_colors[predicted_idx],
                fg=fg_colors[predicted_idx]
            )

            metrics_str = f"Confidence: {confidence:.2f}% | Decision Entropy: {entropy:.3f} | {live_acc_text} | {live_prec_text} | {live_rec_text}"
            self.metrics_label.config(text=metrics_str)

            # 7. Update Probability Chart
            self.ax.clear()
            colors = ['#22C55E' if i == 0 else ('#EAB308' if i == 1 else '#EF4444') for i in range(3)]
            bars = self.ax.barh(CLASS_NAMES, probs, color=colors)

            for bar in bars:
                w = bar.get_width()
                self.ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w*100:.2f}%",
                             va='center', ha='left', fontweight='bold', color='#1E293B')

            self.ax.set_xlim(0, 1.15)
            self.ax.set_xlabel('Confidence Probability Score', fontsize=10, fontweight='bold', color='#475569')
            self.ax.set_title(f'Prediction: {predicted_class} ({confidence:.1f}% Confidence)', fontsize=11, fontweight='bold', color='#0F172A')
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Inference Error", f"An error occurred during prediction:\n{str(e)}")


def main():
    root = tk.Tk()
    app = ParkinsonDiagnosticGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
