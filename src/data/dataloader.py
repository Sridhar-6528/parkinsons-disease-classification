import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.subject_registry import get_subject_label_mapping
from src.data.dataset import CACHE_DIR, NeuroimagingDataset


class Random3DSpatialAugmentation:
    """
    3D Spatial Data Augmentation for Medical Neuroimaging Volumes (1, D, H, W).
    Applies random 3D spatial flips along anatomical axes, intensity scaling, and Gaussian noise.
    """
    def __init__(self, p_flip=0.5, p_noise=0.3, noise_std=0.02):
        self.p_flip = p_flip
        self.p_noise = p_noise
        self.noise_std = noise_std

    def __call__(self, tensor):
        # tensor shape: (1, D, H, W)
        if not isinstance(tensor, torch.Tensor):
            return tensor

        # 1. Random Spatial Flips along D, H, or W axes
        if torch.rand(1).item() < self.p_flip:
            for axis in [1, 2, 3]:
                if torch.rand(1).item() < 0.5:
                    tensor = torch.flip(tensor, dims=[axis])

        # 2. Random Intensity Jitter & Scaling (0.95x to 1.05x)
        if torch.rand(1).item() < self.p_noise:
            scale = 0.95 + torch.rand(1).item() * 0.10
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = (tensor * scale) + noise

        return tensor


def get_stratified_split(subject_map, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Creates stratified subject-level train/val/test splits ensuring zero subject leakage
    and preserving class distributions across all 3 splits.
    """
    subjects = np.array(sorted(list(subject_map.keys())))
    labels = np.array([subject_map[s] for s in subjects])

    # First split into train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(sss1.split(subjects, labels))

    train_subs = subjects[train_idx]
    temp_subs = subjects[temp_idx]
    temp_labels = labels[temp_idx]

    # Second split temp into val and test
    val_prop_in_temp = val_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_prop_in_temp), random_state=seed)
    val_sub_idx, test_sub_idx = next(sss2.split(temp_subs, temp_labels))

    val_subs = temp_subs[val_sub_idx]
    test_subs = temp_subs[test_sub_idx]

    return list(train_subs), list(val_subs), list(test_subs)


def create_dataloaders(modality='mri', batch_size=4, num_workers=0, seed=42, force_match=False, train_subs=None, val_subs=None, test_subs=None):
    """
    Returns train, val, and test PyTorch DataLoaders for the specified modality ('mri', 'spect', or 'multimodal').
    If train_subs, val_subs, test_subs are explicitly provided (e.g. from 5-fold CV), uses them directly.
    """
    subject_map = get_subject_label_mapping(seed=seed, modality=modality, force_match=force_match)
    
    # Filter to only subjects whose cache files actually exist
    valid_subjects = {}
    for sub, label in subject_map.items():
        mri_ok = os.path.exists(os.path.join(CACHE_DIR, f"{sub}_mri.pt"))
        spect_ok = os.path.exists(os.path.join(CACHE_DIR, f"{sub}_spect.pt"))
        if modality in ('mri', 'mri_clinical') and mri_ok:
            valid_subjects[sub] = label
        elif modality in ('spect',) and spect_ok:
            valid_subjects[sub] = label
        elif modality in ('multimodal', 'multimodal_clinical') and mri_ok and spect_ok:
            valid_subjects[sub] = label
    
    subject_map = valid_subjects

    if train_subs is None or val_subs is None or test_subs is None:
        train_subs, val_subs, test_subs = get_stratified_split(subject_map, seed=seed)
    else:
        # Filter explicitly provided subject lists to valid cached subjects
        train_subs = [s for s in train_subs if s in subject_map]
        val_subs = [s for s in val_subs if s in subject_map]
        test_subs = [s for s in test_subs if s in subject_map]

    # Verify ZERO overlap between splits
    overlap_tr_va = set(train_subs).intersection(set(val_subs))
    overlap_tr_te = set(train_subs).intersection(set(test_subs))
    overlap_va_te = set(val_subs).intersection(set(test_subs))
    assert len(overlap_tr_va) == 0, f"Error: Train and Val overlap by {len(overlap_tr_va)} subjects!"
    assert len(overlap_tr_te) == 0, f"Error: Train and Test overlap by {len(overlap_tr_te)} subjects!"
    assert len(overlap_va_te) == 0, f"Error: Val and Test overlap by {len(overlap_va_te)} subjects!"

    # 3D Data Augmentation Instance
    train_transform = Random3DSpatialAugmentation(p_flip=0.5, p_noise=0.3, noise_std=0.02)

    # Fair Clinical Features Processing (Train-only fit for Imputation & Normalization)
    clinical_dict = None
    cache_path = os.path.join(CACHE_DIR, "clinical_raw.pt")
    if os.path.exists(cache_path) and (modality in ('multimodal_clinical', 'mri_clinical')):
        raw_dict = torch.load(cache_path, weights_only=False)
        
        train_matrix = []
        for sub in train_subs:
            if sub in raw_dict:
                train_matrix.append(raw_dict[sub])
        if raw_dict is not None and train_matrix:
            train_matrix = np.array(train_matrix) # (N_train, 3)
            
            with np.errstate(all='ignore'):
                train_medians = np.nanmedian(train_matrix, axis=0)
                train_medians = np.nan_to_num(train_medians, nan=0.0)
            
            inds = np.where(np.isnan(train_matrix))
            train_matrix[inds] = np.take(train_medians, inds[1])
            
            train_means = np.mean(train_matrix, axis=0)
            train_stds = np.std(train_matrix, axis=0) + 1e-8
            
            clinical_dict = {}
            for sub in subject_map.keys():
                if sub not in raw_dict:
                    continue
                arr = np.copy(raw_dict[sub])
                for i in range(3):
                    if np.isnan(arr[i]):
                        arr[i] = train_medians[i]
                arr = (arr - train_means) / train_stds
                clinical_dict[sub] = torch.tensor(arr, dtype=torch.float32)

    # Initialize Datasets
    mod_for_ds = 'multimodal' if modality == 'multimodal_clinical' else modality
    if modality == 'mri_clinical':
        mod_for_ds = 'mri'
    
    train_ds = NeuroimagingDataset(train_subs, subject_map, modality=mod_for_ds, transform=train_transform, clinical_dict=clinical_dict)
    val_ds = NeuroimagingDataset(val_subs, subject_map, modality=mod_for_ds, clinical_dict=clinical_dict)
    test_ds = NeuroimagingDataset(test_subs, subject_map, modality=mod_for_ds, clinical_dict=clinical_dict)

    # Class-Balanced Sampling
    train_labels = [subject_map[s] for s in train_subs]
    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, (train_subs, val_subs, test_subs)


if __name__ == "__main__":
    smap = get_subject_label_mapping()
    tr, va, te, splits = create_dataloaders(modality='mri')
    print("Dataloader sanity check passed! Batches in train:", len(tr))
