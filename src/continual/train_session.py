import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from copy import deepcopy

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.continual.der_buffer import ModalityTaggedBuffer
from src.continual.der_loss import DERPlusPlusLoss
from src.evaluation.forgetting_eval import evaluate_model_on_loader


def get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=35):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(0.01, float(epoch) / float(max(1, warmup_epochs)))
        else:
            import math
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_unimodal_session(
    model,
    train_loader,
    val_loader,
    modality='mri',
    epochs=35,
    lr=1e-4,
    buffer=None,
    device='cpu',
    early_stopping_patience=12
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=epochs)
    der_loss_fn = DERPlusPlusLoss(alpha=0.5, beta=0.5)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_brier': [],
        'val_acc': [],
        'overfit_gap': []
    }

    best_val_acc = 0.0
    best_weights = None
    patience_counter = 0
    scaler = GradScaler('cuda')
    accumulation_steps = 4

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            x, labels, sub_ids = batch
            x, labels = x.to(device), labels.to(device)

            with autocast('cuda', enabled=(device == 'cuda')):
                current_logits = model(x)
                preds = torch.argmax(current_logits, dim=-1)
                correct_train += (preds == labels).sum().item()
                total_train += len(labels)

                # Replay
                replay_logits = None
                stored_logits = None
                replay_targets = None

                if buffer is not None and len(buffer) > 0:
                    replay_data = buffer.sample(batch_size=len(labels), modality='mri' if modality == 'spect' else 'spect', device=device)
                    if replay_data is not None and replay_data['type'] == 'unimodal':
                        replay_x = replay_data['x']
                        stored_logits = replay_data['logits']
                        replay_targets = replay_data['y']
                        replay_logits = model(replay_x)

                loss, details = der_loss_fn(
                    current_logits=current_logits,
                    current_targets=labels,
                    replay_logits=replay_logits,
                    stored_logits=stored_logits,
                    replay_targets=replay_targets
                )
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        scheduler.step()

        train_acc = correct_train / max(1, total_train)
        train_loss = running_loss / max(1, len(train_loader))

        val_metrics = evaluate_model_on_loader(model, val_loader, modality=modality, device=device)
        val_acc = val_metrics['accuracy']
        val_brier = val_metrics['brier_score']
        overfit_gap = train_acc - val_acc

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_brier'].append(val_brier)
        history['val_acc'].append(val_acc)
        history['overfit_gap'].append(overfit_gap)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"[{modality.upper()} Session] Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} Brier: {val_brier:.4f} | Gap: {overfit_gap:+.4f}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Restoring best weights with Val Acc: {best_val_acc:.4f}")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def train_multimodal_session(
    model,
    train_loader,
    val_loader,
    epochs=35,
    lr=1e-4,
    buffer=None,
    device='cpu',
    early_stopping_patience=12
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=epochs)
    der_loss_fn = DERPlusPlusLoss(alpha=0.5, beta=0.5)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_brier': [],
        'val_acc': [],
        'overfit_gap': []
    }

    best_val_acc = 0.0
    best_weights = None
    patience_counter = 0
    scaler = GradScaler('cuda')
    accumulation_steps = 4

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            if len(batch[0]) == 3:
                (mri_x, spect_x, clin_x), labels, sub_ids = batch
                mri_x, spect_x, clin_x, labels = mri_x.to(device), spect_x.to(device), clin_x.to(device), labels.to(device)
                with autocast('cuda', enabled=(device == 'cuda')):
                    current_logits = model(mri_x, spect_x, clinical_vec=clin_x)
            else:
                (mri_x, spect_x), labels, sub_ids = batch
                mri_x, spect_x, labels = mri_x.to(device), spect_x.to(device), labels.to(device)
                clin_x = None
                with autocast('cuda', enabled=(device == 'cuda')):
                    current_logits = model(mri_x, spect_x)

            preds = torch.argmax(current_logits, dim=-1)
            correct_train += (preds == labels).sum().item()
            total_train += len(labels)

            # DER++ Active Replay Distillation & Classification Loss
            replay_logits = None
            stored_logits = None
            replay_targets = None

            if buffer is not None and len(buffer) > 0:
                replay_data = buffer.sample(batch_size=len(labels), modality='multimodal', device=device)
                if replay_data is not None:
                    stored_logits = replay_data['logits']
                    replay_targets = replay_data['y']

                    if replay_data['type'] == 'multimodal':
                        rep_mri = replay_data['mri']
                        rep_spect = replay_data['spect']
                        rep_clin = replay_data['clinical']
                        if rep_clin is not None and hasattr(model, 'use_clinical') and model.use_clinical:
                            replay_logits = model(rep_mri, rep_spect, clinical_vec=rep_clin)
                        else:
                            replay_logits = model(rep_mri, rep_spect)
                    elif replay_data['type'] == 'unimodal':
                        rep_x = replay_data['x']
                        replay_logits = model(rep_x, rep_x)

            with autocast('cuda', enabled=(device == 'cuda')):
                loss, details = der_loss_fn(
                    current_logits=current_logits,
                    current_targets=labels,
                    replay_logits=replay_logits,
                    stored_logits=stored_logits,
                    replay_targets=replay_targets
                )
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        scheduler.step()

        train_acc = correct_train / max(1, total_train)
        train_loss = running_loss / max(1, len(train_loader))

        mod_eval = 'multimodal_clinical' if len(batch[0]) == 3 else 'multimodal'
        val_metrics = evaluate_model_on_loader(model, val_loader, modality=mod_eval, device=device)
        val_acc = val_metrics['accuracy']
        val_brier = val_metrics['brier_score']
        overfit_gap = train_acc - val_acc

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_brier'].append(val_brier)
        history['val_acc'].append(val_acc)
        history['overfit_gap'].append(overfit_gap)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"[FUSION Session] Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} Brier: {val_brier:.4f} | Gap: {overfit_gap:+.4f}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Restoring best weights with Val Acc: {best_val_acc:.4f}")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history
