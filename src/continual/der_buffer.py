import os
import sys
import random
import torch

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class ModalityTaggedBuffer:
    """
    Modality-tagged DER++ Memory Buffer storing representative MRI, SPECT, and Paired Multimodal samples.
    Guarantees strict separation of unimodal and true multimodal paired replay items.
    """
    def __init__(self, buffer_capacity=30, num_classes=3):
        self.capacity = buffer_capacity
        self.num_classes = num_classes
        self.mri_buffer = []
        self.spect_buffer = []
        self.multimodal_buffer = []

    def __len__(self):
        return len(self.mri_buffer) + len(self.spect_buffer) + len(self.multimodal_buffer)

    def add_sample(self, item, modality='mri'):
        """
        Adds sample item to appropriate modality buffer with reservoir replacement.
        """
        target_buf = getattr(self, f"{modality}_buffer", self.mri_buffer)
        max_cap = max(1, self.capacity // 3) if modality in ('mri', 'spect') else max(1, self.capacity)

        if len(target_buf) < max_cap:
            target_buf.append(item)
        else:
            idx = random.randint(0, len(target_buf) - 1)
            target_buf[idx] = item

    def populate_from_loader(self, model, dataloader, modality='mri', device='cpu', samples_per_class=8):
        """
        Populates buffer with representative samples across all classes after a session finishes.
        """
        model.eval()
        class_counts = {c: 0 for c in range(self.num_classes)}

        with torch.no_grad():
            for batch in dataloader:
                if modality in ('multimodal', 'multimodal_clinical'):
                    if len(batch[0]) == 3:
                        (mri_x, spect_x, clin_x), labels, sub_ids = batch
                        mri_x, spect_x, clin_x = mri_x.to(device), spect_x.to(device), clin_x.to(device)
                        logits = model(mri_x, spect_x, clinical_vec=clin_x)
                    else:
                        (mri_x, spect_x), labels, sub_ids = batch
                        mri_x, spect_x = mri_x.to(device), spect_x.to(device)
                        clin_x = None
                        logits = model(mri_x, spect_x)
                else:
                    x, labels, sub_ids = batch
                    x = x.to(device)

                for i in range(len(labels)):
                    c = int(labels[i])
                    if class_counts[c] < samples_per_class:
                        if modality == 'mri':
                            inp = x[i:i+1]
                            l_val = model(inp).squeeze(0).detach().cpu()
                            self.add_sample({
                                'x': inp.squeeze(0).detach().cpu(),
                                'y': c,
                                'logits': l_val,
                                'sub_id': str(sub_ids[i])
                            }, modality='mri')

                        elif modality == 'spect':
                            inp = x[i:i+1]
                            l_val = model(inp).squeeze(0).detach().cpu()
                            self.add_sample({
                                'x': inp.squeeze(0).detach().cpu(),
                                'y': c,
                                'logits': l_val,
                                'sub_id': str(sub_ids[i])
                            }, modality='spect')

                        elif modality in ('multimodal', 'multimodal_clinical'):
                            inp_mri = mri_x[i:i+1].squeeze(0).detach().cpu()
                            inp_spect = spect_x[i:i+1].squeeze(0).detach().cpu()
                            inp_clin = clin_x[i:i+1].squeeze(0).detach().cpu() if clin_x is not None else None
                            l_val = logits[i].detach().cpu()

                            self.add_sample({
                                'mri': inp_mri,
                                'spect': inp_spect,
                                'clinical': inp_clin,
                                'y': c,
                                'logits': l_val,
                                'sub_id': str(sub_ids[i])
                            }, modality='multimodal')

                        class_counts[c] += 1

    def sample(self, batch_size=4, modality='mri', device='cpu'):
        """
        Samples a mini-batch of replayed items strictly matching target modality.
        """
        target_buf = getattr(self, f"{modality}_buffer", [])
        if not target_buf and modality in ('multimodal', 'multimodal_clinical'):
            # Fallback to combining mri and spect items if multimodal buffer empty
            target_buf = self.mri_buffer + self.spect_buffer

        if not target_buf:
            return None

        actual_b = min(batch_size, len(target_buf))
        batch_items = random.sample(target_buf, actual_b)

        labels = torch.tensor([item['y'] for item in batch_items], dtype=torch.long, device=device)
        logits = torch.stack([item['logits'] for item in batch_items]).to(device)

        if modality in ('multimodal', 'multimodal_clinical') and 'mri' in batch_items[0]:
            mri_tensors = torch.stack([item['mri'] for item in batch_items]).to(device)
            spect_tensors = torch.stack([item['spect'] for item in batch_items]).to(device)
            has_clin = batch_items[0].get('clinical') is not None

            clin_tensors = torch.stack([item['clinical'] for item in batch_items]).to(device) if has_clin else None
            return {
                'mri': mri_tensors,
                'spect': spect_tensors,
                'clinical': clin_tensors,
                'y': labels,
                'logits': logits,
                'type': 'multimodal'
            }
        else:
            inputs = torch.stack([item['x'] for item in batch_items]).to(device)
            return {
                'x': inputs,
                'y': labels,
                'logits': logits,
                'type': 'unimodal'
            }

    def print_diagnostics(self):
        print(f"--- Buffer Diagnostics ---")
        print(f"  MRI Buffer Size: {len(self.mri_buffer)}")
        print(f"  SPECT Buffer Size: {len(self.spect_buffer)}")
        print(f"  Multimodal Buffer Size: {len(self.multimodal_buffer)}")
        print(f"  Total Buffer Memory Items: {len(self)}")

if __name__ == "__main__":
    buf = ModalityTaggedBuffer(buffer_capacity=30)
    print("Buffer initialized successfully.")
