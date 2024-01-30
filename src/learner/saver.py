"""
Author:
    Ahmed Telili, Research Engineer @ Technology Innovation Institute (TII)

Description:
    This script, developed and maintained by Ahmed Telili, is part of a challenge focused on super-resolution sponsored by TII. For any queries or contributions regarding this script, please contact ahmed.telili@tii.ae.
"""

import torch
from pathlib import Path

class Saver:
    def __init__(self, model, optimizer, save_dir=Path('checkpoints'), best_val_metric=float('inf')):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.best_val_metric = best_val_metric
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def save_checkpoint(self, step, val_metric, filename='checkpoint.pth.tar'):
        """Saves the model and optimizer states."""
        self.step = step
        save_path = self.save_dir / f'step_{step}_{filename}'
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metric': val_metric
        }, save_path)
        print(f"Checkpoint saved at step {step} to {save_path}")

    def load_checkpoint(self, filepath):
        """Loads model and optimizer states from a checkpoint."""
        checkpoint = torch.load(filepath)
        self.step = checkpoint.get('step', 0)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")
        

    def update_and_save(self, step, val_metric, filename='checkpoint.pth.tar'):
        """Updates the best metric and saves the model if the current metric is better."""
        if val_metric < self.best_val_metric:
            self.best_val_metric = val_metric
            self.save_checkpoint(step, val_metric, filename)
            print(f"New best model saved with metric {val_metric}")
