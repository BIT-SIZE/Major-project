import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str = "models/checkpoints",
    model_name: str = "unet"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = Path(checkpoint_dir) / f"{model_name}_last.pth"
    torch.save(state, filename)

    if is_best:
        best_path = Path(checkpoint_dir) / f"{model_name}_best.pth"
        torch.save(state, best_path)
        print(f"[CHECKPOINT] Saved new best model to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
