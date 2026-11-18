import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str = "models/checkpoints",
    model_name: str = "unet"
):
    """
    Saves the training checkpoint to disk.
    Always saves the latest state to {model_name}_last.pth.
    If is_best is True, also saves a copy to {model_name}_best.pth.
    Supports extended metadata (optimizer, scheduler, AMP scaler, epoch, metrics, history).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = Path(checkpoint_dir) / f"{model_name}_last.pth"
    torch.save(state, filename)

    if is_best:
        best_path = Path(checkpoint_dir) / f"{model_name}_best.pth"
        torch.save(state, best_path)
        val_dice = state.get("val_dice", state.get("best_dice", "N/A"))
        dice_str = f"{val_dice:.4f}" if isinstance(val_dice, (int, float)) else str(val_dice)
        print(f"[CHECKPOINT] Saved new best model to {best_path} (Val Dice: {dice_str})")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """
    Loads model weights and optional training states from a checkpoint.
    Maintains full backward compatibility with legacy checkpoints that only contain state_dict.
    Restores optimizer, scheduler, and AMP GradScaler states if provided and available.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 1. Restore Model State Dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint at '{checkpoint_path}' has 'state_dict' of invalid type {type(state_dict)}")
        model.load_state_dict(state_dict)
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint at '{checkpoint_path}' has 'model_state_dict' of invalid type {type(state_dict)}")
        model.load_state_dict(state_dict)
    elif isinstance(checkpoint, dict) and len(checkpoint) > 0 and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
        # Raw state dictionary mapping parameter/buffer names to PyTorch Tensors
        model.load_state_dict(checkpoint)
    elif hasattr(checkpoint, "state_dict") and callable(getattr(checkpoint, "state_dict")):
        model.load_state_dict(checkpoint.state_dict())
    else:
        sample_keys = list(checkpoint.keys())[:10] if isinstance(checkpoint, dict) else "N/A"
        raise ValueError(
            f"Unrecognized checkpoint format at '{checkpoint_path}'. "
            f"Expected a training dictionary with 'state_dict' key or a raw PyTorch model state_dict (mapping str -> torch.Tensor). "
            f"Encountered type {type(checkpoint)} with keys: {sample_keys}"
        )

    # 2. Restore Optimizer State
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"[CHECKPOINT WARNING] Could not restore optimizer state: {e}")

    # 3. Restore Scheduler State
    if scheduler is not None and isinstance(checkpoint, dict) and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as e:
            print(f"[CHECKPOINT WARNING] Could not restore scheduler state: {e}")

    # 4. Restore AMP GradScaler State
    if scaler is not None and isinstance(checkpoint, dict) and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception as e:
            print(f"[CHECKPOINT WARNING] Could not restore AMP scaler state: {e}")

    return checkpoint
