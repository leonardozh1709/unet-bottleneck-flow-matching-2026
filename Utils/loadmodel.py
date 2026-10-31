import dill
import torch
import torch.nn as nn

from Utils.UNet import UNetPipeline
from Utils.paths import OUTPUTS_DIR


def load_models(model_suffix: str, device: str | torch.device | None = None) -> tuple[nn.Module, nn.Module]:
    """
    Load the saved diffusion UNet checkpoint and config for a given model suffix.
    Returns (model, ema_model) with weights loaded and moved to the device.
    """

    output_dir = OUTPUTS_DIR / model_suffix
    config_path = output_dir / f"config_{model_suffix}.pkl"
    model_path = output_dir / f"unet_{model_suffix}.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    with open(config_path, "rb") as f:
        config = dill.load(f)

    ckpt = torch.load(model_path, map_location="cpu")
    target_device = torch.device(device) if device is not None else getattr(config, "device", torch.device("cpu"))

    model = UNetPipeline(config).to(target_device)
    ema_model = UNetPipeline(config).to(target_device)

    model.load_state_dict(ckpt["model"])
    ema_model.load_state_dict(ckpt["ema_model"])

    model.eval()
    ema_model.eval()

    print(f"Successfully loaded model and ema_model from {model_path}")
    return model, ema_model