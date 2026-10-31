"""Measure early-training loss for freshly initialized base-mini models.

This script intentionally does not load or save model checkpoints. Each seed gets
its own fresh model, is trained through epoch 10, and is discarded afterward.
"""

import copy
import gc
import math
import random
import statistics

import torch
from tqdm import tqdm

from Utils.config import config as base_config
from Utils.data import get_dataloader
from Utils.UNet import UNetPipeline


FIRST_MEASURED_EPOCH = 5
LAST_MEASURED_EPOCH = 10
SEEDS = (0, 1, 2, 3, 4)
DEVICE = None  # torch device, for example "cuda" or "cpu". None -> config.device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model: torch.nn.Module, config, device: torch.device) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or "embedding" in name.lower() or "emb_table" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr_max,
        fused=device.type == "cuda",
    )


def run_seed(seed: int, device: torch.device) -> float:
    set_seed(seed)

    config = copy.deepcopy(base_config)
    config.epochs = LAST_MEASURED_EPOCH
    config.device = device
    config.verbose = False

    loader = get_dataloader(config, split="train")
    model = UNetPipeline(config).to(device)
    model.train()
    optimizer = make_optimizer(model, config, device)

    use_autocast = device.type == "cuda" and config.training_autocasting
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_autocast
        else torch.autocast(device_type="cpu", enabled=False)
    )

    measured_losses = []
    total_steps = config.epochs * len(loader)
    step = 0

    try:
        for epoch in range(1, config.epochs + 1):
            progress = tqdm(loader, desc=f"seed {seed}, epoch {epoch}/{config.epochs}")
            for images, labels in progress:
                step += 1
                x_1 = images.to(device, non_blocking=config.nonblocking_gpu_transfers)
                labels = labels.to(device, non_blocking=config.nonblocking_gpu_transfers)
                batch_size = x_1.shape[0]

                t = torch.sigmoid(torch.randn(batch_size, device=device, dtype=x_1.dtype))
                t_broadcast = t.view(-1, 1, 1, 1)
                x_0 = torch.randn_like(x_1)
                x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1
                u_t = x_1 - x_0

                classids = labels.to(dtype=torch.long).clone()
                drop_mask = torch.rand(batch_size, device=device) < config.class_conditioning_drop_prob
                classids[drop_mask] = config.n_classes

                with autocast_context:
                    prediction = model(x_t, t, classids)
                    loss = torch.nn.functional.mse_loss(prediction, u_t)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                        norm_type=2.0,
                        error_if_nonfinite=True,
                    )
                optimizer.step()

                if FIRST_MEASURED_EPOCH <= epoch <= LAST_MEASURED_EPOCH:
                    measured_losses.append(loss.detach().float().item())
                    progress.set_postfix(loss=f"{measured_losses[-1]:.6f}")

    finally:
        expected_steps = (LAST_MEASURED_EPOCH - FIRST_MEASURED_EPOCH + 1) * len(loader)
        if len(measured_losses) != expected_steps:
            raise RuntimeError(
                f"Expected {expected_steps} measured steps, got {len(measured_losses)}."
            )

        del optimizer
        del model
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return statistics.fmean(measured_losses)


def main() -> None:
    configured_device = torch.device(DEVICE) if DEVICE else torch.device(base_config.device)
    if configured_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    seed_means = []
    for seed in SEEDS:
        mean_loss = run_seed(seed, configured_device)
        seed_means.append(mean_loss)
        print(f"seed={seed}: measured-step mean loss={mean_loss:.8f}")

    overall_mean = statistics.fmean(seed_means)
    across_seed_std = statistics.stdev(seed_means)

    print("\nFresh base-mini calibration")
    print(f"epochs measured: {FIRST_MEASURED_EPOCH}-{LAST_MEASURED_EPOCH} inclusive")
    print(f"seeds: {SEEDS}")
    print(f"average loss across seed means: {overall_mean:.8f}")
    print(f"standard deviation across seed means: {across_seed_std:.8f}")


if __name__ == "__main__":
    main()
