"""Train a fresh model and measure post-bottleneck KL for penalty calibration.

This script does not load or save model checkpoints or write training outputs.
The CIFAR-10 dataset must already exist under this project's data directory.
"""

import copy
import gc
import math
import statistics

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from Utils.config import config as base_config
from Utils.UNet import UNetPipeline
from Utils.paths import DATA_DIR


EPOCHS = 10
FIRST_MEASURED_EPOCH = 5
N_SEEDS = 5
DEVICE = None  # torch device such as "cuda" or "cpu". None -> CUDA when available


def make_training_loader(config, device: torch.device) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        prefetch_factor=4,
    )


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


def kl_to_unit_gaussian(activations: torch.Tensor) -> torch.Tensor:
    values = activations.float().transpose(0, 1).flatten(1)
    mean = values.mean(dim=1)
    variance = values.var(dim=1, unbiased=False).clamp_min(1e-6)
    return 0.5 * (mean.square() + variance - variance.log() - 1).mean()


def run_calibration(device: torch.device, seed: int) -> float:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = copy.deepcopy(base_config)
    config.epochs = EPOCHS
    config.device = device
    config.verbose = False

    if config.use_cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(
        "medium" if config.extra_low_matmul_precision else "high"
    )

    loader = make_training_loader(config, device)
    model = UNetPipeline(config).to(device)
    model.train()
    optimizer = make_optimizer(model, config, device)

    use_autocast = device.type == "cuda" and config.training_autocasting
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_autocast
        else torch.autocast(device_type="cpu", enabled=False)
    )

    measured_epoch_means = []
    total_steps = EPOCHS * len(loader)
    step = 0

    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_kl_total = 0.0
            epoch_batch_count = 0
            progress = tqdm(loader, desc=f"epoch {epoch}/{EPOCHS}")

            for images, labels in progress:
                step += 1
                x_1 = images.to(
                    device,
                    non_blocking=config.nonblocking_gpu_transfers,
                )
                labels = labels.to(
                    device,
                    non_blocking=config.nonblocking_gpu_transfers,
                )
                batch_size = x_1.shape[0]

                timesteps = torch.sigmoid(
                    torch.randn(batch_size, device=device, dtype=x_1.dtype)
                )
                t_broadcast = timesteps.view(-1, 1, 1, 1)
                noise = torch.randn_like(x_1)
                x_t = (1 - t_broadcast) * noise + t_broadcast * x_1
                target_velocity = x_1 - noise

                classids = labels.to(dtype=torch.long).clone()
                drop_mask = (
                    torch.rand(batch_size, device=device)
                    < config.class_conditioning_drop_prob
                )
                classids[drop_mask] = config.n_classes

                with autocast_context:
                    prediction, bottleneck_activations = model(
                        x_t,
                        timesteps,
                        classids,
                        return_bottleneck=True,
                    )
                    loss = torch.nn.functional.mse_loss(prediction, target_velocity)

                kl_value = kl_to_unit_gaussian(bottleneck_activations)
                epoch_kl_total += kl_value.detach().item()
                epoch_batch_count += 1

                if config.use_constant_lr:
                    learning_rate = config.lr_max
                elif step <= config.warmup_iters:
                    learning_rate = config.lr_max * step / config.warmup_iters
                else:
                    progress_fraction = (step - config.warmup_iters) / (
                        total_steps - config.warmup_iters
                    )
                    learning_rate = config.lr_min + 0.5 * (
                        config.lr_max - config.lr_min
                    ) * (1 + math.cos(math.pi * progress_fraction))

                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate

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

                progress.set_postfix(loss=f"{loss.detach().item():.5f}")

            epoch_kl_mean = epoch_kl_total / epoch_batch_count
            if epoch >= FIRST_MEASURED_EPOCH:
                measured_epoch_means.append(epoch_kl_mean)
                print(f"Epoch {epoch} mean post-bottleneck KL: {epoch_kl_mean:.8f}")

        return statistics.fmean(measured_epoch_means)
    finally:
        del optimizer
        del model
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    if DEVICE:
        device = torch.device(DEVICE)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    seed_averages = []
    for seed in range(N_SEEDS):
        average_kl = run_calibration(device, seed)
        seed_averages.append(average_kl)
        print(
            f"Seed {seed + 1}/{N_SEEDS} (seed={seed}) mean KL, "
            f"epochs {FIRST_MEASURED_EPOCH}-{EPOCHS}: {average_kl:.8f}"
        )

    mean_kl = statistics.fmean(seed_averages)
    std_kl = statistics.stdev(seed_averages)
    print(
        f"Across {N_SEEDS} seeds, post-bottleneck KL, "
        f"epochs {FIRST_MEASURED_EPOCH}-{EPOCHS}: "
        f"{mean_kl:.8f} +/- {std_kl:.8f} (mean +/- sample standard deviation)"
    )


if __name__ == "__main__":
    main()