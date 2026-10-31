"""Compare KL-to-MSE gradient magnitudes for fresh models across KL scales."""

import copy
import gc
import math

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from one_shots.calibrate_kl_penalty_scale import (
    base_config,
    kl_to_unit_gaussian,
    make_optimizer,
    make_training_loader,
)
from main.training import loss_gradient_magnitude
from Utils.UNet import UNetPipeline


KL_PENALTY_SCALES = [0.05, 0.1, 0.25, 0.5, 1.0]
N_BATCHES = 10
N_SEEDS = 10
BASE_SEED = 0
DEVICE = None  # torch device such as "cuda" or "cpu". None -> CUDA when available


def train_scale(
    scale: float,
    device: torch.device,
    seed: int,
) -> tuple[list[float], list[float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = copy.deepcopy(base_config)
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

    gradient_ratios = []
    loss_ratios = []
    total_steps = N_BATCHES
    loader_iterator = iter(loader)

    try:
        progress = tqdm(range(1, N_BATCHES + 1), desc=f"KL scale {scale}")
        for batch_index in progress:
            try:
                images, labels = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(loader)
                images, labels = next(loader_iterator)

            x_1 = images.to(device, non_blocking=config.nonblocking_gpu_transfers)
            labels = labels.to(device, non_blocking=config.nonblocking_gpu_transfers)
            batch_size = x_1.shape[0]
            timesteps = torch.sigmoid(
                torch.randn(batch_size, device=device, dtype=x_1.dtype)
            )
            noise = torch.randn_like(x_1)
            t_broadcast = timesteps.view(-1, 1, 1, 1)
            x_t = (1 - t_broadcast) * noise + t_broadcast * x_1
            target_velocity = x_1 - noise

            classids = labels.to(dtype=torch.long).clone()
            drop_mask = (
                torch.rand(batch_size, device=device)
                < config.class_conditioning_drop_prob
            )
            classids[drop_mask] = config.n_classes

            with autocast_context:
                if scale > 0:
                    prediction, bottleneck_activations = model(
                        x_t,
                        timesteps,
                        classids,
                        return_bottleneck=True,
                    )
                    kl_penalty = kl_to_unit_gaussian(bottleneck_activations)
                    scaled_kl_loss = scale * kl_penalty
                else:
                    prediction = model(x_t, timesteps, classids)
                    scaled_kl_loss = None
                mse_loss = torch.nn.functional.mse_loss(prediction, target_velocity)
                loss = mse_loss if scaled_kl_loss is None else mse_loss + scaled_kl_loss

            optimizer.zero_grad(set_to_none=True)
            if scaled_kl_loss is not None:
                mse_grad = loss_gradient_magnitude(mse_loss, model)
                kl_grad = loss_gradient_magnitude(scaled_kl_loss, model)
                gradient_ratio = (kl_grad / mse_grad.clamp_min(1e-12)).item()
                loss_ratio = (
                    scaled_kl_loss.detach() / mse_loss.detach().clamp_min(1e-12)
                ).item()
                gradient_ratios.append(gradient_ratio)
                loss_ratios.append(loss_ratio)
                progress.set_postfix(
                    gradient_ratio=f"{gradient_ratio:.5f}",
                    loss_ratio=f"{loss_ratio:.5f}",
                )
                print(
                    f"KL scale {scale}, batch {batch_index}: "
                    f"KL/MSE gradient ratio={gradient_ratio:.8f}, "
                    f"KL/MSE loss ratio={loss_ratio:.8f}"
                )

            if config.use_constant_lr:
                learning_rate = config.lr_max
            elif batch_index <= config.warmup_iters:
                learning_rate = config.lr_max * batch_index / config.warmup_iters
            else:
                progress_fraction = (batch_index - config.warmup_iters) / max(
                    total_steps - config.warmup_iters, 1
                )
                learning_rate = config.lr_min + 0.5 * (
                    config.lr_max - config.lr_min
                ) * (1 + math.cos(math.pi * progress_fraction))
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            loss.backward()
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, norm_type=2.0, error_if_nonfinite=True
                )
            optimizer.step()

        return gradient_ratios, loss_ratios
    finally:
        del optimizer
        del model
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    device = torch.device(
        DEVICE if DEVICE else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if N_BATCHES < 1:
        raise ValueError("N_BATCHES must be at least 1.")
    if not KL_PENALTY_SCALES or any(scale <= 0 for scale in KL_PENALTY_SCALES):
        raise ValueError("KL_PENALTY_SCALES must contain positive values.")

    batch_indices = range(1, N_BATCHES + 1)
    for scale in KL_PENALTY_SCALES:
        seed_ratios = [
            train_scale(scale, device, BASE_SEED + seed)[0]
            for seed in range(N_SEEDS)
        ]
        ratios = torch.tensor(seed_ratios, dtype=torch.float64)
        mean_ratio = ratios.mean(dim=0)
        std_ratio = ratios.std(dim=0, unbiased=True)
        mean_values = mean_ratio.tolist()
        std_values = std_ratio.tolist()
        line, = plt.plot(batch_indices, mean_values, label=f"KL scale={scale:g}")
        plt.fill_between(
            batch_indices,
            [mean - std for mean, std in zip(mean_values, std_values)],
            [mean + std for mean, std in zip(mean_values, std_values)],
            color=line.get_color(),
            alpha=0.2,
        )

    plt.xlabel("Batch")
    plt.ylabel("KL penalty / MSE gradient magnitude")
    plt.title("KL-to-MSE Gradient Magnitude Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
