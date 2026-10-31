# calculate the derivative of a model 
# x: input, y: output, h: hidden state (pre-bottleneck activations)
# derivatives we want: dy/dx (DDPM-IP), dy/dh (ours), dh/dx (CAEs) 

import torch
from tqdm import tqdm
from contextlib import nullcontext
import torch.nn as nn

from Utils.UNet import UNetPipeline, sinusoidial_embedding, ResBlock
from Utils.config import config
from Utils.data import get_dataloader
from Utils.loadmodel import load_models
from Utils.paths import CACHE_DIR, PLOTS_DIR
from Utils.registry import BASE, BIP_XTICKS, DEVICE, SUFFIX_TO_NAME, TAG, name_of
from Utils.seeds import agg, cached, label_n, seeds_of

import os
import numpy as np
import matplotlib.pyplot as plt


def encode(model: UNetPipeline, x, t, c):
    """
    Passes x through encoder.
    returns x, skips, emb
    """
    t = t.float()

    if model.t_emb_table is not None:
        t_emb = model.t_emb_table(t)
    else:
        t_emb = sinusoidial_embedding(t, model.embvector_dim)

    c_emb = model.class_emb_table(c)
    embvector = c_emb + t_emb
    emb = model.net.emb_mlp(embvector)

    x = model.net.inconv(x)
    x = x + model.net.emb_proj(emb)[:, :, None, None]

    skips = []
    for block in model.net.encoder:
        x, skip = block(x, emb)
        skips.append(skip)


    return x, skips, emb


def decode(model: UNetPipeline, h, skips, emb):
    """
    Decodes representation `h` using intermediate skip connections.
    """
    x = h

    # bottleneck 
    for block in model.net.bottleneck:
        if isinstance(block, ResBlock):
            x = block(x, emb)
        else:
            x = block(x)

    # Decode using saved skip connections
    for block, skp in zip(model.net.decoder, reversed(skips)):
        x = block(x, skp, emb)

    x = model.net.outnorm(x)
    x = model.net.outact(x)
    y = model.net.outconv(x)
    
    return y


def get_model_derivatives(config, device, model: UNetPipeline, max_batches: int, split: str = "test", seed: int | None = None) -> tuple[float, float, float]:
    model.eval()
    if seed is not None:
        torch.manual_seed(seed if seed is not None else 0)

    loader = get_dataloader(config, split)
    gen = torch.Generator(device=device).manual_seed(seed) 

    dydx_lst, dydh_lst, dhdx_lst = [], [], []

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available() else nullcontext()
    )

    for i, (images, labels) in enumerate(tqdm(loader)):
        if i == max_batches:
            break

        x_1 = images.to(device)
        labels = labels.to(device)
        B = x_1.shape[0]

        x_0 = torch.randn_like(x_1)

        t = torch.randn(size=(B,), dtype=x_1.dtype, device=device)
        t = torch.sigmoid(t)
        t_broadcast = t.view(-1, 1, 1, 1)

        x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1
        x_t = x_t.detach().requires_grad_(True)

        with autocast_context:
            # 1. dy/dx (Full Gradient)
            y = model(x_t, t, labels)
            v_y = torch.randn(y.shape, generator=gen, device=device, dtype=y.dtype)

            (dydx_grad,) = torch.autograd.grad(
                outputs=y,
                inputs=x_t,
                grad_outputs=v_y,
                retain_graph=True,
                create_graph=False
            )
            dydx_norm = torch.linalg.vector_norm(dydx_grad, dim=(1,2,3)) * torch.std(x_t.detach(), dim=(1, 2, 3))
            dydx_lst.append(dydx_norm.detach().cpu())


            # 2. dh/dx (Input to Pre-Bottleneck Gradient)
            h, skips, emb = encode(model, x_t, t, labels)

            v_h = torch.randn(h.shape, generator=gen, device=device, dtype=h.dtype)
            (dhdx_grad,) = torch.autograd.grad(
                outputs=h,
                inputs=x_t,
                grad_outputs=v_h,
                retain_graph=True,
                create_graph=False
            )
            dhdx_norm = torch.linalg.vector_norm(dhdx_grad, dim=(1,2,3)) * torch.std(x_t.detach(), dim=(1, 2, 3)) / torch.std(h.detach(), dim=(1,2,3))
            dhdx_lst.append(dhdx_norm.detach().cpu())

            # 3. dy/dh (Pre-Bottleneck to Output Gradient)
            # Detach h to make it a leaf variable for tracking grad wrt h specifically
            h_in = h.detach().requires_grad_(True)
            
            # Forward pass from h through bottleneck and decoder
            y_from_h = decode(model, h_in, skips, emb)

            # check that we get the same model outputs
            assert torch.allclose(y_from_h, y, atol=1e-5), f"y_from_h that was decoded from h diverges from y: {(y_from_h - y).abs().max()}"

            (dydh_grad,) = torch.autograd.grad(
                outputs=y_from_h,
                inputs=h_in,
                grad_outputs=v_y,
                retain_graph=False,
                create_graph=False
            )
            dydh_norm = torch.linalg.vector_norm(dydh_grad, dim=(1,2,3)) * torch.std(h_in.detach(), dim=(1, 2, 3)) # (B, )
            dydh_lst.append(dydh_norm.detach().cpu())

    # average
    mean_dydx = torch.cat(dydx_lst).mean().item()
    mean_dydh = torch.cat(dydh_lst).mean().item()
    mean_dhdx = torch.cat(dhdx_lst).mean().item()

    return mean_dydx, mean_dydh, mean_dhdx


def get_model_derivatives_seeded(config, device, model: UNetPipeline, max_batches: int, split: str = "test", n_seeds: int = 5) -> tuple[float, float, float, float, float, float]:
    dydx_outputs, dydh_outputs, dhdx_outputs = [], [], []
    for i in range(n_seeds):
        print(f"STARTING SEED {i+1}/{n_seeds}")
        dydx_, dydh_, dhdx_ = get_model_derivatives(config, device, model, max_batches, split, seed=i)
        dydx_outputs.append(dydx_)
        dydh_outputs.append(dydh_)
        dhdx_outputs.append(dhdx_)

    dydx_values = torch.tensor(dydx_outputs)
    dydh_values = torch.tensor(dydh_outputs)
    dhdx_values = torch.tensor(dhdx_outputs)

    return (
        dydx_values.mean().item(),
        dydh_values.mean().item(),
        dhdx_values.mean().item(),
        dydx_values.std().item(),
        dydh_values.std().item(),
        dhdx_values.std().item(),
    )



def get_results(modeldict: dict[str, nn.Module], N_SEEDS: int = 5, max_batches: int = 100) -> list[dict]:
    results = []
    for name, model in modeldict.items():
        dydx, dydh, dhdx, dydx_std, dydh_std, dhdx_std = get_model_derivatives_seeded(config, "cuda", model, max_batches=max_batches, split="test", n_seeds=N_SEEDS)
        results.append({
            "name": name,
            "dydx": round(dydx, 4),
            "dydh": round(dydh, 4),
            "dhdx": round(dhdx, 4),
            "dydx_std": round(dydx_std, 4),
            "dydh_std": round(dydh_std, 4),
            "dhdx_std": round(dhdx_std, 4),
        })
    return results



def derivatives_vs_model(models_xticks: list[str | None], models_ybars: list[str], device: str, save_dir: str, N_SEEDS: int, max_batches: int, suffix_to_name: dict[str]) -> None:
    """
    Plots the model derivatives against a list of models. Put None to leave an empty space.
    For example: 
        models_xticks = ['KL_005', 'KL_01', None, None, 'KL_025']
        models_ybars = ['base_mini']
    
    Creates one plot per derivative (dydx, dydh, dhdx) with one datapoint at each non-None xtick
    and one horizontal bar with the value of each model in models_ybars. Every single datapoint 
    (including y bars) has stddev bands. The plots are not shown, but simply saved to save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Helper to load only the EMA model and compute its metrics
    def get_model_metrics(model_name):
        _, ema_model = load_models(model_name, device=device)
        model_dict = {model_name: ema_model}
        res = get_results(model_dict, N_SEEDS, max_batches)[0]
        return res

    # Gather all unique model names to avoid redundant loading/computation
    all_names = [m for m in models_xticks if m is not None] + list(models_ybars)
    unique_names = list(dict.fromkeys(all_names)) # preserve order, remove duplicates
    
    metrics_cache = {}
    for name in unique_names:
        print(f"Computing metrics for: {name}")
        metrics_cache[name] = get_model_metrics(name)

    derivatives = ['dydx', 'dydh', 'dhdx']
    x_indices = list(range(len(models_xticks)))
    
    for deriv in derivatives:
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # 1. Plot xtick models as points with error bars
        x_vals = []
        y_means = []
        y_stds = []
        
        for idx, name in enumerate(models_xticks):
            if name is not None:
                res = metrics_cache[name]
                x_vals.append(idx)
                y_means.append(res[deriv])
                y_stds.append(res[f"{deriv}_std"])
                print(f"derivative {deriv}, model {suffix_to_name[name]}: {res[deriv]} +/- {res[f'{deriv}_std']}")
        
        ax.errorbar(
            x_vals, y_means, yerr=y_stds, 
            fmt='o-', capsize=5, elinewidth=1.5, markeredgewidth=1.5, 
            label='Models', color='royalblue'
        )
        
        # Maintain exact spacing including empty slots for None entries
        ax.set_xticks(x_indices)
        xtick_labels = [suffix_to_name[m] if m is not None else "" for m in models_xticks]
        ax.set_xticklabels(xtick_labels, rotation=30, ha='right')
        
        # 2. Plot y-bars (each with its own horizontal reference line and shaded band)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(models_ybars), 1)))
        for ybar_name, color in zip(models_ybars, colors):
            res = metrics_cache[ybar_name]
            mean_val = res[deriv]
            std_val = res[f"{deriv}_std"]
            print(f"derivative {deriv}, model {suffix_to_name[ybar_name]}: {res[deriv]} +/- {res[f'{deriv}_std']}")

            # suffix to name
            ybar_name = suffix_to_name[ybar_name]
            
            # Horizontal mean line
            ax.axhline(
                y=mean_val, color=color, linestyle='--', linewidth=1.5, 
                label=f'{ybar_name} (mean)'
            )
            # Shaded standard deviation band
            ax.axhspan(
                mean_val - std_val, mean_val + std_val, 
                color=color, alpha=0.2, label=f'{ybar_name} ($\\pm 1$ std)'
            )

        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel(f"Derivative ({deriv})", fontsize=12)
        ax.set_title(f"Model Comparison: {deriv} vs Models", fontsize=14, pad=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{deriv}_vs_model.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot to {save_path}")


# region ==== SEEDED RUNS (mean +- std over training seeds) ====
DERIV_MAX_BATCHES = 100
DERIV_EVAL_SEEDS = 5
DERIVS = ["dydx", "dydh", "dhdx"]
OUT_DIR = PLOTS_DIR / "modelderivative"


def seeded_metrics(name: str) -> dict:
    """{derivative: (mean, std)} over the training seeds of a model. Each training seed contributes its
    mean over the evaluation seeds. Cached in plots/cache."""
    def compute():
        res = []
        for s in seeds_of(name):
            _, ema = load_models(s, device=DEVICE)
            res.append(get_results({s: ema}, DERIV_EVAL_SEEDS, DERIV_MAX_BATCHES)[0])
        out = {}
        for d in DERIVS:
            mean, std = agg([r[d] for r in res])
            out[d] = (float(mean), 0.0 if std is None else float(std))
        return out

    return cached(CACHE_DIR / f"derivatives_{name}.pth", compute)


def derivatives_vs_model_seeded(models_xticks: list, models_ybars: list[str], save_dir) -> None:
    """Same layout as derivatives_vs_model. Error bars and bands are std over training seeds."""
    metrics = {n: seeded_metrics(n) for n in dict.fromkeys([m for m in models_xticks if m] + models_ybars)}
    names = [n for n in models_xticks if n]
    xs = [i for i, n in enumerate(models_xticks) if n]
    for deriv in DERIVS:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.errorbar(xs, [metrics[n][deriv][0] for n in names], yerr=[metrics[n][deriv][1] for n in names],
                    fmt="o-", capsize=5, elinewidth=1.5, color="royalblue", label="b-IP models")
        ax.set_xticks(range(len(models_xticks)))
        ax.set_xticklabels([name_of(n) if n else "" for n in models_xticks], rotation=30, ha="right")
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(models_ybars), 1)))
        for n, color in zip(models_ybars, colors):
            m, sd = metrics[n][deriv]
            ax.axhline(m, color=color, linestyle="--", linewidth=1.5, label=f"{label_n(n)} (mean)")
            ax.axhspan(m - sd, m + sd, color=color, alpha=0.2, label=f"{name_of(n)} ($\\pm 1$ std over training seeds)")
        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel(f"Derivative ({deriv})", fontsize=12)
        ax.set_title(f"Model Comparison: {deriv} vs Models", fontsize=14, pad=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_dir / f"{deriv}_vs_model_{TAG}.png", dpi=300)
        plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    derivatives_vs_model_seeded(BIP_XTICKS, [BASE], OUT_DIR)
    # original single-run plots (first training run, std over evaluation seeds)
    derivatives_vs_model(
        models_xticks=BIP_XTICKS,
        models_ybars=[BASE],
        device="cuda",
        save_dir=str(OUT_DIR),
        N_SEEDS=5,
        max_batches=100,
        suffix_to_name=SUFFIX_TO_NAME,
    )

# endregion


if __name__ == "__main__":
    main()