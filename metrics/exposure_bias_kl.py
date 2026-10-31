"""
Proxy for exposure bias

Features are encoded (post-bottleneck)
KL div is computed by approximating globally average pooled 
(B, C, H, W) -> (B, C) TF and AR features as multivariable Gaussians
(torch.distributions.MultivariateNormal). One 'sample' from the distribution
is one of 9600 images from the test set. 

  - teacher forcing (TF):  x_t = (1 - t) * x0 + t * x1, 
    the exact interpolation from the conditional flow psi_t

  - autoregressive (AR):   x_t from Euler-integrating the model's 
    own predicted velocity from x0

NOTE: only compare same precision runs
NOTE: only compare same ODE step count tuns since euler intergration error
      has an influence and is bigger for lower ODE steps

"""


from torch.distributions import MultivariateNormal, kl_divergence
from contextlib import nullcontext
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import os

import numpy as np

from Utils.config import config
from Utils.data import get_dataloader
from Utils.loadmodel import load_models
from Utils.paths import CACHE_DIR, PLOTS_DIR
from Utils.registry import BASE, BIP_02, BIP_XTICKS, DEVICE, FROZEN_SPACE, HEATMAP_SUFFIXES, IP_CONST, IP_CUSTOM_01, SUFFIX_TO_NAME, TAG, name_of
from Utils.seeds import agg, cached, label_n, seeds_of

USE_BF16 = False  # False -> measure in fp32 (only compare same precision runs)

autocast_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if USE_BF16 and torch.cuda.is_available()
    else nullcontext()
)


class _StopForward(Exception):
    pass


class BottleneckTap:
    """Grabs the global-average-pooled (mean over H, W) output of the module during the
    model's own forward pass 
    -> (B, C)
    If stop is True, ends the forward right there (saves the decoder compute).
    
    Uses a pytorch forward hook"""
    def __init__(self, module):
        self.feat = None
        self.stop = False
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.feat = output.float().mean(dim=(-2, -1)) if output.ndim == 4 else output.float()
        if self.stop:
            raise _StopForward

    def close(self):
        self.handle.remove()


def compute_gaussian_kl(P_samples: torch.Tensor, Q_samples: torch.Tensor) -> float:
    """
    KL(P || Q) between full-covariance Gaussians fit to samples using pytorch
    Done in fp64
    """
    P, Q = P_samples.double(), Q_samples.double()
    p = MultivariateNormal(P.mean(0), covariance_matrix=torch.cov(P.T))
    q = MultivariateNormal(Q.mean(0), covariance_matrix=torch.cov(Q.T))
    return kl_divergence(p, q).item()


def truncated_features(m, tap, x, t, labels):
    """Run m only up to the tapped module and return the global avg pooled features (B, C)."""
    tap.stop = True
    try:
        m(x, t, labels)
    except _StopForward:
        pass
    return tap.feat


@torch.inference_mode()
def get_activations(model, feat_model, tap, device, images, labels, ODE_steps: int, seed: int, compute_null_floor: bool = False):
    B = images.shape[0]
    labels = labels.to(device)
    x_1 = images.to(device)

    g = torch.Generator(device=device).manual_seed(seed)
    x_0 = torch.randn(x_1.shape, generator=g, device=device, dtype=x_1.dtype)

    acts_tf, acts_autoreg = [], []
    x_t_autoreg = x_0.clone()
    dt = 1.0 / ODE_steps

    for step in range(ODE_steps):
        t = torch.full((B,), dt * step, device=device, dtype=torch.float32)
        t0 = t.clone()
        t_view = t.view(-1, 1, 1, 1)
        x_t_tf = (1.0 - t_view) * x_0 + t_view * x_1

        if not compute_null_floor:
            with autocast_context:
                feat_tf = truncated_features(feat_model, tap, x_t_tf, t, labels)
                if feat_model is model:
                    tap.stop = False                 # one full pass gives features AND velocity
                    v_autoreg = model(x_t_autoreg, t, labels)
                    feat_autoreg = tap.feat
                else:                                # frozen space: tap is on feat_model only
                    feat_autoreg = truncated_features(feat_model, tap, x_t_autoreg, t, labels)
                    v_autoreg = model(x_t_autoreg, t, labels)

            assert torch.equal(t, t0), f"The model mutated t! means: {t0.mean()} -> {t.mean()}"  
            x_t_autoreg = x_t_autoreg + dt * v_autoreg # euler update

        else:
            with autocast_context:
                # compute the null floor:
                # kl div output when the distributions are effectively the same
                # we do this by forcing both to be TF
                feat_tf = truncated_features(feat_model, tap, x_t_tf, t, labels)
                # new x_t random noise sample
                x_t_tf2 = (1.0 - t_view) * torch.randn(x_0.shape, generator=g, device=device, dtype=x_0.dtype) + t_view * x_1
                feat_autoreg = truncated_features(feat_model, tap, x_t_tf2, t, labels)
                assert torch.equal(t, t0), f"The model mutated t! means: {t0.mean()} -> {t.mean()}"  

        # save on cpu
        acts_tf.append(feat_tf.cpu())
        acts_autoreg.append(feat_autoreg.cpu())

    return torch.stack(acts_tf), torch.stack(acts_autoreg)  # (steps, B, C)

@torch.inference_mode()
def get_exposure_bias_KL_from_loader(
    model, device, loader, max_batches: int, ode_steps: int, n_bins: int,
    seed: int = 1234, frozen_sample_space_model=None, compute_null_floor: bool = False
) -> dict[int, float]:
    """If compute_null_floor=True, this computes the KL between TF and a second TF.
    Anything below this 'null' KL is noise."""
    
    assert ode_steps % n_bins == 0, f"ODE_steps ({ode_steps}) must be divisible by n_bins ({n_bins})"

    # features come from the tested model itself
    # or from a fixed reference model
    feat_model = model if frozen_sample_space_model is None else frozen_sample_space_model
    model.eval()
    feat_model.eval()
    tap = BottleneckTap(feat_model.net.bottleneck[-1])
    tf_all, ar_all = [], []
    try:
        for i, (images, labels) in enumerate(tqdm(loader, total=max_batches, desc="Processing Batches")):
            if i >= max_batches:
                break
            tf, ar = get_activations(model, feat_model, tap, device, images, labels, ode_steps, seed=seed + i, compute_null_floor=compute_null_floor)
            tf_all.append(tf)
            ar_all.append(ar)
    finally:
        tap.close()

    tf_all = torch.cat(tf_all, dim=1)  # [steps, N_total, D]
    ar_all = torch.cat(ar_all, dim=1)

    kl_per_step = [compute_gaussian_kl(tf_all[s], ar_all[s]) for s in range(ode_steps)]
    if not compute_null_floor:
        print(f"[sanity] KL at step 0 (x_tf == x_autoreg, should be about 0): {kl_per_step[0]:.3e}")

    spb = ode_steps // n_bins # steps per bin
    return {b: sum(kl_per_step[b * spb:(b + 1) * spb]) / spb for b in range(n_bins)} 




# region ==== PLOTS ====

def plot_exposure_bias_KL(
    kl_dicts: dict[str, dict[int, float]], suffix_to_name: dict, save_path: str = None, n_batches: int = 100, 
):

    bins = list(next(iter(kl_dicts.values())).keys())

    plt.figure(figsize=(10, 6))
    for model_name, kl_dict in kl_dicts.items():
        plt.bar(
            bins,
            [kl_dict[bin_idx] for bin_idx in bins],
            alpha=0.6,
            label=suffix_to_name[model_name],
        )
    plt.xlabel('Timestep Bins')
    plt.ylabel('KL Divergence')
    plt.title(f'KL-Div of Post-Bottleneck Activations under TF vs. Autoreg Sampling. Averaged across {n_batches} batches.')
    plt.xticks(bins)
    plt.grid(axis='y')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


import matplotlib.pyplot as plt
import numpy as np


def plot_exposure_bias_kl_3d(
    kl_dicts: dict[str, dict[int, float]],
    save_path: str = None,
    n_batches: int = 100,
    suffix_to_name: dict[str, str] = None,
):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Defined color palette
    colors = {
        "base_mini64": "skyblue",
        "mini_IP64_constant": "salmon",
        "mini_IP64_custom": "darkorange",
    }
    # Fallback palette for unmapped keys
    fallback_colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

    # Extract bin keys
    bins = list(next(iter(kl_dicts.values())).keys())

    # Sort models by mean KL divergence (highest in back at index 0, lowest in front)
    sorted_models = sorted(
        kl_dicts.items(),
        key=lambda item: np.mean(list(item[1].values())),
        reverse=True,
    )

    # Use suffix_to_name mapping if provided; fall back to original key if missing or None
    display_names = []
    for model_name, _ in sorted_models:
        if suffix_to_name and model_name in suffix_to_name:
            display_names.append(suffix_to_name[model_name])
        else:
            display_names.append(model_name)

    # Bar dimensions
    dx = 0.6  # Width of pillars along X (Timestep Bins)
    dy = 0.4  # Depth of pillars along Y (Models)

    for y_idx, (model_name, kl_dict) in enumerate(sorted_models):
        # Positions
        x_coords = np.array(bins) - dx / 2
        y_coords = np.full(len(bins), y_idx - dy / 2)
        z_coords = np.zeros(len(bins))  # Start pillars from ground level (z=0)

        # Height of pillars
        dz = [kl_dict[bin_idx] for bin_idx in bins]

        # Explicitly assign color for each bar in the row
        base_color = colors.get(
            model_name, fallback_colors[y_idx % len(fallback_colors)]
        )
        bar_colors = [base_color] * len(bins)

        ax.bar3d(
            x_coords,
            y_coords,
            z_coords,
            dx,
            dy,
            dz,
            color=bar_colors,
            alpha=0.8,
        )

    # Axis Labels & Ticks
    ax.set_xlabel("Timestep Bins", labelpad=10)
    ax.set_ylabel("Model", labelpad=10)
    ax.set_zlabel("KL Divergence", labelpad=10)

    ax.set_xticks(bins)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names)

    ax.set_title(
        f"KL Divergence of Post-Bottleneck Activations under Teacher Forcing vs. Autoregressive Sampling.\nAveraged across {n_batches} batches.",
        pad=20,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_exposure_bias_kl_heatmap(
    kl_dicts: dict[str, dict[int, float]],
    save_path: str = None,
    n_batches: int = 100,
    suffix_to_name: dict[str, str] = None,
    sorted_order: bool = True,
):
    # Extract bin keys
    bins = list(next(iter(kl_dicts.values())).keys())

    # Determine row order
    if sorted_order:
        # Sort models by mean KL divergence (highest mean at top/first)
        models_items = sorted(
            kl_dicts.items(),
            key=lambda item: np.mean(list(item[1].values())),
            reverse=True,
        )
    else:
        # Preserve original dictionary order
        models_items = list(kl_dicts.items())

    # Extract ordered keys and map display names
    raw_names = [model_name for model_name, _ in models_items]
    display_names = []
    for name in raw_names:
        if suffix_to_name and name in suffix_to_name:
            display_names.append(suffix_to_name[name])
        else:
            display_names.append(name)

    # Build 2D numpy array [n_models, n_bins] for the heatmap
    heatmap_data = np.array(
        [[kl_dict[bin_idx] for bin_idx in bins] for _, kl_dict in models_items]
    )
    

    # Figure setup
    plt.figure(figsize=(10, max(4, len(raw_names) * 1.2)))

    # Draw Heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=bins,
        yticklabels=display_names,
        cbar_kws={"label": "KL Divergence"},
        linewidths=0.5,
    )

    # Labels and Title
    ax.set_xlabel("Timestep Bins", labelpad=10)
    ax.set_ylabel("Model", labelpad=10)
    ax.set_title(
        f"KL Divergence of Post-Bottleneck Activations under Teacher Forcing vs. Autoregressive Sampling.\nAveraged across {n_batches} batches.",
        pad=15,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



# TODO delete
import matplotlib.pyplot as plt
import numpy as np


def plot_average_kl_gap(
    kl_dicts: dict[str, dict[int, float]],
    model_order: list[str | None],
    suffix_to_name: dict[str, str] = None,
    model_y_bars: list[str] = None,
    save_path: str = None,
    n_batches: int = 100,
):
    """Plots mean KL divergence across step bins for models in a set order with optional gaps."""

    colors = {
        "base_mini64": "skyblue",
        "mini_IP64_constant": "salmon",
        "mini_IP64_custom": "darkorange",
    }
    fallback_colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

    # Calculate and print exact mean values to terminal
    print(f"--- Exact Mean KL Divergence (Averaged across {n_batches} batches) ---")
    mean_kl_map = {}
    for model_name, kl_dict in kl_dicts.items():
        display_name = (
            suffix_to_name.get(model_name, model_name)
            if suffix_to_name
            else model_name
        )
        mean_val = np.mean(list(kl_dict.values()))
        mean_kl_map[model_name] = mean_val
        print(f"{display_name} ({model_name}): {mean_val:.6f}")
    print("-" * 65)

    plt.figure(figsize=(10, 6))

    x_ticks = []
    x_tick_labels = []

    # 1. Plot models ordered along the X-axis
    for x_idx, model_name in enumerate(model_order):
        x_ticks.append(x_idx)

        # Handle empty spots
        if model_name is None:
            x_tick_labels.append("")
            continue

        display_label = (
            suffix_to_name.get(model_name, model_name)
            if suffix_to_name
            else model_name
        )
        x_tick_labels.append(display_label)

        if model_name not in mean_kl_map:
            continue

        mean_kl = mean_kl_map[model_name]
        color = colors.get(
            model_name, fallback_colors[x_idx % len(fallback_colors)]
        )

        # Plot dot on the x-axis slot
        plt.scatter(
            x_idx,
            mean_kl,
            color=color,
            s=120,
            zorder=3,
            label=display_label,
        )

        # Drop-line to zero
        plt.vlines(
            x_idx,
            ymin=0,
            ymax=mean_kl,
            colors=color,
            linestyles="dashed",
            alpha=0.5,
        )

    # 2. Plot horizontal y-bars (parallel to X axis)
    if model_y_bars:
        for idx, bar_model in enumerate(model_y_bars):
            if bar_model not in mean_kl_map:
                continue

            bar_mean = mean_kl_map[bar_model]
            bar_label = (
                suffix_to_name.get(bar_model, bar_model)
                if suffix_to_name
                else bar_model
            )
            bar_color = colors.get(
                bar_model,
                fallback_colors[(len(model_order) + idx) % len(fallback_colors)],
            )

            # Draw horizontal line across x-axis range
            plt.axhline(
                y=bar_mean,
                color=bar_color,
                linestyle="--",
                linewidth=1.8,
                label=f"{bar_label} (Mean: {bar_mean:.4f})",
                zorder=2,
            )

    # Format Ticks and Axis
    plt.xticks(ticks=x_ticks, labels=x_tick_labels, rotation=45, ha="right")
    plt.xlabel("Models", labelpad=10)
    plt.ylabel("Mean KL Divergence")
    plt.title(
        f"Average KL Divergence across ODE Step Bins\n(Averaged across {n_batches} batches)"
    )

    plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.ylim(bottom=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# endregion


# region ==== SEEDED KL PLOTS (mean +- std over training seeds) ====
# KL_SPACE "own": KL in each model's own features. "frozen": KL in the features of FROZEN_SPACE for all models.
KL_SPACE = "own"
KL_SEED = 1709
KL_MAX_BATCHES = 100  # 100 batches of the test set
RUNS = {
    "10bins": dict(ode_steps=50, n_bins=10),
    "20bins": dict(ode_steps=100, n_bins=20),
}
OUT_DIR = PLOTS_DIR / "exposure_bias_kl"


def get_kl(suffix: str, run: str) -> tuple[dict, dict]:
    """(KL per bin, estimator floor per bin) of one training run. Cached in plots/cache."""
    cfg = RUNS[run]

    def compute():
        _, ema = load_models(suffix, device=DEVICE)
        frozen = load_models(FROZEN_SPACE, device=DEVICE)[1] if KL_SPACE == "frozen" else None
        kw = dict(max_batches=KL_MAX_BATCHES, ode_steps=cfg["ode_steps"], n_bins=cfg["n_bins"],
                  seed=KL_SEED, frozen_sample_space_model=frozen)
        loader = get_dataloader(config, split="test")
        kl = get_exposure_bias_KL_from_loader(ema, DEVICE, loader, **kw)
        floor = get_exposure_bias_KL_from_loader(ema, DEVICE, loader, compute_null_floor=True, **kw)
        return kl, floor

    return cached(CACHE_DIR / f"kl_{run}_{KL_SPACE}_{suffix}.pth", compute)


def kl_stats(suffix: str, run: str) -> dict:
    """Over training seeds: per-bin mean and std of the KL, mean floor, and mean/std of the average over bins."""
    runs = [get_kl(s, run) for s in seeds_of(suffix)]
    bins = sorted(runs[0][0])
    kl_mean, kl_std = agg([[k[b] for b in bins] for k, _ in runs])
    floor, _ = agg([[f[b] for b in bins] for _, f in runs])
    avg = agg([np.mean(list(k.values())) for k, _ in runs])
    return dict(bins=bins, mean=kl_mean, std=kl_std, floor=floor, avg=avg)


def kl_title(run: str) -> str:
    return f"KL(TF || AR), {KL_MAX_BATCHES} batches, {RUNS[run]['ode_steps']} Euler steps, space: {KL_SPACE}"


def plot_bars(stats: dict, run: str, save_path):
    """One bar per model and timestep bin, error bars = std over training seeds.
    Dashed line = estimator floor (max over the shown models)."""
    bins = next(iter(stats.values()))["bins"]
    n = len(stats)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(min(16, 9 + 0.9 * n), 6))
    for i, (suffix, st) in enumerate(stats.items()):
        xs = np.arange(len(bins)) + (i - (n - 1) / 2) * width
        ax.bar(xs, st["mean"], width=width * 0.92, label=label_n(suffix), yerr=st["std"], capsize=2.5, ecolor="black")
    floor = np.max([st["floor"] for st in stats.values()], axis=0)
    ax.plot(range(len(bins)), floor, "k--", marker="_", markersize=18, linewidth=1.2, label="est. floor (max)")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_xlabel("Timestep bins")
    ax.set_ylabel("KL divergence")
    ax.set_title(kl_title(run))
    ax.grid(axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_heatmap(stats: dict, run: str, save_path, sort: bool):
    """Cells read 'mean +- std'. The floor is the last row."""
    items = list(stats.items())
    if sort:
        items.sort(key=lambda it: -it[1]["avg"][0])
    bins = items[0][1]["bins"]
    floor = np.max([st["floor"] for _, st in items], axis=0)
    rows = [label_n(s) for s, _ in items] + ["est. floor (max)"]
    data = np.array([st["mean"] for _, st in items] + [floor])
    annot = [[f"{m:.2f}" if st["std"] is None else f"{m:.2f}\n±{st['std'][j]:.2f}" for j, m in enumerate(st["mean"])]
             for _, st in items]
    annot.append([f"{v:.2f}" for v in floor])
    plt.figure(figsize=(13, max(4, len(rows) * 1.3)))
    ax = sns.heatmap(data, annot=np.array(annot), fmt="", cmap="YlOrRd", annot_kws={"fontsize": 8},
                     xticklabels=bins, yticklabels=rows, cbar_kws={"label": "KL divergence"}, linewidths=0.5)
    ax.set_xlabel("Timestep bins")
    ax.set_title(kl_title(run))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_avg(stats: dict, order: list, ybars: list[str], run: str, save_path):
    """Mean over bins per training seed, then mean +- std over seeds. Dots for the x-axis models, lines for ybars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    for x, s in enumerate(order):
        labels.append("" if s is None else name_of(s))
        if s is None:
            continue
        m, sd = stats[s]["avg"]
        ax.errorbar(x, m, yerr=sd, fmt="o", markersize=11, capsize=5, elinewidth=1.5, zorder=3, label=label_n(s))
    for s in ybars:
        m, sd = stats[s]["avg"]
        ax.axhline(m, linestyle="--", linewidth=1.8, label=f"{label_n(s)} (mean {m:.3f})", zorder=2)
        if sd is not None:
            ax.axhspan(m - sd, m + sd, alpha=0.2, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Models")
    ax.set_ylabel("Mean KL divergence (over bins)")
    ax.set_title("Average " + kl_title(run))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bip = [m for m in BIP_XTICKS if m]
    ybars = [BASE, IP_CUSTOM_01]
    stats = {s: kl_stats(s, "10bins") for s in dict.fromkeys([BASE, IP_CONST, IP_CUSTOM_01, BIP_02, *HEATMAP_SUFFIXES, *bip])}

    plot_bars({s: stats[s] for s in [BASE, IP_CONST]}, "10bins", OUT_DIR / f"klgap_base_vs_IPconstant_{TAG}.png")
    plot_bars({BASE: kl_stats(BASE, "20bins")}, "20bins", OUT_DIR / f"klgap_base_20bins_{TAG}.png")
    plot_bars({s: stats[s] for s in [BASE, IP_CONST, IP_CUSTOM_01]}, "10bins", OUT_DIR / f"klgap_base_IPconstant_IPcustom01_{TAG}.png")
    plot_heatmap({s: stats[s] for s in HEATMAP_SUFFIXES}, "10bins", OUT_DIR / f"experiment1_klgap_unordered_heatmap_{TAG}.png", sort=False)
    plot_heatmap({s: stats[s] for s in HEATMAP_SUFFIXES}, "10bins", OUT_DIR / f"experiment1_klgap_ordered_heatmap_{TAG}.png", sort=True)
    plot_avg(stats, BIP_XTICKS, ybars, "10bins", OUT_DIR / f"avg_kl_plot_bIP_{TAG}.png")
    plot_bars({s: stats[s] for s in [BIP_02, IP_CUSTOM_01, BASE]}, "10bins", OUT_DIR / f"experiment1_only_3_{TAG}.png")

# endregion


if __name__ == "__main__":
    main()