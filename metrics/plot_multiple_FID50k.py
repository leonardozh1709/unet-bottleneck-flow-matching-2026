import glob
import os
import random
from pathlib import Path
import torch

import matplotlib.pyplot as plt

from Utils.paths import OUTPUTS_DIR, PLOTS_DIR
from Utils.registry import BASE, BIP_01, BIP_02, BIP_XTICKS, INTERPOLATE_SUFFIXES, IP_CUSTOM_01, KL_01, KL_05, KL_1, KL_XTICKS, SUFFIX_TO_NAME, TAG
from Utils.seeds import agg, curve_stats, interpolation_error, label_n, load_fid25k, load_fid25k_raw, plot_curve, seeds_of

USE_DUMMY_DATA = False # set to False to read real fid_results files


def get_best_FID50k(suffix: str) -> float | None:
    if USE_DUMMY_DATA:
        return 15 + random.random() * 4

    file_path = f"{OUTPUTS_DIR}/{suffix}/fid_results_{suffix}.txt"
    if not Path(file_path).exists():
        print(f"{suffix} does not have a valid fid_results file.")
        return None

    for filepath in glob.glob(file_path):
        with open(filepath, "r") as file:
            for line in file:
                if "Best FID-50k:" in line:
                    return float(line.split(":")[-1].strip())
    return None


def plot_FID25k_dicts(save_dir: str):
    dict_paths = {
        "b-IP$^*$ 0.2": OUTPUTS_DIR / "bottleneck_IP_02_constant_lr" / "training_bottleneck_IP_02_constant_lr_2026-09-30_08-05-30" / "fid_dict.pt",
        "base": OUTPUTS_DIR / "base_mini64_constant_lr" / "training_base_mini64_2026-09-17_07-19-41" / "fid_dict.pt",
        "b-IP$^*$ 0.1": OUTPUTS_DIR / "bottleneck_IP_01_constant_lr" / "training_bottleneck_IP_01_constant_lr_2026-09-30_00-12-09" / "fid_dict.pt",
        "IP$^*$ 0.1": OUTPUTS_DIR / "mini_IP64_custom_constant_lr" / "training_mini_IP64_custom_2026-09-22_07-01-03" / "fid_dict.pt",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for name, path in dict_paths.items():
        fid_dict = torch.load(path, map_location="cpu", weights_only=False)
        epochs = sorted(fid_dict)
        ax.plot(epochs, [fid_dict[e] for e in epochs], marker="o", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID-25k")
    ax.legend()
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "FID25k_across_training_comparison_2.png"))
    plt.show()


def scale_aware_plot(
    models_xticks: list,
    model_to_best_FID50k: dict,
    suffix_to_name: dict | None = None,
    models_ybars: list | None = None,
    save_dir: str = "plots",
    model_to_std: dict | None = None,
    file_name: str = "FID50kplot_2.png",
) -> None:
    """models_xticks can contain None entries for empty slots.
    models_ybars: list of model keys (or list of lists) drawn as horizontal lines.
    """
    os.makedirs(save_dir, exist_ok=True)
    suffix_to_name = suffix_to_name or {}

    def get_name(key):
        return "" if key is None else suffix_to_name.get(key, key)

    def get_val(key):
        if key is None:
            return None
        if key in model_to_best_FID50k:
            return model_to_best_FID50k[key]
        return model_to_best_FID50k.get(get_name(key))

    fig, ax = plt.subplots(figsize=(9, 6))
    all_vals = []

    # 1. Dots (error bars = std over training seeds, if given)
    x_vals, y_vals, y_err = [], [], []
    for i, key in enumerate(models_xticks):
        val = get_val(key)
        if val is not None:
            x_vals.append(i)
            y_vals.append(val)
            y_err.append((model_to_std or {}).get(key) or 0)
    all_vals += y_vals
    ax.errorbar(x_vals, y_vals, yerr=y_err, fmt="o", color="blue", markersize=8, capsize=5, zorder=3, label="other models")

    # 2. Horizontal bars (lines spanning the full plot width)
    if models_ybars:
        if not isinstance(models_ybars[0], (list, tuple)):
            series_list = [[m] for m in models_ybars]
        else:
            series_list = models_ybars
        colors = ["green", "red", "purple", "orange", "cyan"]
        for s_idx, group in enumerate(series_list):
            color = colors[s_idx % len(colors)]
            for key in group:
                val = get_val(key)
                if val is None:
                    print(f"Warning: no value for horizontal bar '{key}', skipping.")
                    continue
                all_vals.append(val)
                ax.axhline(
                    y=val, color=color, linestyle="--", linewidth=2,
                    alpha=0.9, zorder=2, label=str(get_name(key))[:25],
                )
                sd = (model_to_std or {}).get(key)
                if sd:
                    ax.axhspan(val - sd, val + sd, color=color, alpha=0.2, zorder=1)

    ax.set_xticks(list(range(len(models_xticks))))
    ax.set_xticklabels([get_name(m) for m in models_xticks], rotation=30, ha="right")

    # make sure every dot and bar is inside the visible y-range
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max((hi - lo) * 0.15, 1.0)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("FID-50k", fontsize=12)
    ax.set_title("FID-50k across models", fontsize=14, pad=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


# region ==== SEEDED RUNS (mean +- std over training seeds) ====
OUT_DIR = PLOTS_DIR / "FID"


def fid50k_mean_std(suffix: str) -> tuple[float | None, float | None]:
    """Best FID-50k mean and std over the training seeds of a model."""
    vals = [v for v in (get_best_FID50k(s) for s in seeds_of(suffix)) if v is not None]
    if not vals:
        return None, None
    mean, std = agg(vals)
    return float(mean), (None if std is None else float(std))


def plot_fid50k_seeded(models_xticks: list, models_ybars: list[str], file_name: str):
    means, stds = {}, {}
    for key in [m for m in models_xticks if m] + models_ybars:
        means[key], stds[key] = fid50k_mean_std(key)
    scale_aware_plot(
        models_xticks=models_xticks,
        model_to_best_FID50k={k: v for k, v in means.items() if v is not None},
        suffix_to_name=SUFFIX_TO_NAME,
        models_ybars=models_ybars,
        save_dir=str(OUT_DIR),
        model_to_std=stds,
        file_name=file_name,
    )


def plot_fid25k_seeded(models: list[str], save_path):
    """FID-25k curves, mean +- std over training seeds. Hollow markers = interpolated epochs."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in models:
        loaded = [load_fid25k(s) for s in seeds_of(m)]
        epochs, mean, std = curve_stats([d for d, _ in loaded])
        plot_curve(ax, epochs, mean, std, label=label_n(m), hollow=set().union(*[h for _, h in loaded]))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID-25k")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def write_interpolation_error():
    for s in sorted(INTERPOLATE_SUFFIXES):
        text = interpolation_error(load_fid25k_raw(s))
        (OUT_DIR / f"interpolation_error_fid25k_{s}.txt").write_text(text + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_interpolation_error()
    plot_fid25k_seeded([BASE], OUT_DIR / f"FID25k_across_training_base_{TAG}.png")
    plot_fid25k_seeded([BIP_02, BASE, BIP_01, IP_CUSTOM_01], OUT_DIR / f"FID25k_across_training_comparison_{TAG}.png")
    plot_fid25k_seeded([KL_05, BASE, KL_01, KL_1], OUT_DIR / f"FID25k_across_training_comparison_KL_{TAG}.png")
    plot_fid50k_seeded(BIP_XTICKS, [BASE, IP_CUSTOM_01], f"FID50kplot_{TAG}.png")
    plot_fid50k_seeded(KL_XTICKS, [BASE], f"FID50kplot_KL_{TAG}.png")

    # original single-run plots (first training run of each model)
    plot_FID25k_dicts(save_dir=str(OUT_DIR))
    model_to_best_FID50k = {name: get_best_FID50k(name) for name in [s for s in KL_XTICKS if s] + [BASE]}
    scale_aware_plot(
        models_xticks=KL_XTICKS,
        model_to_best_FID50k=model_to_best_FID50k,
        suffix_to_name=SUFFIX_TO_NAME,
        models_ybars=[BASE],
        save_dir=str(OUT_DIR),
    )

# endregion


if __name__ == "__main__":
    main()