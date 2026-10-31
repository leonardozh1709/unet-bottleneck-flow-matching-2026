from pathlib import Path

import matplotlib.pyplot as plt
import torch


from Utils.paths import OUTPUTS_DIR, PLOTS_DIR
from Utils.registry import BASE, IP_CONST, IP_CUSTOM_01, IP_CUSTOM_02, INTERPOLATE_SUFFIXES, SUFFIX_TO_NAME, TAG
from Utils.seeds import curve_stats, interpolate_to_80_epoch_grid, label_n, plot_curve, seeds_of

DICT_FILENAME = "pathstraightness_dict.pt"
PLOT_PATH = PLOTS_DIR / "pathstraightness" / "pathstraightness_all_models.png"


def load_pathstraightness_dicts(outputs_dir: Path) -> list[tuple[str, dict]]:
    """Load every saved path-straightness dictionary below the outputs directory."""
    loaded = []
    for dict_path in sorted(outputs_dir.rglob(DICT_FILENAME)):
        values = torch.load(dict_path, map_location="cpu", weights_only=True)
        if not isinstance(values, dict) or not values:
            print(f"Skipping invalid or empty dictionary: {dict_path}")
            continue

        label = dict_path.parent.parent.name
        loaded.append((label, values))
    return loaded


def plot_pathstraightness(suffix_list: list[str], suffix_to_name: dict, outputs_dir: Path = OUTPUTS_DIR, plot_path: Path = PLOT_PATH) -> None:
    pathstraightness_dicts = load_pathstraightness_dicts(outputs_dir)
    if not pathstraightness_dicts:
        raise FileNotFoundError(
            f"No {DICT_FILENAME} files found below {outputs_dir}"
        )

    # take only selected ones
    pathstraightness_dicts = [tupl for tupl in pathstraightness_dicts if tupl[0] in suffix_list]

    figure, axis = plt.subplots(figsize=(11, 7))
    if len(pathstraightness_dicts) <= 3:
        colors = ["blue", "red", "purple"][:len(pathstraightness_dicts)]
    else:
        colors = plt.get_cmap("turbo")(
            [index / max(len(pathstraightness_dicts) - 1, 1) for index in range(len(pathstraightness_dicts))]
        )

    for color, (label, values) in zip(colors, pathstraightness_dicts):
        epochs = sorted(values)
        axis.plot(
            epochs,
            [values[epoch] for epoch in epochs],
            marker="o",
            color=color,
            label=suffix_to_name[label],
        )

    axis.set_title("Path Straightness Across Trained Models")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Path Straightness (Integrated Trajectory Length / Linear Trajectory Length)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    print(f"Saved combined path-straightness plot to {plot_path}")


# region ==== SEEDED RUNS (mean +- std over training seeds) ====
OUT_DIR = PLOTS_DIR / "pathstraightness"


def load_pathstraightness(suffix: str) -> tuple[dict, set]:
    """({epoch: value}, interpolated epochs) of a model. Only INTERPOLATE_SUFFIXES are interpolated."""
    d = {}
    for p in sorted((OUTPUTS_DIR / suffix).glob(f"*/{DICT_FILENAME}")):
        d.update({int(k): float(v) for k, v in torch.load(p, map_location="cpu", weights_only=True).items()})
    if suffix in INTERPOLATE_SUFFIXES:
        return interpolate_to_80_epoch_grid(d)
    return d, set()


def plot_pathstraightness_seeded(models: list[str], save_path):
    """Mean +- std over training seeds. Hollow markers = interpolated epochs."""
    fig, ax = plt.subplots(figsize=(11, 7))
    for m in models:
        loaded = [load_pathstraightness(s) for s in seeds_of(m)]
        epochs, mean, std = curve_stats([d for d, _ in loaded])
        plot_curve(ax, epochs, mean, std, label=label_n(m), hollow=set().union(*[h for _, h in loaded]))
    ax.set_title("Path Straightness Across Trained Models")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Path Straightness (Integrated Trajectory Length / Linear Trajectory Length)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = [BASE, IP_CONST, IP_CUSTOM_01, IP_CUSTOM_02]
    plot_pathstraightness_seeded(models, OUT_DIR / f"pathstraightness_all_models_{TAG}.png")
    # original single-run plot
    plot_pathstraightness(suffix_list=models, suffix_to_name=SUFFIX_TO_NAME, plot_path=OUT_DIR / "pathstraightness_all_models.png")

# endregion


if __name__ == "__main__":
    main()