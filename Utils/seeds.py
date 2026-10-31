import numpy as np
import torch

from Utils.paths import CACHE_DIR, OUTPUTS_DIR
from Utils.registry import FORCE_RECOMPUTE, INTERPOLATE_SUFFIXES, SEEDED, TARGET_EPOCH_GAP, name_of


def seeds_of(suffix: str) -> list[str]:
    """Training runs of a model that have a final checkpoint (a reseed may still be training)."""
    return [s for s in SEEDED.get(suffix, [suffix]) if (OUTPUTS_DIR / s / f"unet_{s}.pth").is_file()]


def label_n(suffix: str) -> str:
    """Display name with the number of training seeds, e.g. 'base (n=3)'."""
    return f"{name_of(suffix)} (n={len(seeds_of(suffix))})"


def agg(values: list) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean and sample std (ddof=1) over the first axis. std is None for a single seed."""
    arr = np.asarray(values, dtype=float)
    return arr.mean(axis=0), (arr.std(axis=0, ddof=1) if len(arr) > 1 else None)


def cached(path, compute):
    """Loads path if it exists (unless FORCE_RECOMPUTE), otherwise computes and saves it."""
    if path.exists() and not FORCE_RECOMPUTE:
        return torch.load(path, weights_only=False)
    data = compute()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def interpolate_to_80_epoch_grid(d: dict) -> tuple[dict, set]:
    """Linear interpolation of {epoch: value} onto the 80-epoch grid (79, 159, ...) inside the measured range.
    Returns (new dict, set of interpolated epochs)."""
    epochs = np.array(sorted(d), dtype=float)
    values = np.array([float(d[int(e)]) for e in epochs])
    targets = np.arange(TARGET_EPOCH_GAP - 1, epochs.max() + 1e-9, TARGET_EPOCH_GAP)
    targets = targets[targets >= epochs.min()]
    new = {int(t): float(np.interp(t, epochs, values)) for t in targets}
    return new, {e for e in new if e not in d}


def load_fid25k_raw(suffix: str) -> dict:
    """FID-25k of a model as stored on disk. A resumed run has several training_* folders: they are merged."""
    merged = {}
    for p in sorted((OUTPUTS_DIR / suffix).glob("training_*/fid_dict.pt")):
        merged.update({int(k): float(v) for k, v in torch.load(p, map_location="cpu", weights_only=False).items()})
    return merged


def load_fid25k(suffix: str) -> tuple[dict, set]:
    """(FID-25k dict, interpolated epochs). Only INTERPOLATE_SUFFIXES are interpolated."""
    d = load_fid25k_raw(suffix)
    if suffix in INTERPOLATE_SUFFIXES:
        return interpolate_to_80_epoch_grid(d)
    return d, set()


def interpolation_error(d: dict) -> str:
    """Leave-one-out check of the interpolation: every interior point is predicted from its two neighbours
    (100 epochs apart) and compared with the measurement. Positive error = prediction too high."""
    ep = sorted(d)
    lines = ["epoch | measured | predicted | error [%]"]
    for k in range(1, len(ep) - 1):
        pred = np.interp(ep[k], [ep[k - 1], ep[k + 1]], [d[ep[k - 1]], d[ep[k + 1]]])
        lines.append(f"{ep[k]:5d} | {d[ep[k]]:9.4f} | {pred:9.4f} | {(pred - d[ep[k]]) / d[ep[k]] * 100:+6.2f}")
    return "\n".join(lines)


def curve_stats(dicts: list[dict]) -> tuple[list, np.ndarray, np.ndarray | None]:
    """Mean and std over seeds of {epoch: value} dicts, on their common epochs."""
    epochs = sorted(set.intersection(*[set(d) for d in dicts]))
    mean, std = agg([[d[e] for e in epochs] for d in dicts])
    return epochs, mean, std


def plot_curve(ax, epochs, mean, std, label, hollow=()):
    """Line with markers and a std band. Epochs in `hollow` (interpolated) get hollow markers."""
    (line,) = ax.plot(epochs, mean, "-", label=label)
    c = line.get_color()
    for e, m in zip(epochs, mean):
        if e in hollow:
            ax.plot(e, m, "o", color=c, markerfacecolor="white", markeredgewidth=1.5)
        else:
            ax.plot(e, m, "o", color=c)
    if std is not None:
        ax.fill_between(epochs, mean - std, mean + std, color=c, alpha=0.2)