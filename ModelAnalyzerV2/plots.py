import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import math
import torch
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from matplotlib.colors import LogNorm, Normalize

from .gui.tools import get_leafs_in_order



def plot_stats(module: nn.Module, title: str, stats_to_plot: list[str]):
    """creates 2 subplots, one for the modules bwd_stats and one for its fwd_stats. Each of the subplots show stats in stats_to_plot"""

    fig, (ax_fwd, ax_bwd) = plt.subplots(2, 1, figsize=(10, 8))

    # forward stats
    for name, history in module._fwd_stats.items():
        history = [h.cpu() if isinstance(h, torch.Tensor) else h for h in history]
        if name in stats_to_plot:
            ax_fwd.plot(np.asarray(history, dtype=np.float32), label=name)

    ax_fwd.set_title("Forward stats")
    ax_fwd.set_ylabel("Value")
    ax_fwd.set_xlabel("Iteration")
    ax_fwd.legend(fontsize=8)
    ax_fwd.grid(True)

    # backward stats 
    for name, history in module._bwd_stats.items():
        history = [h.cpu() if isinstance(h, torch.Tensor) else h for h in history]
        if name in stats_to_plot:
            ax_bwd.plot(np.asarray(history, dtype=np.float32), label=name)

    ax_bwd.set_title("Backward stats")
    ax_bwd.set_xlabel("Iteration")
    ax_bwd.set_ylabel("Value")
    ax_bwd.legend(fontsize=8)
    ax_bwd.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def plot_emastat_vs_order(ordered_modules: list[nn.Module], selected_stats: list, title: str):
    """Plots the module's ema stats vs the execution order. Axis is labeled with the module class names."""

    if selected_stats == [] or selected_stats is None:
        return
    
    fig, (ax_fwd, ax_bwd) = plt.subplots(2, 1, figsize=(10, 8))

    # ticks
    xticks = [f"{i}:{module.__class__.__name__}" for i, module in enumerate(ordered_modules)]

    # create ordered_fwd_ema and ordered_bwd_ema
    ordered_fwd_ema = {}
    ordered_bwd_ema = {}
    for stat in selected_stats:
        ordered_fwd_ema[stat] = []
        ordered_bwd_ema[stat] = []

    for module in ordered_modules:
        for k, v in module._ema_fwd_stats.items():
            if k in selected_stats:
                ordered_fwd_ema[k].append(v.item())
        for k, v in module._ema_bwd_stats.items():
            if k in selected_stats:
                ordered_bwd_ema[k].append(v.item())

    # fwd 
    len_values = 0
    for stat, values in ordered_fwd_ema.items():
        if stat in selected_stats:
            ax_fwd.plot(values, label=stat)
            if len(values) > len_values:
                len_values = len(values)

    ax_fwd.set_title("Forward")
    ax_fwd.set_ylabel("Value")
    ax_fwd.set_xlabel("Model Operation Order ('Depth')")
    ax_fwd.legend(fontsize=8)

    positions = range(len(xticks)) # positions for the ticks
    ax_fwd.set_xticks(positions)
    ax_fwd.set_xticklabels(xticks, rotation=90, fontsize=8)# set tick labels
    ax_fwd.grid(True)

    # bwd 
    for stat, values in ordered_bwd_ema.items():
        if stat in selected_stats:
            ax_bwd.plot(values, label=stat)

    ax_bwd.set_title("Backward")
    ax_bwd.set_ylabel("Value")
    ax_bwd.set_xlabel("Model Operation Order ('Depth')")
    ax_bwd.legend(fontsize=8)

    ax_bwd.set_xticks(positions)
    ax_bwd.set_xticklabels(xticks, rotation=90, fontsize=8)
    ax_bwd.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def plot_gradient_magnitude_in_module_leafs(module: nn.Module, title: str, scale_aware: bool = False, delete_empty: bool = True):
    """Plots gradient magnitudes of all leaf modules of the input module assuming they have ._gradients, ._order and ._is_leaf"""

    # extract all leaf modules from this module
    leafs_in_order = get_leafs_in_order(module)

    # leave out empty ones
    if delete_empty:
        leafs_in_order = [m for m in leafs_in_order if m._gradients]

    # subplots layout
    n = len(leafs_in_order)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    
    # create subplots
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 8)) # ax is a np array of subplots or just a subplot
    subplots = ax.flatten() if hasattr(ax, "flatten") else [ax] 

    # consistent scaling across subplots 
    all_values = [
        grad["scale_aware_influence"] if scale_aware else grad["influence"]
        for leaf in leafs_in_order
        for grad in leaf._gradients.values()
    ]
    ymax = max(all_values) if all_values else 1
    ymax = ymax.cpu() if isinstance(ymax, torch.Tensor) else ymax

    # populate subplots
    for i, leafmodule in enumerate(leafs_in_order):

        p = subplots[i]
        p.set_ylim(0, ymax)

        if not leafmodule._gradients:
            p.set_title(f"{leafmodule.__class__.__name__} #{leafmodule._order} (no data)")
            continue

        x = []
        y = []

        for label, grad in leafmodule._gradients.items():
            # module._gradients[label] = {"influence": influence_ema, "scale_aware_influence": scale_aware_influence_ema}, label is the parameter name
            g = grad["scale_aware_influence"] if scale_aware else grad["influence"]

            x.append(label)
            y.append(g.cpu())

        p.bar(range(len(x)), y) # create bar diagram
        p.set_xticks(range(len(x)))
        p.set_xticklabels(x, rotation=45, ha="right", fontsize=8) # rotate for readability

        p.set_title(f"{leafmodule.__class__.__name__} #{leafmodule._order}")
        p.grid(True)

    # delete empty subplots
    for j in range(n, len(subplots)):
        fig.delaxes(subplots[j])

    plt.tight_layout(h_pad=10)
    plt.gcf().canvas.manager.set_window_title(title)
    plt.show()



def collect_module_stats(modules_in_order: list) -> list[dict]:
    """
    Returns per-module stats (no plotting).
    """

    stats = []

    for i, module in enumerate(modules_in_order):

        is_leaf = len(list(module.children())) == 0

        if not is_leaf:
            continue

        cached_grads = []
        weights = []
        param_count = 0

        for p in module.parameters(recurse=False):
            param_count += p.numel()
            weights.append(p.detach().flatten())

        if not weights:
            continue

        w = torch.cat(weights)

        # The live p.grad values are normally cleared by the training loop.
        # _gradients contains the EMA summaries captured during backward hooks.
        for gradient in module._gradients.values():
            cached_grads.append(gradient["influence"])

        if cached_grads:
            mean_abs_grad = torch.stack([
                value if isinstance(value, torch.Tensor) else torch.tensor(value)
                for value in cached_grads
            ]).mean().item()
            # The cache stores mean absolute gradients per parameter, so this
            # is an RMS-style estimate of the full gradient norm.
            grad_norm = math.sqrt(sum(
                (float(value) ** 2) * parameter.numel()
                for value, parameter in zip(cached_grads, module.parameters(recurse=False))
            ))
        else:
            print(f"Module {i}:{module.__class__.__name__} has no cached gradients.")
            grad_norm = 0.0
            mean_abs_grad = 0.0

        weight_norm = torch.linalg.vector_norm(w).item()

        stats.append({
            "index": i,
            "type": module.__class__.__name__,
            "param_count": param_count,
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            "grad_weight_ratio": grad_norm / (weight_norm + 1e-12),
            "mean_abs_grad": mean_abs_grad,
        })

    return stats


def plot_gradient_flow(modules_in_order: list, metric: str = "grad_weight_ratio", log_scale: bool = True, title: str = "Gradient Flow (EMA cached gradients)") -> list[dict]:
    """Plots EMA-smoothed cached gradient values.

    Available metrics:
    - grad_weight_ratio (best for spotting vanishing/exploding gradients)
    - grad_norm
    - weight_norm
    - mean_abs_grad
    """

    stats = collect_module_stats(modules_in_order)

    values = [s[metric] for s in stats]

    labels = [
        f"{s['index']}:{s['type']}"
        for s in stats
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(values, marker="o", linewidth=1)

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Execution Order (Leaf Modules)")
    ax.set_ylabel(metric)

    if False:  # Optional: show module types on x-axis
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)

    ax.grid(True, which="both")

    plt.tight_layout()
    plt.show()

    return stats


def plot_gradient_flow_by_module_type(modules_in_order: list, metric: str = "grad_weight_ratio", log_scale: bool = True, title: str = "Gradient Flow by Module Type (EMA cached gradients)") -> None:

    stats = collect_module_stats(modules_in_order)

    type_to_weighted_sum = defaultdict(float)
    type_to_param_sum = defaultdict(int)

    for s in stats:
        t = s["type"]

        type_to_weighted_sum[t] += s[metric] * s["param_count"]
        type_to_param_sum[t] += s["param_count"]

    types = []
    values = []

    for t in sorted(type_to_weighted_sum.keys()):
        if type_to_param_sum[t] == 0:
            continue

        types.append(t)

        # FIX 4: safe division (prevents silent NaNs)
        values.append(
            type_to_weighted_sum[t] / max(type_to_param_sum[t], 1)
        )

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(types, values)

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Module Type")
    ax.set_ylabel(f"Weighted Avg {metric}")

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=90, fontsize=8)

    ax.grid(True, which="both")

    plt.tight_layout()
    plt.show()


def plot_stat_vs_time_vs_depth(modules_in_order: list, module: str, selected_stats: list[str], plot_mode: str = "heatmap", stat_mode: str = "fwd", use_log_scale: bool = True, display_mode: str = "raw", window_size: int = 1, model: nn.Module | None = None):
    """
    plot_mode: "heatmap" or "3d"
    stat_mode: "fwd" or "bwd"
    display_mode: "raw" or "per_module_normalized"
    window_size: number of timesteps to average into each plotted timestep
    model: UNet or a wrapper containing the UNet, used for encoder/decoder markers
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if display_mode not in {"raw", "per_module_normalized"}:
        raise ValueError(f"Unknown display_mode: {display_mode}")

    print(f"Module: {module}")

    # create dict with 2d tensors of data
    stat_stacks: dict[str, list] = {}
    stat_modules: dict[str, list[nn.Module]] = {}
    for current_module in tqdm(modules_in_order, "building statVStimeVSdepth plot"):

        if module != "all" and current_module.__class__.__name__ != module:
            continue 



        if stat_mode == "bwd":
            S = current_module._bwd_stats
        else:
            S = current_module._fwd_stats
        for name, history in S.items():
            if name not in selected_stats:
                continue
            history = [h.cpu() if isinstance(h, torch.Tensor) else h for h in history]
            if name not in stat_stacks:
                stat_stacks[name] = []
                stat_modules[name] = []
            stat_stacks[name].append(history)
            stat_modules[name].append(current_module)
            print(f"Module {current_module.__class__.__name__} #{current_module._order}: {name} history length: {len(history)}")

    # stack lists of 1d tensors to 2d tensor
    for k in stat_stacks.keys():
        stat_stacks[k] = torch.stack([
            h if isinstance(h, torch.Tensor) else torch.tensor(h) 
            for h in stat_stacks[k]
        ])
        data = stat_stacks[k].float()

        if window_size > 1:
            windows = [window.mean(dim=1) for window in torch.split(data, window_size, dim=1)]
            data = torch.stack(windows, dim=1)

        if display_mode == "per_module_normalized":
            # get medians across each row (module)
            row_medians = data.median(dim=1, keepdim=True).values
            # get interquartile range (q1 - q3) across each row (module)
            row_iqr = torch.quantile(data, 0.75, dim=1, keepdim=True) - torch.quantile(data, 0.25, dim=1, keepdim=True)
            # subtract median -> new median is 0
            # devide by interquartile range -> new IQR is 1
            # this is better than mean and std because it is more robust to outliers 
            # its more robust to outliers because it uses the median and interquartile range instead of mean and std
            data = (data - row_medians) / row_iqr.clamp_min(torch.finfo(data.dtype).eps)

        stat_stacks[k] = data

    title_parts = []
    if module != "all":
        title_parts.append(f"Module: {module}")
    if display_mode == "per_module_normalized":
        title_parts.append("Per-module normalized")
    title_parts.append(f"Window mean: {window_size} timesteps")
    title_suffix = " - " + " | ".join(title_parts)

    def boundary_rows(plotted_modules: list[nn.Module]) -> dict[str, int]:
        if model is None or not plotted_modules:
            return {}

        unet = model
        if not hasattr(unet, "encoder") and hasattr(unet, "net"):
            unet = unet.net
        if not hasattr(unet, "encoder") or not hasattr(unet, "decoder"):
            return {}

        encoder_modules = set(unet.encoder.modules())
        decoder_modules = set(unet.decoder.modules())
        rows = {}
        encoder_rows = [i for i, candidate in enumerate(plotted_modules) if candidate in encoder_modules]
        decoder_rows = [i for i, candidate in enumerate(plotted_modules) if candidate in decoder_modules]
        if encoder_rows:
            rows["last encoder module"] = max(encoder_rows)
        if decoder_rows:
            rows["first decoder module"] = min(decoder_rows)
        return rows

    # Heatmaps share one figure with multiple subplots. Each 3D plot gets its
    # own single-axis figure because Matplotlib's 3D subplot layout collapses
    # when several projected axes are added to the same 111 grid.
    n_plots = len(stat_stacks)
    if plot_mode == "heatmap":
        fig, ax = plt.subplots(nrows=1, ncols=n_plots)
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])

    for index, (name, data) in enumerate(stat_stacks.items()):

        color_min, color_max = torch.quantile(data, torch.tensor([0.01, 0.99])).tolist()
        if color_min == color_max:
            color_min -= 1e-6
            color_max += 1e-6
        use_log_norm = use_log_scale and bool(torch.all(data > 0))
        color_norm = LogNorm(vmin=color_min, vmax=color_max) if use_log_norm else Normalize(vmin=color_min, vmax=color_max)

        if plot_mode == "heatmap":
            subplot = ax[index]
            im = subplot.imshow(data, aspect="auto", cmap="viridis", norm=color_norm)
            rows = boundary_rows(stat_modules[name])
            for label, row in rows.items():
                color = "white" if label == "last encoder module" else "red"
                subplot.axhline(row, color=color, linewidth=1.5, linestyle="--", label=label)
            subplot.set_title(f"{name} ({stat_mode}){title_suffix}")
            subplot.set_xlabel("Time (Iterations)")
            subplot.set_ylabel("Depth (Module Order)")
            if rows:
                subplot.legend(loc="upper right", fontsize=8)
            fig.colorbar(im, ax=subplot)

        elif plot_mode == "3d":
            X, Y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
            Z = data.numpy()
            fig_3d = plt.figure()
            ax3d = fig_3d.add_subplot(111, projection="3d")
            ax3d.plot_surface(X, Y, Z, cmap='viridis', norm=color_norm)
            ax3d.set_title(f"{name} ({stat_mode}){title_suffix}")
            ax3d.set_xlabel("Time (Iterations)")
            ax3d.set_ylabel("Depth (Module Order)")
            ax3d.set_zlabel(name)
            fig_3d.tight_layout()

    if plot_mode == "heatmap":
        fig.tight_layout()
    plt.show()
        

    


