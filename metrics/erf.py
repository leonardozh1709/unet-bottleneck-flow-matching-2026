# ==================================================================
# effective receptive field using probenius norms of projected jacobain matrix (gaussian probe), see paper
# ==================================================================


import os
import gc
import dill
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import replace

from Utils.loadmodel import load_models
from Utils.UNet import UNetPipeline
from Utils.config import Config
from Utils.paths import OUTPUTS_DIR, PLOTS_DIR
from Utils.registry import BASE, KL_XTICKS, SUFFIX_TO_NAME
from Utils.seeds import seeds_of


def set_seed(seed=42):
    """Sets deterministic random seeds globally per run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def get_resblock_modules(model):
    """Extracts ResBlocks from encoder, bottleneck, and decoder."""
    modules = []
    for db in model.net.encoder:
        modules.append(db.resblock1)
        modules.append(db.resblock2)
    for i in range(0, len(model.net.bottleneck), 2):
        modules.append(model.net.bottleneck[i])  # even indices = ResBlock
    for ub in model.net.decoder:
        modules.append(ub.resblock1)
        modules.append(ub.resblock2)
    assert len(modules) == 14, f"Expected 14 ResBlocks, found {len(modules)}"
    return modules


def compute_erf_batched(model, config=None, n_reps=32, batch_size=8, img_size=32, device="cuda"):
    """
    Computes ERF maps across model ResBlocks using squared input gradients.
    Returns averaged gradient heatmaps and raw sample maps for stddev calculations.
    """
    model.eval()
    model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False

    resblocks = get_resblock_modules(model)
    num_blocks = len(resblocks)
    
    sum_grad_maps = [np.zeros((img_size, img_size), dtype=np.float64) for _ in range(num_blocks)]
    sample_grad_maps = [[] for _ in range(num_blocks)]
    
    n_batches = (n_reps + batch_size - 1) // batch_size
    n_classes = getattr(config, 'n_classes', 10) if config else 10

    for b in tqdm(range(n_batches), desc="Evaluating Batches", leave=False):
        current_b_size = min(batch_size, n_reps - b * batch_size)

        for i, resblock in enumerate(resblocks):
            input_images = torch.randn(
                current_b_size, 3, img_size, img_size, 
                device=device, requires_grad=True
            )

            activation = {}
            def hook_fn(module, input, output):
                activation['output'] = output

            handle = resblock.register_forward_hook(hook_fn)

            t = torch.rand((current_b_size,), device=device)
            c = torch.randint(0, n_classes, (current_b_size,), device=device)

            _ = model(input_images, t, c)
            act = activation['output']

            h_center = act.shape[2] // 2
            w_center = act.shape[3] // 2
            central_units = act[:, :, h_center, w_center]

            v = torch.randn_like(central_units)
            model.zero_grad()
            central_units.backward(v)

            # Squared gradient magnitude aggregation across channels
            sq_grads = input_images.grad.pow(2).sum(dim=1)
            sq_grads_np = sq_grads.detach().cpu().numpy()
            
            for sample_idx in range(current_b_size):
                sample_grad_maps[i].append(sq_grads_np[sample_idx])

            sum_grad_maps[i] += sq_grads.sum(dim=0).detach().cpu().numpy()

            handle.remove()
            input_images.grad = None
            del input_images, act, central_units, sq_grads

        torch.cuda.empty_cache()
        gc.collect()

    avg_grad_maps = [g_map / n_reps for g_map in sum_grad_maps]
    return avg_grad_maps, sample_grad_maps


def compute_erf_diameters_per_sample(sample_grad_maps):
    """Calculates effective RF diameter for every individual repetition sample map."""
    num_blocks = len(sample_grad_maps)
    n_reps = len(sample_grad_maps[0])
    diameters = np.zeros((num_blocks, n_reps))
    
    for i in range(num_blocks):
        for r in range(n_reps):
            grad_map = sample_grad_maps[i][r]
            total_weight = np.sum(grad_map)
            if total_weight == 0:
                diameters[i, r] = 0.0
                continue
                
            y_indices, x_indices = np.indices(grad_map.shape)
            centroid_x = np.sum(x_indices * grad_map) / total_weight
            centroid_y = np.sum(y_indices * grad_map) / total_weight

            rms_radius = np.sqrt(
                np.sum(((x_indices - centroid_x) ** 2 + (y_indices - centroid_y) ** 2) * grad_map) / total_weight
            )
            diameters[i, r] = 2.0 * rms_radius
            
    return diameters



# Single Model Load & Evaluation Engine

def load_and_eval_single_suffix(suffix, device="cuda", n_reps=32, batch_size=8, img_size=32, seed=1709):
    """
    Loads model for a given suffix (or initializes random) and returns avg maps and sample-level diameters.
    """
    set_seed(seed)
    if suffix == "random":
        config = Config()
        model = UNetPipeline(replace(config, do_zero_init=False)).to(device)
    else:
        config_path = OUTPUTS_DIR / suffix / f"config_{suffix}.pkl"
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = dill.load(f)
        else:
            config = Config()
        _, model = load_models(suffix, device=device)

    avg_grad_maps, sample_grad_maps = compute_erf_batched(
        model, config=config, n_reps=n_reps, batch_size=batch_size, img_size=img_size, device=device
    )
    sample_diameters = compute_erf_diameters_per_sample(sample_grad_maps)
    return avg_grad_maps, sample_diameters


def evaluate_suffix_with_reseeds(suffix, device="cuda", n_reps=32, batch_size=8, img_size=32, random_model_n_seeds=1):
    """
    Checks for reseed runs or evaluates multiple random seeds.
    Returns mean over seeds of the per-seed rep-mean diameters and the std (ddof=1) across seeds (0 if one seed).
    """
    all_seed_maps = []
    all_seed_sample_diameters = []

    if suffix == "random":
        for s_idx in range(random_model_n_seeds):
            print(f" -> Evaluating random model seed {s_idx + 1}/{random_model_n_seeds}")
            maps, sample_diameters = load_and_eval_single_suffix(
                suffix, device, n_reps, batch_size, img_size, seed=1709 + s_idx
            )
            all_seed_maps.append(maps)
            all_seed_sample_diameters.append(sample_diameters)
    else:
        for s in seeds_of(suffix):
            print(f" -> Evaluating seed run: {s}")
            maps, sample_diameters = load_and_eval_single_suffix(s, device, n_reps, batch_size, img_size)
            all_seed_maps.append(maps)
            all_seed_sample_diameters.append(sample_diameters)

    # Average spatial maps across seed runs for heatmaps
    mean_maps = [
        np.mean([seed_maps[i] for seed_maps in all_seed_maps], axis=0) 
        for i in range(len(all_seed_maps[0]))
    ]

    # Per-seed mean over reps first, then mean/std across seeds (reps are not independent evidence).
    # n_seeds == 1 -> std = 0 -> no error bars are drawn.
    per_seed_means = np.stack([d.mean(axis=1) for d in all_seed_sample_diameters])  # (n_seeds, num_blocks)
    means = per_seed_means.mean(axis=0)
    stds = per_seed_means.std(axis=0, ddof=1) if per_seed_means.shape[0] > 1 else np.zeros_like(means)

    return mean_maps, means, stds


# plotting

def plot_master_erf_figure(erf_diameters, erf_stds, avg_grad_maps, save_dir: str, display_name: str, resblock_indices=[0, 6, 13], resblock_names=None):
    """Plots combined figure containing ERF diameter progression bar chart with stddev and heatmaps."""
    num_heatmaps = len(resblock_indices)
    fig, axes = plt.subplots(1, 1 + num_heatmaps, figsize=(5 * (1 + num_heatmaps), 4.5))

    def idx_to_color(i):
        return 'crimson' if i in resblock_indices else 'royalblue'

    colors = [idx_to_color(i) for i in range(len(erf_diameters))]

    # Bar chart of ERF diameters across all ResBlocks with stddev error bars
    axes[0].bar(
        range(len(erf_diameters)), erf_diameters, 
        color=colors, edgecolor='black', alpha=0.85
    )

    # filter indices where std > 0 and add error bars only for those
    x_err_indices = [i for i, std in enumerate(erf_stds) if std > 0]
    y_err_vals = [erf_diameters[i] for i in x_err_indices]
    stds_filtered = [erf_stds[i] for i in x_err_indices]

    axes[0].errorbar(
        x_err_indices, y_err_vals, yerr=stds_filtered, 
        fmt='none', c='black', capsize=3
    )
    axes[0].set_xlabel('ResBlock Index')
    axes[0].set_ylabel('Effective RF Diameter (Pixels)')
    axes[0].set_title(f'ERF Diameter: {display_name}')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Spatial heatmaps for specified layers (using discrete raw grid layout)
    for ax_idx, idx in enumerate(resblock_indices):
        ax = axes[ax_idx + 1]
        grad_map = avg_grad_maps[idx]
        
        im = ax.imshow(grad_map, cmap='hot', interpolation='nearest')
        title = resblock_names[ax_idx] if resblock_names else f'ResBlock {idx}'
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_dir, dpi=300)
    plt.close()


def plot_resblock_comparison(resblock_idx, models_xticks, models_ybars, model_results, suffix_to_name, output_dir):
    """
    Generates a comparison plot for a single ResBlock.
    Models in `models_xticks` are plotted as x-axis points.
    Models in `models_ybars` are evaluated and drawn as horizontal benchmark lines.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x_indices = []
    x_labels = []
    y_means = []
    y_stds = []

    # 1. Process x-axis data points
    for idx, suffix in enumerate(models_xticks):
        display_label = f"Pos {idx}"
        if suffix is not None:
            display_label = suffix_to_name.get(suffix, suffix)
            if suffix in model_results:
                _, mean_d, std_d = model_results[suffix]
                y_means.append(mean_d[resblock_idx])
                y_stds.append(std_d[resblock_idx])
            else:
                y_means.append(np.nan)
                y_stds.append(0.0)
        else:
            y_means.append(np.nan)
            y_stds.append(0.0)

        x_indices.append(idx)
        x_labels.append(display_label if suffix is not None else "")

    valid_x = [x for x, y in zip(x_indices, y_means) if not np.isnan(y)]
    valid_y = [y for y in y_means if not np.isnan(y)]
    valid_yerr = [e for y, e in zip(y_means, y_stds) if not np.isnan(y)]

    ax.errorbar(
        valid_x, valid_y, yerr=valid_yerr, fmt='o-', color='royalblue', 
        ecolor='crimson', elinewidth=1.5, capsize=4, label='Model ERF Diameter'
    )

    # 2. Process y-bar benchmark model lines
    if models_ybars:
        colors = plt.cm.Dark2.colors
        for line_idx, ybar_suffix in enumerate(models_ybars):
            if ybar_suffix is not None and ybar_suffix in model_results:
                _, ybar_means, ybar_stds = model_results[ybar_suffix]
                y_val = ybar_means[resblock_idx]
                y_std = ybar_stds[resblock_idx]
                
                label_name = suffix_to_name.get(ybar_suffix, ybar_suffix)
                color = colors[line_idx % len(colors)]
                
                # Draw parallel line for y-bar model
                ax.axhline(y=y_val, color=color, linestyle='--', alpha=0.8, label=f'Benchmark: {label_name}')
                
                # Shaded band for y-bar stddev if available
                if y_std > 0:
                    ax.axhspan(y_val - y_std, y_val + y_std, color=color, alpha=0.15)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Effective RF Diameter (Pixels)')
    ax.set_xlabel('Model Configuration')
    ax.set_title(f'ResBlock {resblock_idx} ERF Diameter Progression Across Models')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"resblock_{resblock_idx}_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()




if __name__ == "__main__":
    set_seed(1709)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 32
    n_reps = 200
    batch_size = 8

    # Number of independent random model initializations/seeds to evaluate 
    RANDOM_MODEL_N_SEEDS = 10

    SESSION_TAG = "KL"
    output_dir = PLOTS_DIR / "erf" / SESSION_TAG
    os.makedirs(output_dir, exist_ok=True)

    # Models plotted as x-axis points
    models_xticks = KL_XTICKS

    # parallel horizontal lines
    models_ybars = ['random', BASE]

    suffix_to_name = {**SUFFIX_TO_NAME, 'random': 'Random Init'}

    all_active_suffixes = list(set(
        [s for s in models_xticks if s is not None] + 
        [s for s in models_ybars if s is not None]
    ))

    # Evaluate all models once
    model_results = {}
    master_plots_dir = os.path.join(output_dir, "master_figures")
    os.makedirs(master_plots_dir, exist_ok=True)

    for suffix in all_active_suffixes:
        display_name = suffix_to_name.get(suffix, suffix)
        print(f"\nEvaluating Model Suffix: {suffix} ({display_name})")
        
        maps, mean_d, std_d = evaluate_suffix_with_reseeds(
            suffix, device=_device, 
            n_reps=n_reps, batch_size=batch_size, img_size=img_size,
            random_model_n_seeds=RANDOM_MODEL_N_SEEDS
        )
        model_results[suffix] = (maps, mean_d, std_d)

        # Generate individual Master Figures
        master_save_path = os.path.join(master_plots_dir, f"master_figure_{suffix}.png")
        plot_master_erf_figure(
            mean_d, std_d, maps, save_dir=master_save_path, display_name=display_name,
            resblock_indices=[0, 6, 13], 
            resblock_names=["First Encoder", "First Bottleneck", "Final Decoder"]
        )

    if len(all_active_suffixes) > 1:
        # Generate cross-model comparison plots per ResBlock (0 to 13)
        num_resblocks = 14
        for r_idx in range(num_resblocks):
            print(f"Generating ResBlock {r_idx} comparison plot...")
            plot_resblock_comparison(
                r_idx, models_xticks, models_ybars, model_results, suffix_to_name, output_dir
            )

    print(f"\nEvaluation complete. Output saved to: {output_dir}")