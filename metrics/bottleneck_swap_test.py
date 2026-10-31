# ==================================================================
# swap the encoded state h and see the resulting change in y using average MSE
# ==================================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from contextlib import nullcontext
from datetime import datetime
from tqdm import tqdm
import math

from Utils.UNet import ResBlock, UNetPipeline, sinusoidial_embedding
from Utils.loadmodel import load_models
from Utils.config import config, Config
from Utils.data import get_dataloader
from Utils.paths import PLOTS_DIR
from Utils.registry import BASE, KL_005, KL_01, KL_025, KL_05, KL_1, SUFFIX_TO_NAME
from Utils.seeds import seeds_of

# ==================================================================


autocast_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if torch.cuda.is_available()
    else nullcontext()
)


@torch.inference_mode()
def encode(model: UNetPipeline, x, t, c, use_bottleneck: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    returns (h, skips, emb)
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

    if use_bottleneck:
        for block in model.net.bottleneck:
            if isinstance(block, ResBlock):
                x = block(x, emb)
            else:
                x = block(x)

    return x, skips, emb

@torch.inference_mode()
def decode(model: UNetPipeline, x, emb, skips):
    """
    returns y
    """
    
    for block, skp in zip(model.net.decoder, reversed(skips)):
        x = block(x, skp, emb)

    x = model.net.outnorm(x)
    x = model.net.outact(x)
    x = model.net.outconv(x)

    return x

@torch.inference_mode()
def measure_swap_sensitivity(
        device: str,
        split: str,
        model: nn.Module,
        n_samples: int, # number of images to do in total
        batch_size: int = 96, # number of images to encode/decode at once
        n_timestep_bins: int = 25 # number of timestep bins
):
    model.eval()

    if split == "test":
        loader = get_dataloader(Config(batch_size=batch_size), "test")
    else:
        loader = get_dataloader(Config(batch_size=batch_size), "train")


    h_lst = []
    skips_lst = []
    emb_lst = []
    timestep_lst = []
    # get encoded states h
    total_samples = 0
    for image, label in tqdm(loader, total = math.ceil(n_samples/batch_size)):
        if total_samples >= n_samples:
            break

        x_1 = image.to(device) # (B, C, H, W)
        label = label.to(device) # (B,)

        B = label.shape[0]

        # sample x_0 
        x_0 = torch.randn_like(x_1)

        # sample t uniformly
        t = torch.rand((B,), device=device)

        # conditional flow (linear interpolation)
        x_t = t.view(-1, 1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1, 1)) * x_0

        with autocast_context:
            h, skips, emb = encode(model, x_t, t, label, use_bottleneck=True)

        h_lst.append(h)
        skips_lst.append(skips)
        emb_lst.append(emb)
        timestep_lst.append(t)

        total_samples += B

    # turn lists into batch tensors
    h_lst = torch.cat(h_lst, dim=0)
    emb_lst = torch.cat(emb_lst, dim=0)
    timestep_all = torch.cat(timestep_lst, dim=0)

    # Rebuild the skip tensors per encoder depth so each batch slice keeps its own skip stack.
    if skips_lst and len(skips_lst[0]) > 0:
        skips_all = [
            torch.cat([batch_skips[layer_idx] for batch_skips in skips_lst], dim=0)
            for layer_idx in range(len(skips_lst[0]))
        ]
    else:
        skips_all = []

    # shuffle 
    # (without self-swaps)
    perm = (torch.arange(h_lst.shape[0], device=device) + 1) % h_lst.shape[0] # move ahead by one and warp around
    h_lst_shuffled = h_lst[perm]

    MSE_lst = []
    
    # get decoded outputs y with and without shuffling h
    for batch_idx in range(0, h_lst.shape[0], batch_size):
        h = h_lst[batch_idx:batch_idx + batch_size]
        h_shuffled = h_lst_shuffled[batch_idx:batch_idx + batch_size]
        _emb = emb_lst[batch_idx:batch_idx + batch_size]
        _skip = [skip[batch_idx:batch_idx + batch_size] for skip in skips_all]

        with autocast_context:
            y = decode(model, h, _emb, _skip)
            y_shuffled = decode(model, h_shuffled, _emb, _skip)
        MSE = ((y - y_shuffled)**2).mean(dim=(1, 2, 3)) # shape (B,)
        MSE_lst.append(MSE)

    # bin by timestep
    MSE_all = torch.cat(MSE_lst, dim=0)
    avg_MSE_by_timestep = {}
    for bin_idx in range(n_timestep_bins):
        lower = bin_idx / n_timestep_bins
        upper = (bin_idx + 1) / n_timestep_bins
        if bin_idx == n_timestep_bins - 1:
            mask = (timestep_all >= lower) & (timestep_all <= upper)
        else:
            mask = (timestep_all >= lower) & (timestep_all < upper)

        if torch.any(mask):
            avg_MSE_by_timestep[bin_idx] = float(MSE_all[mask].mean().item())

    return avg_MSE_by_timestep


def main(suffix_list, suffix_to_name, show_plot=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = PLOTS_DIR / "bottleneck_swap_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_curves = {}
    for group in suffix_list:
        group_curves = []
        for suffix in group:
            _, ema_model = load_models(suffix, device=device)
            model = ema_model.eval()
            avg_mse_by_timestep = measure_swap_sensitivity(
                device=device,
                split="test",
                model=model,
                n_samples=9600,
                batch_size=200,
                n_timestep_bins=10,
            )
            group_curves.append(avg_mse_by_timestep)

        # Aggregate re-seeds (mean and std across group members)
        bin_idx = sorted(group_curves[0].keys())
        values_matrix = torch.tensor([[curve[idx] for idx in bin_idx] for curve in group_curves])
        
        mean_tensor = values_matrix.mean(dim=0)
        std_tensor = values_matrix.std(dim=0, unbiased=False) if values_matrix.shape[0] > 1 else torch.zeros_like(mean_tensor)
        
        mean_values = mean_tensor.numpy()
        std_values = std_tensor.numpy()

        model_curves[tuple(group)] = {
            "bin_idx": bin_idx,
            "mean": mean_values,
            "std": std_values,
            "has_reseeds": values_matrix.shape[0] > 1
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    for group_tuple, data in model_curves.items():
        bin_idx = data["bin_idx"]
        mean = data["mean"]
        std = data["std"]
        label = suffix_to_name.get(group_tuple[0], group_tuple[0])

        line = ax.plot(bin_idx, mean, marker="o", linewidth=1.8, markersize=4, label=label)
        color = line[0].get_color()
        if data["has_reseeds"]:
            ax.fill_between(bin_idx, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("timestep bin index")
    ax.set_ylabel("avg MSE")
    ax.set_title("Bottleneck swap sensitivity by timestep bin")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if show_plot:
        plt.show()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"avg_mse_by_timestep_{timestamp}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    suffix_list = [seeds_of(m) for m in [BASE, KL_05, KL_1, KL_025, KL_01, KL_005]]
    suffix_to_name = SUFFIX_TO_NAME
    main(suffix_list, suffix_to_name, show_plot=True)
