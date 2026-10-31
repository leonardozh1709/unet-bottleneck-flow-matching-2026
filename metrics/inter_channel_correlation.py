# measure absolute off-diagonal correlation 
# -> tells how much features influence each other in total


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
import matplotlib.pyplot as plt

from Utils.data import get_dataloader
from metrics.bottleneck_swap_test import encode
from Utils.UNet import UNetPipeline
from Utils.loadmodel import load_models
from Utils.config import config
from Utils.registry import BASE, BIP_0025, BIP_005, BIP_01, BIP_02, BIP_04, IP_CUSTOM_01, KL_005, KL_01, KL_025, KL_05, KL_1, SUFFIX_TO_NAME
from Utils.seeds import seeds_of


# ---
USE_BF16 = True
autocast_ctx = torch.autocast("cuda", torch.bfloat16) if USE_BF16 and torch.cuda.is_available() else nullcontext
# ---


@torch.inference_mode()
def get_channel_covariance(
        loader: DataLoader,
        models: list[UNetPipeline],
        max_batches: int, 
        device: str,
        seed: int,
        use_bottleneck: bool,
        reps: int | None = None,
) -> tuple[float, float]:

    """
    models: list different seeds of the same model 
    (like base_reseed1, base_reseed2 etc.)

    max_batches: number of batches to use from the dataloader

    reps: (optional) number of times to repeat going 
          through all max_batches batches

    returns mean, std (sample standard dev)
    std is None if there is only one model in models (single seed)
    """

    total_samples = [] # list of collected samples, lives on cpu

    # phase 1: collect samples
    for model in models:
        current_samples = []

        for j in range(reps):
            gen = torch.Generator(device).manual_seed(seed+j)

            for i, (image, label) in enumerate(tqdm(loader, total=max_batches)):
                if i == max_batches:
                    break

                # move stuff to device
                image, label = image.to(device), label.to(device)
                B = image.shape[0]

                # sample timestep uniformly
                t = torch.rand((B,), device=device, generator=gen)
                _t = t.view(-1, 1, 1, 1)

                # sample noise
                x_0 = torch.randn(image.shape, generator=gen, device=device)

                # cond flow
                x_t = _t * image + (1 - _t) * x_0

                with autocast_ctx:
                    h, skips, emb = encode(model, x_t, t, label, use_bottleneck=use_bottleneck) 

                # standardize across batches
                h_pool = h.mean(dim=(2, 3))
                current_samples.append(h_pool.cpu())

        total_samples.append(current_samples)

    # samples have been collected
    avg_covs_lst = []
    for seeded in total_samples:

        # Flatten the list of batch tensors into a single 2D tensor of shape (Total_Samples, C)
        samplestensor = torch.cat(seeded, dim=0).to(device)
        print(f"A total of {samplestensor.shape[0]} samples of dim {samplestensor.shape[1]} have been collected.")

        # Globally standardize across all samples so covariance equals correlation
        samplestensor = (samplestensor - samplestensor.mean(dim=0)) / (samplestensor.std(dim=0) + 1e-5)

        # transpose bcuz torch.cov wants (variables, observations)
        cov = torch.cov(samplestensor.T, correction=0) 

        # we want the off-diagonal, we dont care about the individual variable's variances
        # absolute off-diagonal correlation tells how much features influence each other in total
        off_diagonal = cov.abs().clone()
        off_diagonal = off_diagonal.fill_diagonal_(0)

        # this matrix has C(C-1) off-diagonal pairs
        # lets get the average absolute covariance per channel pair
        C = samplestensor.shape[1]
        avg_cov_per_pair = off_diagonal.sum().item() / (C*(C-1))

        avg_covs_lst.append(avg_cov_per_pair)

    if len(avg_covs_lst) == 1: # only one seed, no stddev avaiable
        return avg_covs_lst[0], None

    else:
        temp = torch.tensor(avg_covs_lst)
        return temp.mean().item(), temp.std().item()

    


def main():

    DEVICE = "cuda"
    MAX_BATCHES = 100 # 96 * 100 = 9600
    REPS = 3
    SEED = 1709
    USE_BOTTLENECK = False


    loader = get_dataloader(config, "test")

    models_list = [seeds_of(m) for m in [BASE, KL_005, KL_01, KL_025, KL_05, KL_1]]

    models_list2 = [seeds_of(m) for m in [BASE, IP_CUSTOM_01, BIP_0025, BIP_005, BIP_01, BIP_02, BIP_04]]

    suffix_to_name = SUFFIX_TO_NAME

    results = {}
    for models in models_list:

        modules = []
        for model in models:
            raw, ema_model = load_models(model, DEVICE)
            modules.append(ema_model)

        mean, std = get_channel_covariance(
            loader,
            modules,
            MAX_BATCHES,
            DEVICE,
            SEED,
            USE_BOTTLENECK,
            REPS
        )
        model_key = models[0]
        results[model_key] = {"mean": mean, "std": std}
        print(f"Model {models[0]} ({len(models)} reseeds): mean: {mean}, std: {std if std is not None else '---'}")


    # --- Bar Plot Visualization ---
    # plot results on bar plot, models with std other than None get an error bar
    labels = list(results.keys())
    means = [results[k]["mean"] for k in labels]
    stds = [results[k]["std"] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    bars = ax.bar([suffix_to_name.get(l, l) for l in labels], means, color='skyblue', edgecolor='black', alpha=0.8, capsize=5)

    # Add error bars only for models where std is available (not None)
    for i, bar in enumerate(bars):
        if stds[i] is not None:
            ax.errorbar(bar.get_x() + bar.get_width() / 2, means[i], yerr=stds[i],
                        fmt='none', ecolor='black', capsize=5, elinewidth=1.5)

    ax.set_ylabel("Average Absolute Off-Diagonal Covariance")
    ax.set_title("Channel Covariance Across Model Configurations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()