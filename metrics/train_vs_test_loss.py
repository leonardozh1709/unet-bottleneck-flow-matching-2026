import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from Utils.data import get_dataset
from Utils.config import config
from Utils.loadmodel import load_models
from Utils.paths import PLOTS_DIR
from Utils.registry import BASE, BIP_0025, BIP_005, BIP_01, BIP_02, BIP_04, KL_005, KL_01, KL_025, KL_05, KL_1, name_of
from Utils.seeds import seeds_of


def get_loaders(batch_size, train_dataset, test_dataset) -> tuple[DataLoader, DataLoader]:
    """loaders without shuffling so each epoch sees the same image sequence
    returns (train_loader, test_loader)"""
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=4),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=4)
    )


@torch.inference_mode()
def train_test_loss_per_epoch(
        model,
        device,
        train_loader,
        test_loader,
        max_batches,
        epochs,
        n_bins,
        seed=1709,
):
    """one 'epoch' = one pass over max_batches batches of the SAME images with new t and x_0
    (t and x_0 are seeded per epoch, so every model sees exactly the same t and noise)

    relative gap = mean(test loss) / mean(train loss) - 1   (ratio of means, not mean of ratios)
    all returned arrays have one entry per epoch:
    returns gap_bins (epochs, n_bins), gap_overall (epochs,), train_loss (epochs,), test_loss (epochs,)"""

    model.eval()
    gap_bins = np.zeros((epochs, n_bins))
    gap_overall = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    for epoch in range(epochs):
        generator = torch.Generator(device).manual_seed(seed + epoch)
        all_t, all_tr, all_te = [], [], []

        for i, ((tr_img, tr_lbl), (te_img, te_lbl)) in enumerate(zip(tqdm(train_loader, total=max_batches, desc=f"epoch {epoch+1} / {epochs}", leave=False), test_loader)):

            if i == max_batches:
                break

            assert tr_img.shape == te_img.shape, f"Train {tr_img.shape} and test {te_img.shape} shape mismatch"
            B = tr_img.shape[0]

            tr_img, tr_lbl, te_img, te_lbl = tr_img.to(device), tr_lbl.to(device), te_img.to(device), te_lbl.to(device)

            # sample timestep and noise (same for train and test images)
            t = torch.rand((B,), generator=generator, device=device)
            t_before = t.clone()
            t_view = t.view(-1, 1, 1, 1)
            x_0 = torch.randn(tr_img.shape, generator=generator, device=device)

            # conditional flow
            x_t_tr = t_view * tr_img + (1 - t_view) * x_0
            x_t_te = t_view * te_img + (1 - t_view) * x_0

            # ground-truth VF
            v_tr = tr_img - x_0
            v_te = te_img - x_0

            tr_preds = model(x_t_tr, t, tr_lbl)
            te_preds = model(x_t_te, t, te_lbl)
            assert torch.equal(t, t_before), "the model changed t!"

            # per-sample losses
            all_t.append(t.cpu())
            all_tr.append((tr_preds - v_tr).pow(2).view(B, -1).mean(dim=1).cpu())
            all_te.append((te_preds - v_te).pow(2).view(B, -1).mean(dim=1).cpu())

        t_all = torch.cat(all_t).double()
        tr_all = torch.cat(all_tr).double()
        te_all = torch.cat(all_te).double()
        bin_indices = (t_all * n_bins).long().clamp(0, n_bins - 1)  # .long does floor op

        for b in range(n_bins):
            mask = bin_indices == b
            gap_bins[epoch, b] = (te_all[mask].mean() / tr_all[mask].mean() - 1).item()

        gap_overall[epoch] = (te_all.mean() / tr_all.mean() - 1).item()  # all t pooled = the training objective
        train_loss[epoch] = tr_all.mean().item()
        test_loss[epoch] = te_all.mean().item()

    return gap_bins, gap_overall, train_loss, test_loss


def evaluate_group(suffixes, device, train_loader, test_loader, max_batches, epochs, n_bins) -> dict:
    """evaluates every training seed of ONE model and averages over the epochs.
    every returned array has the seeds on the first axis"""
    gap_bins, gap_overall, train_loss, test_loss, eval_noise = [], [], [], [], []

    for suffix in suffixes:
        print(f"evaluating {suffix}")
        _, ema_model = load_models(suffix, device)
        g_bins, g_overall, tr_loss, te_loss = train_test_loss_per_epoch(
            ema_model, device, train_loader, test_loader, max_batches, epochs, n_bins
        )
        gap_bins.append(g_bins.mean(axis=0))  # average over epochs -> one value per bin for this seed
        gap_overall.append(g_overall.mean())
        train_loss.append(tr_loss.mean())
        test_loss.append(te_loss.mean())
        eval_noise.append(g_overall.std(ddof=1))  # std across epochs: only the noise from t and x_0
        del ema_model
        torch.cuda.empty_cache()

    return {
        "gap_bins": np.array(gap_bins),          # (seeds, n_bins)
        "gap_overall": np.array(gap_overall),    # (seeds,)
        "train_loss": np.array(train_loss),      # (seeds,)
        "test_loss": np.array(test_loss),        # (seeds,)
        "eval_noise": np.array(eval_noise),      # (seeds,)
    }


def seed_bars(ax, x, values_per_model: dict[str, np.ndarray]) -> None:
    """grouped bars: height = mean over training seeds, error bar = std across seeds (only if n > 1),
    black dots = the individual seeds. values_per_model: name -> array (seeds, len(x))"""
    width = 0.8 / len(values_per_model)

    for i, (name, values) in enumerate(values_per_model.items()):
        n = values.shape[0]
        offset = (i - len(values_per_model) / 2 + 0.5) * width
        std = values.std(axis=0, ddof=1) if n > 1 else None
        ax.bar(x + offset, values.mean(axis=0), width, yerr=std, capsize=4, label=f"{name} (n={n})")
        for seed_values in values:
            ax.scatter(x + offset, seed_values, color="black", s=10, zorder=3)

    ax.grid(axis='y', linestyle='--', alpha=0.7)


def calculate_and_plot(groups: dict[str, list[str]], device, train_loader, test_loader,
                       max_batches, epochs, n_bins, output_dir) -> None:
    """groups: model name -> list of training seed suffixes"""

    results = {name: evaluate_group(suffixes, device, train_loader, test_loader, max_batches, epochs, n_bins)
               for name, suffixes in groups.items()}

    # --- table ---
    print("\nmodel | n seeds | overall gap (mean +- std across seeds) | eval noise (std across epochs) | test loss")
    for name, r in results.items():
        n = len(r["gap_overall"])
        gap_std = f"{100 * r['gap_overall'].std(ddof=1):.3f}" if n > 1 else "n/a"
        print(f"{name} | {n} | {100 * r['gap_overall'].mean():+.3f} % +- {gap_std} % | "
              f"{100 * r['eval_noise'].mean():.3f} % | {r['test_loss'].mean():.5f}")

    # --- plot 1: gap per timestep bin ---
    plt.figure(figsize=(10, 6))
    seed_bars(plt.gca(), np.arange(n_bins), {name: r["gap_bins"] for name, r in results.items()})
    plt.xlabel('Timestep Bins', fontsize=12)
    plt.ylabel('Test Loss Relative Increase ((Test - Train) / Train)', fontsize=12)
    plt.title('Train vs Test Loss Increase Across Timestep Bins\n(mean +- std across training seeds, dots = single seeds)', fontsize=13)
    plt.legend(title='Models')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gap_per_bin.png")
    plt.close()

    # --- plot 2: overall gap and absolute test loss ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    seed_bars(axes[0], np.arange(1), {name: r["gap_overall"][:, None] for name, r in results.items()})
    axes[0].set_ylabel('Overall relative increase (Test - Train) / Train')
    seed_bars(axes[1], np.arange(1), {name: r["test_loss"][:, None] for name, r in results.items()})
    axes[1].set_ylabel('Mean test loss')
    for ax in axes:
        ax.set_xticks([])
    axes[0].legend(title='Models')
    fig.suptitle('Overall gap and absolute test loss (mean +- std across training seeds, dots = single seeds)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gap_overall_and_test_loss.png")
    plt.close()


def main(): # todo run ts

    EPOCHS = 10
    N_BINS = 20
    MAX_BATCHES = 100
    BATCH_SIZE = 96
    SESSION_TAG = "KL"
    OUTPUT_DIR = PLOTS_DIR / "train_vs_test_loss" / SESSION_TAG
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = "cuda"

    # model name -> training seeds (single-seed models have one entry)
    groups1 = {
        name_of(BASE): seeds_of(BASE),
        name_of(BIP_02): seeds_of(BIP_02),
        #name_of(KL_05): seeds_of(KL_05),
        name_of(BIP_0025): seeds_of(BIP_0025),
        name_of(BIP_005): seeds_of(BIP_005),
        name_of(BIP_01): seeds_of(BIP_01),
        name_of(BIP_04): seeds_of(BIP_04),
    }

    groups = {
        name_of(BASE): seeds_of(BASE),
        name_of(KL_005): seeds_of(KL_005),
        name_of(KL_01): seeds_of(KL_01),
        name_of(KL_025): seeds_of(KL_025),
        name_of(KL_05): seeds_of(KL_05),
        name_of(KL_1): seeds_of(KL_1)

    }

    train_loader, test_loader = get_loaders(BATCH_SIZE, get_dataset(config, "train"), get_dataset(config, "test"))
    calculate_and_plot(groups, DEVICE, train_loader, test_loader, MAX_BATCHES, EPOCHS, N_BINS, OUTPUT_DIR)


if __name__ == "__main__":
    main()