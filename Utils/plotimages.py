import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from pathlib import Path

def plot_images(
    images: torch.Tensor,
    label: str,
    normalize: bool = False,
    clamp: bool = True,
    compact: bool = False,
    save_dir: str | Path | None = None,
    filename: str | None = None,
):
    """Input: (B,C,H,W), Plots the images."""
    assert images.dim() == 4, f"Images must be of shape B,C,H,W but found {images.shape}"

    if normalize:
        images = images.clone() # avoid modifying original tensor
        min_vals = images.view(images.size(0), -1).min(dim=1)[0].view(-1,1,1,1) # dont norm over batch
        max_vals = images.view(images.size(0), -1).max(dim=1)[0].view(-1,1,1,1)
        images = (images - min_vals) / (max_vals - min_vals + 1e-8)  # add epsilon to avoid div by 0

    if clamp:
        images = torch.clamp(images, min=0, max=1)


    B = images.shape[0]
    cols = math.ceil(math.sqrt(B))
    rows = math.ceil(B/cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(2*cols, 2*rows), squeeze=False)
    fig.canvas.manager.set_window_title(label)

    axs = axs.flatten()

    for idx, image in enumerate(images):
        # C,H,W --> H,W,C
        image = image.permute(-2,-1,0)

        ax = axs[idx]
        ax.imshow(image.cpu().float())
        if compact:
            ax.axis("off")
        else:
            ax.set_title(f"#{idx+1}")

    # Hide unused subplots
    for i in range(B, len(axs)):
        axs[i].axis("off")

    if compact:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    else:
        plt.grid()
        plt.tight_layout()

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            raise ValueError("filename is required when save_dir is provided")
        fig.savefig(save_path / filename, dpi=300, bbox_inches="tight", pad_inches=0 if compact else 0.1)

    plt.show()

