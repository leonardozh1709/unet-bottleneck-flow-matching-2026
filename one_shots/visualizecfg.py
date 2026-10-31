
import torch

from Utils.config import config
from Utils.loadmodel import load_models
from Utils.plotimages import plot_images
from Utils.sample import EulerODEsolver
from Utils.paths import OUTPUTS_DIR


# Select the model, classes, and sampling settings here.
suffix = "base_mini64_constant_lr"
classidxs = [1, 9, 7, 8]
cfg_scales = [5, 40] #[1, 2, 3, 4, 6, 8, 12, 20]
seed = 12915 
n_steps = 50


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def main():
    device = config.device if torch.cuda.is_available() else "cpu"
    model, ema_model = load_models(suffix, device=device)
    ema_model.eval()

    output_dir = OUTPUTS_DIR / suffix / "samples"

    if len(classidxs) != 4:
        raise ValueError("classidxs must contain exactly four classes")

    for cfg_scale in cfg_scales:
        set_seed(seed)
        class_samples = []
        for classidx in classidxs:
            class_samples.append(EulerODEsolver(
                config=config,
                device=device,
                model=ema_model,
                n_steps=n_steps,
                classid=classidx,
                n_samples=4,
                do_cfg=True,
                cfg_strength=cfg_scale,
                use_tqdm=True,
            ))

        # Arrange four samples from each class into one 2x2 quadrant.
        samples = torch.stack([
            class_samples[0][0], class_samples[0][1], class_samples[1][0], class_samples[1][1],
            class_samples[0][2], class_samples[0][3], class_samples[1][2], class_samples[1][3],
            class_samples[2][0], class_samples[2][1], class_samples[3][0], class_samples[3][1],
            class_samples[2][2], class_samples[2][3], class_samples[3][2], class_samples[3][3],
        ])

        plot_images(
            (samples + 1) / 2,
            label=f"Classes {classidxs}, CFG {cfg_scale}",
            compact=True,
            save_dir=output_dir,
            filename=f"cfg_{cfg_scale:g}_classes_{classidxs[0]}-{classidxs[1]}-{classidxs[2]}-{classidxs[3]}_seed_{seed}.png",
            normalize=False,
            clamp=True,
        )



if __name__ == "__main__":
    main()