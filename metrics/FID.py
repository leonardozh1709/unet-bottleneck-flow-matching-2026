import time
import torch
import numpy as np
from PIL import Image
from cleanfid import fid
from torchvision import datasets
from Utils.sample import EulerODEsolver, RK4_ODEsolver
from Utils.UNet_max import UNetMaxPipeline
from Utils.config import Config
from dataclasses import replace
from Utils.config import config as default_config
from Utils.paths import DATA_DIR, OUTPUTS_DIR

@torch.no_grad()
def calculate_fid_from_model(
    model: torch.nn.Module,
    sample_fn,  # Returns Tensor in [-1, 1] on GPU/CPU
    num_samples: int = 50000,
    batch_size: int = 256,
    dataset_name: str = "cifar10",
    dataset_res: int = 32,
    dataset_split: str = "train",
    device: str = "cuda"
) -> float:
    model.eval()

    # Clean-FID custom wrapper must return uint8 numpy arrays or 
    # uint8 torch Tensors formatted as (N, H, W, C) or (N, C, H, W)
    def custom_gen_wrapper(z):
        current_batch_size = z.shape[0]
        # Generate samples: Tensor in [-1, 1], shape (B, C, H, W)
        imgs = sample_fn(model, current_batch_size, device)
        
        # Convert [-1, 1] to uint8 [0, 255]
        imgs = ((imgs + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8)
        return imgs

    score = fid.compute_fid(
        gen=custom_gen_wrapper,
        dataset_name=dataset_name,
        dataset_res=dataset_res,
        dataset_split=dataset_split,
        mode="clean",
        num_gen=num_samples,
        batch_size=batch_size,
        device=torch.device(device)
    )
    return float(score)


def default_sample_fn(model, batch_size, device):
    return EulerODEsolver(
        config=default_config,
        device=device,
        model=model,
        n_steps=50, 
        classid=None, # random classes
        n_samples=batch_size,
        do_cfg=True,
        cfg_strength=2.0,
        use_tqdm=False
    )


def test_fid_on_dataset(num_samples: int = 50000, batch_size: int = 256):
    """Sanity test: Evaluates raw CIFAR-10 PIL images using custom_gen_wrapper."""
    
    # Load raw CIFAR-10 dataset (returns PIL Images directly without torchvision transforms)
    dataset = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    
    # Convert entire dataset to a single memory-mapped or contiguous uint8 tensor (N, C, H, W)
    data_imgs = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)  # (50000, 3, 32, 32)
    
    ptr = 0
    def dataset_sample_fn(model, batch_sz, device):
        nonlocal ptr
        end_ptr = ptr + batch_sz
        if end_ptr > len(data_imgs):
            ptr = 0
            end_ptr = batch_sz
            
        batch = data_imgs[ptr:end_ptr].to(device)
        ptr = end_ptr
        
        # Convert uint8 [0, 255] to float [-1, 1] expected by sample_fn contract
        return (batch.float() / 127.5) - 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_model = torch.nn.Identity()

    fid_score = calculate_fid_from_model(
        model=dummy_model,
        sample_fn=dataset_sample_fn,
        num_samples=num_samples,
        batch_size=batch_size,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        device=device,
    )

    print(f"Sanity Check Clean-FID (Ground Truth): {fid_score:.4f}")
    return fid_score


if __name__ == "__main__":
    from Utils.loadmodel import load_models

    model, ema_model = load_models("KL_05_reseed2", "cuda")
    _fid = calculate_fid_from_model(ema_model, default_sample_fn, num_samples=50000)
    print(f"FINAL FID50k: {_fid}")

    if False:
        def sample_fn_1(model, batch_size, device):
            return EulerODEsolver(
                config=default_config,
                device=device,
                model=model,
                n_steps=50,
                classid=None,
                n_samples=batch_size,
                do_cfg=True,
                cfg_strength=1.9,
                use_tqdm=False,
            )
        _fid = calculate_fid_from_model(ema_model, sample_fn=sample_fn_1, num_samples=10000, batch_size=128) 
        print(f"fid-10k with 50 step euler, cfg 1.9: {_fid}")


        def sample_fn_2(model, batch_size, device):
            return RK4_ODEsolver(
                config=default_config,
                device=device,
                model=model,
                use_RK45=False,
                n_steps=12,
                classid=None,
                n_samples=batch_size,
                do_cfg=True,
                cfg_strength=1.9,
                use_tqdm=False,
                last_step_euler=True
            )
        _fid = calculate_fid_from_model(ema_model, sample_fn=sample_fn_2, num_samples=10000, batch_size=128) 
        print(f"fid-10k with 12 step RK4, cfg 1.9: {_fid}")




        if False:

            # create missing file
            model_output_dir = OUTPUTS_DIR / "bottleneck_IP_005_constant_lr"

            fid_dict_path = model_output_dir / "training_bottleneck_IP_005_2026-09-25_06-51-10" / "fid_dict.pt"
            fid_dict = torch.load(fid_dict_path)

            final_fid_50k = _fid  

            final_fid_25k = list(fid_dict.values())[-1] if fid_dict else None
            fid_results_path = model_output_dir / f"fid_results_{suffix}.txt"

            fid_25k_text = f"{final_fid_25k:.4f}" if final_fid_25k is not None else "n/a"
            fid_50k_text = f"{final_fid_50k:.4f}"

            fid_results_path.write_text(
                f"Best FID-25k (last logged): {fid_25k_text}\n"
                f"Best FID-50k: {fid_50k_text}\n",
                encoding="utf-8",
            )