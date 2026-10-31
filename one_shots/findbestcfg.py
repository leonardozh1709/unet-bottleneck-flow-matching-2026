import torch
import numpy as np
from functools import partial
from metrics.FID import calculate_fid_from_model
from Utils.config import config
from Utils.sample import EulerODEsolver

# STAGE 1 CONFIG
CFG_SEARCH_SPACE = np.unique(
    np.concatenate([
        np.arange(1.0, 2.0, 0.1),   # [1.0, 1.1, 1.2, ..., 1.9]
        np.arange(2.0, 3.5, 0.25),  # [2.0, 2.25, 2.5, 2.75, 3.0, 3.25]
        np.arange(3.5, 5.5, 0.5)    # [3.5, 4.0, 4.5, 5.0]
    ])
).tolist()

S1_N_SAMPLES: int = 5000 
S1_ODE_STEPS: int = 10

# STAGE 2 (TOP 3 CANDIDATES)
S2_N_SAMPLES: int = 10000
S2_ODE_STEPS: int = 20

# STAGE 3 (FINAL EVAL)
S3_N_SAMPLES: int = 50000
S3_ODE_STEPS: int = 50 


def make_sample_fn(cfg_strength: float, n_steps: int):
    """Factory function that returns a sample_fn matching (model, batch_size, device)."""
    def sample_fn(model, batch_size, device):
        return EulerODEsolver(
            config=config,
            device=device,
            model=model,
            n_steps=n_steps,
            classid=None,  # random classes
            n_samples=batch_size,
            do_cfg=(cfg_strength != 1.0),
            cfg_strength=cfg_strength,
            use_tqdm=False
        )
    return sample_fn


if __name__ == "__main__":
    from pathlib import Path
    from Utils.config import Config
    from dataclasses import replace
    from Utils.config import config as default_config
    from Utils.paths import OUTPUTS_DIR
    from Utils.loadmodel import load_models

    # load mid-training suffix
    model, ema_model = load_models("base_mini64_constant_lr", "cuda")

    # STAGE 1
    print("--- Starting Stage 1 ---")
    print(f"Search space size: {len(CFG_SEARCH_SPACE)}")
    stage_1_results: dict[float, float] = {}

    for cfg_scale in CFG_SEARCH_SPACE:
        current_sample_fn = make_sample_fn(cfg_strength=cfg_scale, n_steps=S1_ODE_STEPS)
        
        fid = calculate_fid_from_model(
            model=ema_model,
            sample_fn=current_sample_fn,
            num_samples=S1_N_SAMPLES,
            batch_size=256
        )
        
        stage_1_results[cfg_scale] = fid
        print(f"[Stage 1] CFG Scale: {cfg_scale:.2f} | FID-{S1_N_SAMPLES//1000}k ({S1_ODE_STEPS} steps): {fid:.4f}")

    # Select top 3 candidates from Stage 1
    top_3_cfg = sorted(stage_1_results, key=stage_1_results.get)[:3]
    print(f"Top 3 CFG candidates from Stage 1: {top_3_cfg}")

    # STAGE 2
    print("--- Starting Stage 2 ---")
    stage_2_results: dict[float, float] = {}

    for cfg_scale in top_3_cfg:
        current_sample_fn = make_sample_fn(cfg_strength=cfg_scale, n_steps=S2_ODE_STEPS)
        
        fid = calculate_fid_from_model(
            model=ema_model,
            sample_fn=current_sample_fn,
            num_samples=S2_N_SAMPLES,
            batch_size=256
        )
        
        stage_2_results[cfg_scale] = fid
        print(f"[Stage 2] CFG Scale: {cfg_scale:.2f} | FID-{S2_N_SAMPLES//1000}k ({S2_ODE_STEPS} steps): {fid:.4f}")

    best_cfg = min(stage_2_results, key=stage_2_results.get)
    print(f"\nWinning CFG scale after Stage 2: {best_cfg}")



    # plotting the results (stage 1 and stage 2)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(list(stage_1_results.keys()), list(stage_1_results.values()), marker='o', linestyle='None', label=f'Stage 1: n_samples={S1_N_SAMPLES}, steps={S1_ODE_STEPS}')
    plt.plot(list(stage_2_results.keys()), list(stage_2_results.values()), marker='s', linestyle='None', label=f'Stage 2: n_samples={S2_N_SAMPLES}, steps={S2_ODE_STEPS}')
    plt.xlabel('CFG Scale')
    plt.ylabel('FID')
    plt.title('FID vs CFG Scale')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
