"""
Sequential multi-model training script.

- Writes progress to a text file (outputs/training_status.txt), overwritten
  once per epoch, sends email progress notifications.
- Periodically checkpoints mid-training (model, EMA, optimizer, step counter) so a
  crash doesn't cost the whole run. It resumes automatically from the last
  checkpoint.
- Each queue entry can override any Config field; anything not overridden uses
  config.py's current defaults.


Edit RUN_QUEUE at the bottom of this file, then run.
"""




# SHUT DOWN WHEN DONE AND SEND EMAIL REPORTS
SHUT_DOWN_WHEN_DONE = True
EMAIL_PROGRESS_INTERVAL_MINUTES = 60*3 # every 3 hours
END_OF_RUN_REST_TIME_MINUTES = 5 # rest for 5 minutes at the end of every run


import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()

GMAIL = os.getenv("GMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")

def sendmail(msg, subject):
    e = EmailMessage()
    e["From"] = GMAIL
    e["To"] = GMAIL
    e["Subject"] = subject
    e.set_content(msg)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL, APP_PASSWORD)
        s.send_message(e)


import time
import traceback
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import dill
import torch

from Utils.config import Config, config as default_config
from Utils.UNet import UNetPipeline
from Utils.UNet_max import UNetMaxPipeline
from main.training import train
from Utils.data import get_dataloader
from Utils.paths import OUTPUTS_DIR


STATUS_FILE = OUTPUTS_DIR / "training_status.txt"
EMAIL_ERROR_FILE = OUTPUTS_DIR / "email_errors.txt"


def notify(subject: str, msg: str):
    try:
        sendmail(msg, subject)
    except Exception:
        error_text = traceback.format_exc()
        try:
            EMAIL_ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(EMAIL_ERROR_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] Email delivery failed:\n{error_text}\n")
            print(f"Email delivery failed. Details written to {EMAIL_ERROR_FILE}")
        except Exception:
            print(f"Email delivery and local error logging both failed:\n{error_text}")


def write_status(text: str):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        f.write(text)


def build_optimizer(model, config):
    """no weight decay on norms/embeddings"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "embedding" in name.lower() or "emb_table" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.lr_max, fused=torch.cuda.is_available(),
    )


def train_one_model(
    suffix: str,
    config,
    IP_type=None,
    gamma_0=None,
    queue_position: str = "",
    kl_penalty_scale: float = 0.0,
):
    """Trains a single model end-to-end:
    setup, (resume if possible), train, save."""

    run_start = time.time()
    output_dir = OUTPUTS_DIR
    model_output_dir = output_dir / suffix
    model_output_dir.mkdir(parents=True, exist_ok=True)

    final_paths = {
        "model": str(model_output_dir / f"unet_{suffix}.pth"),
        "config": str(model_output_dir / f"config_{suffix}.pkl"),
        "analyzer": str(model_output_dir / f"analyzer_state_dict_{suffix}.pkl"),
    }
    # checkpoint, separate from the final save
    checkpoint_path = str(model_output_dir / f"checkpoint_{suffix}.pth")

    def status(line: str):
        elapsed_min = (time.time() - run_start) / 60
        write_status(
            f"=== Master Training Run ===\n"
            f"Current model: {suffix} {queue_position}\n"
            f"Elapsed on this model: {elapsed_min:.1f} min\n"
            f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"----------------------------------------\n"
            f"{line}\n"
        )

    status("Setting up dataloader and model...")

    # technical stuff
    torch.set_float32_matmul_precision("medium" if config.extra_low_matmul_precision else "high")
    if config.use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    loader = get_dataloader(config, split="train")

    device = config.device
    if not config.force_max_architecture:
        model = UNetPipeline(config).to(device) 
    else:
        model = UNetMaxPipeline(config).to(device)
        print("[FORCING MAX ARCHITECTURE]")

    ema_model = deepcopy(model).to(device)
    optimizer = build_optimizer(model, config)

    # !! resume support: if a checkpoint exists for this suffix, load it
    resume_from = None
    if Path(checkpoint_path).exists():
        status(f"Found existing checkpoint for {suffix}, resuming...")
        resume_from = torch.load(checkpoint_path, map_location=device)

    # !! ModelAnalyzer setup
    analyzer = None
    _hidden_root = None
    if config.use_model_analyzer:
        from ModelAnalyzerV2.hooks import ModelAnalyzer
        import tkinter as tk
        _hidden_root = tk.Tk()
        _hidden_root.withdraw() # created but never shown to not block anything


        epsilon = 1e-6
        TRACKING_FUNCS = {
            "mean": lambda x: x.mean(),
            "std": lambda x: x.std(),
            "zero_frac": lambda x: (x.abs() < epsilon).float().mean(), # count of values that are effectively zero (< epsilon)
        } 

        analyzer = ModelAnalyzer(
            root=_hidden_root,
            model=model,
            ema_decay=config.ema_decay_analyzer,
            max_history_size=config.max_history_size,
            tensorlogger_interval=config.tensorlogger_interval,
            tracking_funcs=TRACKING_FUNCS,
            disable_prints=True,
        )

        dummyinputs = [
            torch.randn(config.batch_size, config.ch_in, config.img_size, config.img_size, device=device),
            torch.rand((config.batch_size,), device=device),
            torch.randint(0, config.n_classes + 1, (config.batch_size,), device=device),
        ]
        analyzer.dummy_forward_pass(
            dummyinputs,
            gpu_estimate_iterations=config.gpu_estimate_iterations,
            runtime_memory_iterations=config.runtime_memory_iterations,
            quick_hook_iterations=1,
            modules_to_skip=config.modules_to_ignore,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_subdir = model_output_dir / f"training_{suffix}_{timestamp_str}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    last_progress_email_at = run_start

    def on_epoch_end(epoch, loss, fid_dict, straightness_dict):
        nonlocal last_progress_email_at
        latest_fid = f"{list(fid_dict.values())[-1]:.4f}" if fid_dict else "n/a"
        status(
            f"Epoch {epoch + 1}/{config.epochs}\n"
            f"Latest training loss: {loss:.5f}\n"
            f"Latest logged FID: {latest_fid}\n"
        )
        now = time.time()
        if now - last_progress_email_at >= EMAIL_PROGRESS_INTERVAL_MINUTES * 60:
            notify(
                f"Training Progress: {suffix} {queue_position}",
                f"Training progress: {suffix} {queue_position}\n"
                f"Epoch: {epoch + 1}/{config.epochs}\n"
                f"Latest training loss: {loss:.5f}\n"
                f"Latest logged FID: {latest_fid}\n"
                f"Elapsed on this model: {(now - run_start) / 3600:.1f} hours\n"
                f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            last_progress_email_at = now

    status("Starting training...")
    model.train()
    ema_model.eval()

    fid_dict, final_fid_50k = train(
        config=config,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        loader=loader,
        analyzer=analyzer,
        FID_log_path=str(output_subdir),
        IP_type=IP_type,
        gamma_0=gamma_0,
        kl_penalty_scale=kl_penalty_scale,
        on_epoch_end=on_epoch_end,
        on_final_fid_start=lambda: status("Training complete. Calculating final FID-50k..."),
        checkpoint_path=checkpoint_path,
        checkpoint_every=50,
        resume_from=resume_from,
    )

    final_fid_25k = list(fid_dict.values())[-1] if fid_dict else None
    fid_results_path = model_output_dir / f"fid_results_{suffix}.txt"
    fid_25k_text = f"{final_fid_25k:.4f}" if final_fid_25k is not None else "n/a"
    fid_50k_text = f"{final_fid_50k:.4f}"
    fid_results_path.write_text(
        f"Best FID-25k (last logged): {fid_25k_text}\n"
        f"Best FID-50k: {fid_50k_text}\n",
        encoding="utf-8",
    )
    status(
        f"Final FID-25k (last logged): {fid_25k_text}\n"
        f"Final FID-50k: {fid_50k_text}\n"
        f"Saved FID results to {fid_results_path.name}"
    )

    if analyzer is not None:
        if hasattr(analyzer, "main_hooks"):
            analyzer.remove_main_hooks()
        analyzer.add_kl_to_unit_gaussian()

    # !! final save
    torch.save({"model": model.state_dict(), "ema_model": ema_model.state_dict()}, final_paths["model"])
    with open(final_paths["config"], "wb") as f:
        dill.dump(config, f)
    if analyzer is not None:
        with open(final_paths["analyzer"], "wb") as f:
            dill.dump(analyzer.get_state_dict(), f)

    # training finished cleanly: the mid-training checkpoint is no longer needed
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    status(f"DONE. Total time this session: {(time.time() - run_start) / 60:.1f} min.")

    del model, ema_model, optimizer, loader
    if analyzer is not None:
        del analyzer
    if _hidden_root is not None:
        _hidden_root.destroy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "elapsed_minutes": (time.time() - run_start) / 60,
        "fid_25k": fid_25k_text,
        "fid_50k": fid_50k_text,
        "model_path": final_paths["model"],
    }


def run_queue(queue: list[dict], shutdown_when_done: bool = False):
    """
    queue: list of dicts, each with keys:
        "suffix":               str, unique model name / output folder (required)
        "config":               Config instance (optional, defaults to a copy of config.py's current state)
        "IP_type":              None | "constant" | "custom" | "bottleneck"
        "gamma_0":  None | float
        "kl_penalty_scale":     float, KL-to-unit-Gaussian loss multiplier (default 0.0)
    """
    n = len(queue)
    queue_start = time.time()
    results = []
    notify(
        "Training Queue Started",
        f"Training queue started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Models queued ({n}): {', '.join(entry.get('suffix', '<unknown>') for entry in queue)}\n"
        f"Progress emails are sent every {EMAIL_PROGRESS_INTERVAL_MINUTES} minutes during training."
    )
    for idx, entry in enumerate(queue):
        suffix = "<unknown>"
        model_start = time.time()
        try:
            suffix = entry["suffix"]
            config = entry.get("config", deepcopy(default_config))
            IP_type = entry.get("IP_type", None)
            gamma_0 = entry.get("gamma_0", None)
            kl_penalty_scale = entry.get("kl_penalty_scale", 0.0)
            write_status(f"=== Master Training Run ===\nQueued: model {idx + 1}/{n} ({suffix})\nStarting shortly...\n")

            model_result = train_one_model(
                suffix=suffix,
                config=config,
                IP_type=IP_type,
                gamma_0=gamma_0,
                kl_penalty_scale=kl_penalty_scale,
                queue_position=f"({idx + 1}/{n})",
            )

            # rest after the full run
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            time.sleep(60*END_OF_RUN_REST_TIME_MINUTES) # cooldown

            result = {
                "suffix": suffix,
                "status": "SUCCESS",
                **model_result,
            }
            results.append(result)
            notify(
                f"Training Complete: {suffix}",
                f"Training model finished successfully: {suffix} ({idx + 1}/{n})\n"
                f"Elapsed: {model_result['elapsed_minutes']:.1f} minutes\n"
                f"Final FID-25k: {model_result['fid_25k']}\n"
                f"Final FID-50k: {model_result['fid_50k']}\n"
                f"Model saved to: {model_result['model_path']}\n"
                f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception:
            error_trace = traceback.format_exc()
            error_text = (
                f"=== Master Training Run: ERROR ===\n"
                f"Model {suffix} ({idx + 1}/{n}) FAILED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"{error_trace}\n"
                f"A checkpoint (if any epochs completed) was saved and will be used to "
                f"resume automatically if you re-run this suffix.\n"
                f"Continuing with the next model in the queue in 10 seconds.\n"
            )
            results.append({
                "suffix": suffix,
                "status": "FAILED",
                "elapsed_minutes": (time.time() - model_start) / 60,
                "error": error_trace,
            })
            write_status(error_text)
            print(error_text)
            notify(f"Training Error: {suffix}", error_text)
            time.sleep(10)
            continue



    report_lines = [
        "Training queue report",
        f"Started: {datetime.fromtimestamp(queue_start).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total elapsed: {(time.time() - queue_start) / 3600:.2f} hours",
        f"Models completed: {sum(result['status'] == 'SUCCESS' for result in results)}/{n}",
        f"Models failed: {sum(result['status'] == 'FAILED' for result in results)}/{n}",
        "",
    ]
    for result in results:
        report_lines.append(
            f"{result['status']}: {result['suffix']} "
            f"({result['elapsed_minutes']:.1f} minutes)"
        )
        if result["status"] == "SUCCESS":
            report_lines.extend([
                f"  FID-25k: {result['fid_25k']}",
                f"  FID-50k: {result['fid_50k']}",
                f"  Model: {result['model_path']}",
            ])
        else:
            report_lines.append(f"  Error:\n{result['error']}")
    report = "\n".join(report_lines)
    notify("Training Queue Report", report)
    write_status(report)


    shutdown_message = "Shutdown was not requested."
    if shutdown_when_done:
        try:
            import subprocess
            subprocess.run(["shutdown", "/s", "/t", "60"], check=True)
            shutdown_message = "The computer will automatically shut down in 60 seconds."
        except Exception:
            shutdown_message = f"Automatic shutdown could not be scheduled:\n{traceback.format_exc()}"
    notify(subject="PC SHUTDOWN", msg=shutdown_message)
    


if __name__ == "__main__":

    # !! DEFINE TRAINING QUEUE HERE.
    # Each entry uses config.py's current defaults
    # override via replace(default_config, field=value, ...).
    mini_config = replace(default_config, epochs=400, training_FID_interval=80, ch=64, att_depth=2, ema_decay=0.9995)

    RUN_QUEUE_1 = [
        # base_max and best_IP_max
        {
            "suffix": "example123",
            "config": replace(mini_config, use_constant_lr=True), 
            "kl_penalty_scale": 0.5
        },
        {
            "suffix": "example456",
            "config": replace(mini_config, use_constant_lr=True), 
            "IP_type": "bottleneck",
            "gamma_0": 0.2, 
        }
    ]

    run_queue(RUN_QUEUE_1, shutdown_when_done=SHUT_DOWN_WHEN_DONE)

    # ============================================================
    # live progress
    # In windows powershell: Get-Content -Wait -Tail 20 outputs\training_status.txt
    # ============================================================
