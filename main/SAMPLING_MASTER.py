import torch

from datetime import datetime
import random
import numpy as np
import dill

import tkinter as tk
from tkinter import ttk
from Utils.plotimages import plot_images

from Utils.loadmodel import load_models

from Utils.sample import EulerODEsolver, RK4_ODEsolver
from Utils.paths import OUTPUTS_DIR


def main():

    # create tk root for all interfaces
    global_root = tk.Tk()
    global_root.withdraw()



    # ====
    # SELECT MODEL HERE 
    # Change to load different model
    suffix = "KL_05"

    print(f"!!! CURRENT MODEL SUFFIX: {suffix}")
    # ====


    def loadpkl(root: str):
        with open(root, "rb") as f:
            return dill.load(f)

    model_output_dir = OUTPUTS_DIR / suffix
    root = {
        "config": model_output_dir / f"config_{suffix}.pkl",
        "analyzer": model_output_dir / f"analyzer_state_dict_{suffix}.pkl",
    }

    config = loadpkl(root["config"])
    print(f"!!! CURRENT MODEL'S USE_CONSTANT_LR: {config.use_constant_lr}")
    _device = config.device
    model, ema_model = load_models(suffix, device=_device)

    from ModelAnalyzerV2.hooks import ModelAnalyzer

    # functions for visualisation
    plus1div2 = lambda x: (x + 1) / 2
    minusmindiv2 = lambda x: (x - torch.min(x)) / 2


    # analyzer args
    if config.use_model_analyzer:
        analyzer_args = {
            "root": global_root,
            "model": model,
            "ema_decay": config.ema_decay_analyzer,
            "max_history_size": config.max_history_size,
            "tensorlogger_interval": config.tensorlogger_interval,
            "tracking_funcs": None, # use the GUI to select which tracking funcs to use
            "disable_prints": True # disable printing to avoid conflict with tqdm progress bar # TODO (remember that this is here)
        }
        analyzer = ModelAnalyzer(**analyzer_args)
        analyzer.load_state_dict(
            loadpkl(root["analyzer"]),
            modules_to_skip=config.modules_to_ignore,
        )
        analyzer.open_interface(root=global_root)



# !! SAMPLING / INFERENCE (interface made with gemini)
    model.eval()
    ema_model.eval()

    def set_seed(seed_val: int):
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @torch.no_grad()
    def generate(use_EMA: bool, do_CFG: bool, sample_processing_type: str, display_type: str, classidx: int | None, cfg_guidance_scale: float, n_samples: int, n_steps: int, solver_type: str, seed_str: str, one_per_class: bool = False, compact: bool = False):
        if one_per_class:
            class_label = "One Per Class"
            classidx = None
        else:
            class_label = f"Classidx {int(classidx) if classidx is not None else -1}"
            classidx = int(classidx) if classidx is not None else -1

        # Optionalen Seed anwenden, falls das Feld im UI nicht leer ist
        if seed_str.strip():
            try:
                chosen_seed = int(seed_str.strip())
                set_seed(chosen_seed)
                print(f"Seed fixed to: {chosen_seed}")
            except ValueError:
                print("Warning: Invalid seed input, using random generation instead.")

        # Dynamische Auswahl des Solvers basierend auf dem Dropdown
        current_model = ema_model if use_EMA else model
        if solver_type == "Euler":
            samples = EulerODEsolver(config, _device, current_model,
                                     n_steps=n_steps, classid=classidx, n_samples=n_samples,
                                     do_cfg=do_CFG, cfg_strength=cfg_guidance_scale,
                                     one_per_class=one_per_class)
        elif solver_type == "RK4":
            samples = RK4_ODEsolver(config, _device, current_model, use_RK45=False,
                                    n_steps=n_steps, classid=classidx, n_samples=n_samples,
                                    do_cfg=do_CFG, cfg_strength=cfg_guidance_scale,
                                    one_per_class=one_per_class)
        elif solver_type == "RK45":
            samples = RK4_ODEsolver(config, _device, current_model, use_RK45=True,
                                    n_steps=n_steps, classid=classidx, n_samples=n_samples,
                                    do_cfg=do_CFG, cfg_strength=cfg_guidance_scale,
                                    one_per_class=one_per_class)

        if sample_processing_type == "plus1div2":
            samples_viz = plus1div2(samples)
        elif sample_processing_type == "minusmindiv2":
            samples_viz = minusmindiv2(samples)
        
        if display_type == "clamp":
            display_label = f"{class_label} Samples"
            plot_images(
                samples_viz,
                display_label,
                normalize=False,
                clamp=True,
                compact=compact,
                save_dir=model_output_dir / "samples",
                filename=f"samples_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.png",
            )
        elif display_type == "normalize":
            display_label = f"{class_label} Samples"
            plot_images(
                samples_viz,
                display_label,
                normalize=True,
                clamp=False,
                compact=compact,
                save_dir=model_output_dir / "samples",
                filename=f"samples_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.png",
            )



    # region | tkinter sampling UI
    # Settings dict
    settings = {}

    # CIFAR-10 classes
    dataset_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]


    # Tkinter
    root = tk.Tk()
    root.deiconify()
    root.title("Diffusion Model Interface")
    root.geometry("420x680") # Höhe leicht vergrößert für die neuen Elemente

    # ---------- Helper ----------
    def update_settings_and_generate():
        settings["use_EMA"] = use_ema_var.get()
        settings["do_CFG"] = do_cfg_var.get()
        settings["compact"] = compact_var.get()

        settings["sample_processing_type"] = processing_var.get()
        settings["display_type"] = display_var.get()

        class_name = class_var.get()
        settings["one_per_class"] = class_name == "<one of each class>"
        if settings["one_per_class"]:
            settings["classidx"] = None
        else:
            settings["classidx"] = -1 if class_name == "<null>" else dataset_classes.index(class_name)

        settings["cfg_guidance_scale"] = round(cfg_scale_var.get(), 1)
        settings["n_samples"] = n_samples_var.get()
        settings["n_steps"] = n_steps_var.get()
        
        # Neue Werte aus dem Dropdown und Seed-Entry auslesen
        settings["solver_type"] = solver_var.get()
        settings["seed_str"] = seed_var.get()

        print("Generating with settings:")
        for k, v in settings.items():
            print(f"{k}: {v}")
        print("-" * 40)

        generate(**settings)


    # --------------------------------------------------
    # Booleans
    bool_frame = ttk.LabelFrame(root, text="Booleans")
    bool_frame.pack(fill="x", padx=10, pady=5)

    use_ema_var = tk.BooleanVar(value=True)
    do_cfg_var = tk.BooleanVar(value=True)
    compact_var = tk.BooleanVar(value=False)

    ttk.Checkbutton(bool_frame, text="Use EMA", variable=use_ema_var).pack(anchor="w", padx=10)
    ttk.Checkbutton(bool_frame, text="Use CFG", variable=do_cfg_var).pack(anchor="w", padx=10)
    ttk.Checkbutton(bool_frame, text="Compact Images", variable=compact_var).pack(anchor="w", padx=10)
    # --------------------------------------------------
    # Dropdowns
    dropdown_frame = ttk.LabelFrame(root, text="Dropdowns")
    dropdown_frame.pack(fill="x", padx=10, pady=5)

    # Solver Type Dropdown
    solver_var = tk.StringVar(value="Euler")
    ttk.Label(dropdown_frame, text="Solver Algorithm").pack(anchor="w", padx=10)
    ttk.OptionMenu(
        dropdown_frame,
        solver_var,
        "Euler",
        "Euler",
        "RK4",
        "RK45"
    ).pack(fill="x", padx=10)

    # Sample processing type
    processing_var = tk.StringVar(value="plus1div2")
    ttk.Label(dropdown_frame, text="Sample Processing Type").pack(anchor="w", padx=10, pady=(8, 0))
    ttk.OptionMenu(
        dropdown_frame,
        processing_var,
        "plus1div2",
        "plus1div2",
        "minusmindiv2"
    ).pack(fill="x", padx=10)

    # Display type
    display_var = tk.StringVar(value="clamp")
    ttk.Label(dropdown_frame, text="Display Type").pack(anchor="w", padx=10, pady=(8, 0))
    ttk.OptionMenu(
        dropdown_frame,
        display_var,
        "clamp",
        "clamp",
        "normalize"
    ).pack(fill="x", padx=10)

    # Class selector
    class_var = tk.StringVar(value= "<null>")
    ttk.Label(dropdown_frame, text="Class").pack(anchor="w", padx=10, pady=(8, 0))
    ttk.OptionMenu(
        dropdown_frame,
        class_var,
        "<null>",
        "<null>",
        "<one of each class>",
        *dataset_classes
    ).pack(fill="x", padx=10)

    # --------------------------------------------------
    # Seed configuration
    seed_frame = ttk.LabelFrame(root, text="Seed Options")
    seed_frame.pack(fill="x", padx=10, pady=5)
    
    seed_var = tk.StringVar(value="")
    ttk.Label(seed_frame, text="Manual Seed (Leave empty for random)").pack(anchor="w", padx=10)
    ttk.Entry(seed_frame, textvariable=seed_var).pack(fill="x", padx=10, pady=5)

    # --------------------------------------------------
    # Sliders
    slider_frame = ttk.LabelFrame(root, text="Sliders")
    slider_frame.pack(fill="x", padx=10, pady=5)

    # ---------- CFG Guidance Scale (0.1 steps) ----------
    cfg_scale_var = tk.DoubleVar(value=2.0)
    cfg_display_var = tk.StringVar(value="2.0")

    def update_cfg_display(*_):
        cfg_display_var.set(f"{cfg_scale_var.get():.1f}")

    cfg_scale_var.trace_add("write", update_cfg_display)

    cfg_row = ttk.Frame(slider_frame)
    cfg_row.pack(fill="x", padx=10)

    ttk.Label(cfg_row, text="CFG Guidance Scale").pack(side="left")

    tk.Scale(
        slider_frame,
        from_=0.0,
        to=5.0,
        resolution=0.05,
        variable=cfg_scale_var,
        orient="horizontal"
    ).pack(fill="x", padx=10)

    # ---------- Number of Steps (INT ONLY) ----------
    n_steps_var = tk.IntVar(value=50)

    def snap_n_steps(value):
        n_steps_var.set(int(round(float(value))))

    steps_row = ttk.Frame(slider_frame)
    steps_row.pack(fill="x", padx=10, pady=(12, 0))

    ttk.Label(steps_row, text="Number of Steps").pack(side="left")

    tk.Scale(
        slider_frame,
        from_=1,
        to=200,
        orient="horizontal",
        command=snap_n_steps
    ).pack(fill="x", padx=10)

    # ---------- Number of Samples (INT ONLY) ----------
    n_samples_var = tk.IntVar(value=4)

    def snap_n_samples(value):
        n_samples_var.set(int(round(float(value))))

    samples_row = ttk.Frame(slider_frame)
    samples_row.pack(fill="x", padx=10, pady=(12, 0))

    ttk.Label(samples_row, text="Number of Samples").pack(side="left")

    tk.Scale(
        slider_frame,
        from_=1,
        to=64,
        orient="horizontal",
        command=snap_n_samples
    ).pack(fill="x", padx=10)

    # --------------------------------------------------
    # Generate Button
    ttk.Button(
        root,
        text="GENERATE",
        command=update_settings_and_generate
    ).pack(pady=15)


    root.mainloop()

    # endregion


if __name__ == "__main__":
    main()