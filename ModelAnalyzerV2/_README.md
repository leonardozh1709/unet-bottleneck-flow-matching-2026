# ModelAnalyzer usage guide

ModelAnalyzer is a lightweight hook-based analysis utility for PyTorch modules. It records module-level statistics during forward and backward passes, computes EMA summaries, logs tensor values via `TensorLogger`, and can save/load its state for later inspection.
It can be used to inspect intermediate distributions and other model metrics.

The current lifecycle is intentionally manual:

1. Build the analyzer object.
2. Run a dummy forward pass once to collect static metadata and baseline execution information.
3. Attach the main forward/backward hooks before the training step you want to analyze.
4. Run the forward/backward/optimizer pass.
5. Remove the hooks immediately after the pass.

This keeps the analyzer fully decoupled from a training loop counter.
**In my projects I usually fire the modelanalyzer hooks at every 10th loop iteration for the full training, this results in approximately a 3.33% total slowdown over the course of training. **

If you want to run the bundled example code immediately, you can execute the module directly:

```bash
python -m ModelAnalyzerFull.hooks
```

That entry point runs the built-in demo logic defined in `hooks.py` itself. The separate `test.py` file only contains the tiny example model used by that demo, so it can be safely deleted if you are not using the example code.

## Basic construction

If you are only integrating the analyzer into your own project and do not need the sample network from the demo, you can remove `test.py` without affecting the main analyzer API or the core hook-based analysis workflow.

```python
from ModelAnalyzerFull.hooks import ModelAnalyzer

analyzer = ModelAnalyzer(
    root=root_window,              # tkinter root for GUI selection, or any object if tracking_funcs is supplied
    model=model,
    ema_decay=0.9,
    max_history_size=None,
    tensorlogger_interval=100,
    tracking_funcs=None,           # None => GUI opens to let you choose tracking funcs
)
```

If you do not want the GUI selection popup, provide your own tracking function dictionary explicitly:

```python
tracking_funcs = {
    "mean": lambda x: x.mean(),
    "std": lambda x: x.std(),
    "max": lambda x: x.max(),
}

analyzer = ModelAnalyzer(
    root=root_window,
    model=model,
    ema_decay=0.9,
    max_history_size=None,
    tensorlogger_interval=100,
    tracking_funcs=tracking_funcs,
)
```

## One-time setup before training

Call this once before training begins:

```python
dummy_inputs = [torch.randn(B, C, H, W, device=device)]
analyzer.dummy_forward_pass(
    dummyinputs=dummy_inputs,
    gpu_estimate_iterations=5,
    runtime_memory_iterations=5,
    quick_hook_iterations=1,
)
```

What this does:

- traces execution order using `torch.fx`
- estimates per-pass runtime / memory statistics
- records tensor shapes and parameter counts for each module
- initializes analyzer metadata used by the hook system

## Where to attach and remove hooks in a training loop

The key pattern is:

- `attach_main_hooks(...)` immediately before the pass you want to analyze
- `remove_main_hooks()` immediately after the optimizer step

Example:

```python
for step, (images, labels) in enumerate(loader):
    # 1) Decide explicitly when to analyze
    if step % config.update_interval == 0:
        analyzer.attach_main_hooks(
            modules_to_ignore=(nn.Dropout, nn.Identity),
            skip_containers=False,
            track_history=True,
            track_gradients=True,
            additional_attributes=None,
            verbose=False,
        )

    optimizer.zero_grad(set_to_none=True)

    # 2) Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 3) Backward pass
    loss.backward()
    optimizer.step()

    # 4) Remove hooks immediately after the analyzed pass
    analyzer.remove_main_hooks()
```

## Example training loop with logging

```python
import torch
import torch.nn as nn

model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

analyzer = ModelAnalyzer(
    root=root_window,
    model=model,
    ema_decay=0.9,
    max_history_size=None,
    tensorlogger_interval=50,
    tracking_funcs=None,
)

# Run once before training to initialize the analyzer's static metadata.
dummy_inputs = [torch.randn(2, 3, 32, 32, device=device)]
analyzer.dummy_forward_pass(dummyinputs=dummy_inputs)

for step, (images, labels) in enumerate(loader):
    if step % config.update_interval == 0:
        analyzer.attach_main_hooks(
            modules_to_ignore=(nn.Dropout, nn.Identity),
            skip_containers=False,
            track_history=True,
            track_gradients=True,
            additional_attributes=None,
            verbose=False,
        )

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    # Optional: log a tensor manually for TensorLogger visualization
    analyzer.log_tensor(label="loss", value=loss.detach())

    analyzer.remove_main_hooks()

    if step % 100 == 0:
        print(f"step={step}, loss={loss.item():.4f}")
```

## Saving and restoring analyzer state

Once you have trained or inspected the model, you can save the analyzer state together with the model weights:

```python
state_dict = analyzer.get_state_dict()
# save to disk with your normal checkpoint format
```

When restoring later:

```python
analyzer = ModelAnalyzer(
    root=root_window,
    model=model,
    ema_decay=0.9,
    max_history_size=None,
    tensorlogger_interval=100,
    tracking_funcs=None,
)

analyzer.load_state_dict(saved_state_dict)
```

## Opening the GUI after collection

Once you have finished collecting analyzer data, open the explorer UI directly from the analyzer object:

```python
analyzer.open_interface(root=root_window)
```

That call launches the main tree-based module explorer window. The root window is the same tkinter root you already passed into `ModelAnalyzer(...)`.

## What the GUI shows

The main GUI is a recursive module tree of the model. Each node represents a submodule, and the tree is organized as follows:

- top-level summary node
  - total trainable params
  - total non-trainable params
  - total submodules
  - optional extra metadata such as estimated duration and energy
- each submodule node
  - `info` folder
    - `Technical info:`
      - runtime
      - memory used
    - `metadata:`
      - input shape
      - output shape
      - trainable / non-trainable param counts
      - forward and backward EMA stat folders
    - `Gradients:` if the module is a leaf module and gradient summaries exist
    - `[ Gradient plots (right click or Ctrl+click) ]`
    - `[ Stat plots (right click or Ctrl+click) ]`
    - `Attributes:` if any custom extra attributes were attached and EMA-smoothed

The tree uses the model hierarchy to expose the relationship between layers, modules, and their tracked statistics.

## The four windows in `app.py`

### 1. Main tree window
This is the primary explorer window where you browse the full model tree.

- Right-click or Ctrl-click on the `Stat plots` row for a module to open a plot selector for that module's recorded forward/backward stats.
- Right-click or Ctrl-click on the `Gradient plots` row to open gradient-flow plotting options for that module.

### 2. Stats-vs-depth window
This opens from the button labeled `Stats vs module operation order plots`.

It lets you select which tracked statistic to visualize across the model depth / execution order. The plot is built against the module execution order discovered by `dummy_forward_pass(...)`.

### 3. TensorLogger window
This opens from the button labeled `TensorLogger`.

It is used for plotting extra tensors that were manually logged with:

```python
analyzer.log_tensor(label="loss", value=loss.detach())
```

The window lists available logged labels and lets you pick which label(s) and which tracked statistic functions to plot.

### 4. Gradient flow window
This opens from the button labeled `Gradient Flow Plots`.

It provides two plot styles:

- gradient flow across all modules
- gradient flow grouped by module type

You can choose a metric such as `grad_weight_ratio`, `grad_norm`, `weight_norm`, or `mean_abs_grad`.

## Notes

- `attach_main_hooks(...)` should be called only around the exact pass you want to inspect.
- `remove_main_hooks()` is required so the hook overhead does not remain active during the rest of the training loop.
- `TensorLogger` is driven by `analyzer.log_tensor(...)` and is best used for selected scalar or tensor values you want to plot later.
- `add_kl_to_unit_gaussian()` can be used after collecting statistics if you want to derive a KL-to-unit-Gaussian history from tracked mean/std values.



Note:
The code for ModelAnalyzerFull was written by me. I used LLMs to help me write this file (_README.md).