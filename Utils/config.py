import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class Config:
    # Dataset / Data
    img_size: int = 32                                                          # training set images will be resized to this size (img_size x img_size)
    batch_size: int = 96                                                        # images per batch for training (lower this for higher resolutions)
    n_classes: int = 10                                                         # will be auto-set based on dataset in __post_init__

    # Device
    device: torch.device = "cuda"

    # Model Architecture
    time_emb_type: str = "sinusoidial"                                          # "sinusoidial" or "table"
    input_spatial: int = img_size                                               # spatial size of the input to the model (auto-set in __post_init__)
    ch_in: int = 3                                                              # input channels
    ch: int = 64                                                               # base internal (latent) channels
    ch_out: int = 3                                                             # output channels
    embvector_dim: int = 256                                                    # dimension of time & class embedding vectors
    n_layers: int = 3                                                           # number of UNet layers (per side)
    embedding_vector_mlp_layers: int = 1                                        # MLP layers for AdaGN embedding projection
    att_depth: int = 2                                                         # layers without attention (counted from top)
    bottleneck_size: int = 2                                                    # resblock + attentionblock layers in bottleneck
    double_channels_in_bottleneck: bool = False                                 # WARNING: explodes parameter count
    verbose: bool = False                                                       # print UNet architecture during init
    dropout: float = 0.1                                                        # dropout probability
    activation_function: type = nn.SiLU                                         # activation function in resblocks
    do_zero_init: bool = True                                                   # zero-init final convs for stable early training


    # Training
    epochs: int = 400 
    warmup_iters: int = 1000                                                    # warmup iterations
    lr_max: float = 3e-4                                                        # maximum learning rate
    lr_min: float = 1e-5                                                        # minimum learning rate
    use_constant_lr: bool = False                                               # use lr_max throughout training
    clip_grad_norm: bool = True                                                 # gradient clipping
    class_conditioning_drop_prob: float = 0.1                                   # CFG dropout probability
    ema_decay: float = 0.9995                                                    # EMA model decay

    # FID and path straightness 
    training_FID_interval: int = 80 # how many epochs to wait inbetween FID + path straightness checks
    training_FID_steps: int = 25000                                            # how many images per FID check

    # ModelAnalyzer Options
    update_interval: int = 75                                                   # interval for analyzer hook firing
    kl_grad_magnitude_ratio_interval: int = 200                                 # steps between KL/MSE gradient magnitude measurements; 0 disables
    ema_decay_analyzer: float = 0.9                                             # EMA decay for analyzer stats
    max_history_size: int | None = None                                         # max history size (None = unbounded)
    tensorlogger_interval: int = 75                                            # calls before tensor stats saved
    modules_to_ignore: tuple[type] = (nn.Dropout, nn.Identity)                  # ignored modules in hooks
    skip_containers: bool = False                                               # skip ModuleList/Sequential in hooks
    track_history: bool = True                                                  # track fwd/bwd stats over time
    track_gradients: bool = True                                                # track parameter gradients (EMA)
    additional_attributes: list[str] = None                                     # custom attributes to track
    gpu_estimate_iterations: int = 5                                            # iterations for GPU energy estimation
    runtime_memory_iterations: int = 5                                          # iterations for runtime/memory estimation

    # Performance Optimizations
    nonblocking_gpu_transfers: bool = True                                      # non-blocking data transfers to GPU
    extra_low_matmul_precision: bool = False                                    # use bfloat16 for matmuls
    training_autocasting: bool = True                                           # autocast to bfloat16 during training
    use_DSConvs: bool = False                                                   # use depthwise separable convolutions in bottleneck layers

    # Debug / Logging
    use_live_training_dashboard: bool = False                                   # live plots during training
    live_training_dashboard_start_epoch: int = 1                                # epoch index where livetrainingdashboard starts
    time_every_iteration: bool = False                                           # print timing per iteration
    use_model_analyzer: bool = True                                             # enable ModelAnalyzer
    use_cudnn_benchmark: bool = True                                            # Let cuDNN benchmark and pick the fastest kernel the batch size and input size (autotuner)
    use_torch_profiler_tensorboard: bool = False                                 # enables torch profiler and saves logs to view in tensorboard using python -m tensorboard.main --logdir=./tensorboard_log (WARNING: This slows training A LOT)



    def __post_init__(self):
        """Auto-set n_classes and input_spatial based on dataset and img_size."""
        self.n_classes = 10
        self.input_spatial = self.img_size


# Create config instance
config = Config()



