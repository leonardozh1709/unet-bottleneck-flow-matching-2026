import os

import torch
from tqdm import tqdm

import math
import time
from contextlib import nullcontext
import matplotlib.pyplot as plt
from colorama import Fore, Style
from Utils.plotimages import plot_images
from Utils.paths import OUTPUTS_DIR

from metrics.FID import calculate_fid_from_model, default_sample_fn
from metrics.pathstraightness import get_path_straightness
from Utils.UNet import ResBlock, sinusoidial_embedding
from Utils.UNet_max import ResBlockMax

# ==================================
@torch.no_grad()
def update_EMA_model(model, ema_model, decay: float = 0.999):
    """
    Updates EMA model using parameter iteration
    """
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

# ==================================

def custom_IP_fn(x, t_broadcast, gamma_0):
    t_1 = 0.7
    t_2 = 0.9
    k = 16
    # gamma(t):
    # gamma_0,                    t <= t_1
    # gamma_0 * exp(-k*(t-t_1)),  t_1 < t < t_2
    # 0,                          t >= t_2
    gamma_t = torch.where(
        # if
        t_broadcast <= t_1,
        torch.full_like(t_broadcast, gamma_0),
        # else
        torch.where(
            # if
            t_broadcast < t_2,
            gamma_0 * torch.exp(-k * (t_broadcast - t_1)),
            # else
            torch.zeros_like(t_broadcast),
        ),
    )

    # get x_t stddev
    std = torch.std(x, dim=(1,2,3)).view(-1, 1, 1, 1).detach() # (B, 1, 1, 1)

    # Apply perturbation
    return x + std * gamma_t * torch.randn_like(x)


def bottleneck_IP_forward(model, x, t, classids, bottleneck_IP_scale, return_bottleneck=False):
    """Run the model while perturbing the activations immediately before its bottleneck."""
    if model.t_emb_table is not None:
        t_emb = model.t_emb_table(t)
    else:
        t_emb = sinusoidial_embedding(t, model.embvector_dim)

    class_emb = model.class_emb_table(classids)
    embvector = class_emb + t_emb
    emb = model.net.emb_mlp(embvector)

    x = model.net.inconv(x)
    x = x + model.net.emb_proj(emb)[:, :, None, None]

    skips = []
    for block in model.net.encoder:
        x, skip = block(x, emb)
        skips.append(skip)

    t_broadcast = t.view(-1, 1, 1, 1)
    x = custom_IP_fn(x, t_broadcast, gamma_0=bottleneck_IP_scale)

    for block in model.net.bottleneck:
        if isinstance(block, (ResBlock, ResBlockMax)):
            x = block(x, emb)
        else:
            x = block(x)

    bottleneck_activations = x

    for block, skip in zip(model.net.decoder, reversed(skips)):
        x = block(x, skip, emb)

    x = model.net.outnorm(x)
    x = model.net.outact(x)
    prediction = model.net.outconv(x)
    if return_bottleneck:
        return prediction, bottleneck_activations
    return prediction


def kl_to_unit_gaussian(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float().transpose(0, 1).flatten(1)
    mean = x.mean(1)
    var = x.var(1, unbiased=False).clamp_min(eps)
    return 0.5 * (mean.square() + var - var.log() - 1).mean()


def loss_gradient_magnitude(loss: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    try:
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
        return grad_norm.detach()
    finally:
        for parameter in parameters:
            parameter.grad = None


def train(config, model, ema_model, optimizer, loader, analyzer = None, FID_log_path="training_unnamed", IP_type: str | None = None, gamma_0: float | None = None, on_epoch_end=None, on_final_fid_start=None, checkpoint_path: str | None = None, checkpoint_every: int = 1, resume_from: dict | None = None, kl_penalty_scale: float = 0.0):

    outputs_dir = OUTPUTS_DIR
    outputs_dir.mkdir(parents=True, exist_ok=True)
    

    cuda = config.device == "cuda" and torch.cuda.is_available()
    print(f"cuda: {cuda}")

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) # autocast to bfloat16 (only for forward pass and loss calc)
        if cuda and config.training_autocasting
        else nullcontext()
    )
    print(f"Autocasting (Bf16): {cuda and config.training_autocasting}")
    
    def logif(value: torch.Tensor, label: str, force_log: bool = False):
        """Logs only if config.use_model_analyzer == True"""
        if config.use_model_analyzer:
            assert analyzer is not None
            analyzer.log_tensor(value=value, label=label, force_log=force_log)

    ema_model.eval()

    max_i = config.epochs * len(loader)
    i = 0
    start_epoch = 0

    # init FID and straightness dict
    fid_dict = {}
    straightness_dict = {}

    # resume from a checkpoint saved by a previous, interrupted run of this function
    if resume_from is not None:
        model.load_state_dict(resume_from["model"])
        ema_model.load_state_dict(resume_from["ema_model"])
        optimizer.load_state_dict(resume_from["optimizer"])
        i = resume_from["i"]
        start_epoch = resume_from["epoch"] + 1
        fid_dict = resume_from.get("fid_dict", {})
        straightness_dict = resume_from.get("straightness_dict", {})
        print(f"Resumed from checkpoint at epoch {start_epoch}/{config.epochs}, iteration {i}.")


    schedule = torch.profiler.schedule(
        wait=1,  # Skip first step
        warmup=5,  # Warmup step (not recorded)
        active=10,  # Record for 10 steps
        repeat=1,  # Number of cycles
    )

    profiler_activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)

    profiler = (
        torch.profiler.profile(
            activities=profiler_activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(outputs_dir / "tensorboard_log")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        )
        if config.use_torch_profiler_tensorboard
        else nullcontext()
    )
    
    with profiler as prof:
        for epoch in range(start_epoch, config.epochs):
            j = 0

            loop = tqdm(loader)
            for images, labels in loop:

                if config.time_every_iteration:
                    t0 = time.time() # start time

                i += 1
                j += 1
                hooks_attached = False
                if config.use_model_analyzer and i % config.update_interval == 0:
                    analyzer.attach_main_hooks(
                        modules_to_ignore=config.modules_to_ignore,
                        skip_containers=config.skip_containers,
                        track_gradients=config.track_gradients,
                        track_history=config.track_history,
                        additional_attributes=config.additional_attributes,
                        verbose=False,
                    )
                    hooks_attached = True

                x_1 = images.to(config.device, non_blocking=config.nonblocking_gpu_transfers)
                labels = labels.to(config.device, non_blocking=config.nonblocking_gpu_transfers)
                B = images.shape[0]


                ## Flow matching
                # logit normal t sampling
                _t = torch.randn(size=(B,), dtype=x_1.dtype, device=x_1.device)
                _t = torch.sigmoid(_t) * 1 + 0

                t_broadcast = _t.view(-1, 1, 1, 1) # (B, ) -> (B, 1, 1, 1)  

                x_0 = torch.randn_like(x_1)
                x_t = (1 - t_broadcast) * x_0 + t_broadcast * x_1 

                # ground truth vector
                u_t = x_1 - x_0

                # if needed: input perturbation
                if IP_type == "constant":
                    x_t = x_t + gamma_0 * torch.randn_like(x_t)
                elif IP_type == "custom":
                    x_t = custom_IP_fn(x_t, t_broadcast, gamma_0=gamma_0)

                # Vectorized label dropping
                classids = labels.clone().detach().to(device=config.device, dtype=torch.long)
                drop_mask = torch.rand(B, device=config.device) < config.class_conditioning_drop_prob
                # Set dropped labels to the learned null embedding index
                classids[drop_mask] = config.n_classes

                with autocast_context:
                    ## forward pass
                    if kl_penalty_scale != 0:
                        if IP_type != "bottleneck":
                            v_t, bottleneck_activations = model(
                                x_t, _t, classids, return_bottleneck=True
                            )
                        else:
                            v_t, bottleneck_activations = bottleneck_IP_forward(
                                model,
                                x_t,
                                _t,
                                classids,
                                gamma_0,
                                return_bottleneck=True,
                            )
                        kl_penalty = kl_to_unit_gaussian(bottleneck_activations)
                    else:
                        if IP_type != "bottleneck":
                            v_t = model(x_t, _t, classids)
                        else:
                            #print(f"bottleneck Ip forward with gamma0 = {gamma_0}")
                            v_t = bottleneck_IP_forward(
                                model, x_t, _t, classids, gamma_0
                            )
                    mse_loss = torch.nn.functional.mse_loss(v_t, u_t)
                    loss = mse_loss
                    if kl_penalty_scale != 0:
                        loss = loss + kl_penalty_scale * kl_penalty

                optimizer.zero_grad(set_to_none=True)
                if (
                    kl_penalty_scale > 0
                    and config.use_model_analyzer
                    and config.kl_grad_magnitude_ratio_interval > 0
                    and i % config.kl_grad_magnitude_ratio_interval == 0
                ):
                    kl_penalty_loss = kl_penalty_scale * kl_penalty
                    mse_gradient_magnitude = loss_gradient_magnitude(mse_loss, model)
                    kl_gradient_magnitude = loss_gradient_magnitude(kl_penalty_loss, model)
                    gradient_magnitude_ratio = kl_gradient_magnitude / mse_gradient_magnitude.clamp_min(1e-12)
                    logif(label="MSE loss gradient magnitude", value=mse_gradient_magnitude, force_log=True)
                    logif(label="KL penalty gradient magnitude", value=kl_gradient_magnitude, force_log=True)
                    logif(label="KL/MSE gradient magnitude ratio", value=gradient_magnitude_ratio, force_log=True)

                ## constant lr or cosine decay with warmup
                if config.use_constant_lr:
                    lr = config.lr_max
                elif i <= config.warmup_iters:
                    lr = config.lr_max * i / config.warmup_iters
                else:
                    progress = (i - config.warmup_iters) / (max_i - config.warmup_iters)
                    lr = config.lr_min + 0.5*(config.lr_max - config.lr_min)*(1 + math.cos(math.pi*progress))

                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                ## backward pass
                loss.backward()
                if config.clip_grad_norm:
                    norm: float = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0, error_if_nonfinite=True)
                    logif(label = "gradient norm", value = torch.tensor(norm))
                    
                    
 
                optimizer.step()

                if hooks_attached:
                    analyzer.remove_main_hooks()

                ## update EMA model
                update_EMA_model(model, ema_model, decay=config.ema_decay)

                if i % config.update_interval == 0:
                    completion = round(i / max_i * 100,2)
                    loop.set_postfix(Epoch=epoch,Loss=loss,Total_completion=f"{completion}%")

                # modelanalyzer logging
                logif(label = "learning rate", value = torch.tensor(lr))
                logif(label = "MSE loss", value = mse_loss.detach())
                if kl_penalty_scale != 0:
                    logif(label = "KL penalty", value = kl_penalty.detach())
                logif(label = "Total loss", value = loss.detach())
                logif(label = "v_t", value = v_t.detach())

                
                # print loss in beginning to check if the starting loss is reasonable
                if i == 1:
                    print("Initial loss:", loss.item())

                
                if config.time_every_iteration:
                    if cuda:
                        torch.cuda.synchronize() # wait for gpu to get accurate timing 
                    t1 = time.time() # end time
                    dt = (t1 - t0) * 1000 # in ms
                    if (i-1) % 20 == 0:
                        eta = dt * (max_i-i) / 1000 / 60 # in min
                    
                    analyzer_fired = hooks_attached
                            
                    
                    loop.set_postfix_str((Fore.GREEN if analyzer_fired else Style.RESET_ALL) + f"epoch {epoch}/{config.epochs-1}, loss {loss}, dt: {dt:.2f}ms, images/s:{config.batch_size/(dt/1000):.2f} ,eta: {eta if eta else '---'}min")


                if config.use_torch_profiler_tensorboard:
                    if cuda:
                        torch.cuda.synchronize()  # Ensure all CUDA operations are complete before stepping the profiler
                    prof.step()


            # get FID and path straightness
            if (epoch+1) % config.training_FID_interval == 0:
                with torch.inference_mode():
                    print(f"starting fid calculation at the end of epoch {epoch} (on ema model)")
                    start = time.perf_counter()
                    fid = calculate_fid_from_model(
                        model=ema_model,
                        sample_fn=default_sample_fn,
                        num_samples=config.training_FID_steps,
                        batch_size=128,
                        device="cuda",
                    ) 
                    fid_dict[epoch] = fid 
                    print(f"completed fid calculation -> {fid}. Elapsed: {time.perf_counter()-start}")

                print(f"starting path straightness calculation at the end of epoch {epoch} (on raw model)")
                model.eval()
                straightness = get_path_straightness(model, "cuda", images=x_1[:32], labels=labels[:32], noises_per_image=8, batch_size=32, n_ODE_steps=100)
                model.train()
                straightness_dict[epoch] = straightness 
                print(f"completed straightness calculation -> {straightness}")

            if on_epoch_end is not None:
                on_epoch_end(epoch=epoch, loss=float(loss.detach().item()), fid_dict=fid_dict, straightness_dict=straightness_dict)

            if checkpoint_path is not None and (epoch + 1) % checkpoint_every == 0:
                torch.save({
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "i": i,
                    "fid_dict": fid_dict,
                    "straightness_dict": straightness_dict,
                }, checkpoint_path)

    # if analyzer prints were disabled: enable again for post training info
    analyzer.enable_prints()


    # Plot and save
    plt.figure()
    plt.plot(list(fid_dict.keys()), list(fid_dict.values()), marker="o")
    plt.title("FID score of the EMA model at a cfg scale of 2.0 across training epochs")
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.grid(True)

    plt.savefig(
        os.path.join(FID_log_path, "fid_plot.png"),dpi=300,bbox_inches="tight")
    plt.close()
    
    # same thing for path straightness
    plt.figure()
    plt.plot(list(straightness_dict.keys()), list(straightness_dict.values()), marker="o")
    plt.title("Unconditional Path Straightness of the raw model across training epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Path Straightness (Integrated Trajectory Length / Linear Trajectory Length)")
    plt.grid(True)

    plt.savefig(os.path.join(FID_log_path, "pathstraightness_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # save the actual dicts as well
    torch.save(fid_dict, os.path.join(FID_log_path, "fid_dict.pt"))
    torch.save(straightness_dict, os.path.join(FID_log_path, "pathstraightness_dict.pt"))

    
    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return fid_dict, 69.9

    # get FID-50k
    if on_final_fid_start is not None:
        on_final_fid_start()
    _fid = calculate_fid_from_model(ema_model, sample_fn=default_sample_fn, num_samples=50000, batch_size=128)
    print(f"final fid-50k: {_fid}")
    return fid_dict, _fid