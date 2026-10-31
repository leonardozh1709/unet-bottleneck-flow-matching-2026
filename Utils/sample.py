import torch
import torch.nn as nn
from tqdm import tqdm
from torchdiffeq import odeint

from contextlib import nullcontext

autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()


def call_model_no_cfg(model, x_t, t, c) -> torch.Tensor:
    """Calls the model without cfg"""
    with autocast_ctx:
        v_t = model(x_t, t, c)
    return v_t

def call_model_cfg(model, x_t, t, c, nullids, cfg_scale) -> torch.Tensor:
    """Calls the model with cfg, batched call for efficiency"""
    with autocast_ctx:
        v_cat = model(torch.cat([x_t, x_t], dim=0), 
                    torch.cat([t, t], dim=0), 
                    torch.cat([nullids, c], dim=0))
    v_uncond, v_cond = torch.chunk(v_cat, chunks=2, dim=0)
    v_t = v_uncond + cfg_scale * (v_cond - v_uncond)
    return v_t

def call_model(model, x_t, t, c, nullids, do_cfg, cfg_scale):
    if do_cfg:
        return call_model_cfg(model, x_t, t, c, nullids, cfg_scale)
    else:
        return call_model_no_cfg(model, x_t, t, c)



@torch.inference_mode()
def EulerODEsolver(config, device, model: nn.Module, n_steps: int, classid: int | None, n_samples: int, do_cfg: bool, cfg_strength: float, use_tqdm: bool = True, one_per_class: bool = False):
    """
    Set classid to None to use random classes for the whole batch.
    Set one_per_class=True to sample exactly one image for each class.
    """

    dt = 1 / n_steps

    if one_per_class:
        n_samples = config.n_classes
        c = torch.arange(config.n_classes, device=device, dtype=torch.long)
    elif classid is not None:
        c = torch.full((n_samples,), classid, device=device, dtype=torch.long)
    else:
        c = torch.randint(low=0, high=config.n_classes, size=(n_samples,), device=device, dtype=torch.long)

    nullids = torch.full((n_samples,), config.n_classes, device=device, dtype=torch.long)
    
    x_t = torch.randn((n_samples, 3, config.input_spatial, config.input_spatial), device=device)

    if use_tqdm:
        loop = tqdm(range(n_steps))
    else:
        loop = range(n_steps)

    # precompute tensor
    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)

    for i in loop:
        t = (dt_tensor * i).expand(n_samples)

        v_t = call_model(model, x_t, t, c, nullids, do_cfg, cfg_strength)

        # euler update step
        x_t = x_t + dt * v_t
    
    return x_t


@torch.inference_mode()
def RK4_ODEsolver(config, device, model: nn.Module, use_RK45: bool, n_steps: int, classid: int | None, n_samples: int, do_cfg: bool, cfg_strength: float, use_tqdm: bool = True, one_per_class: bool = False, tolerance: float = 1e-6, last_step_euler: bool = True):
    """
    Set classid to None to use random classes for the whole batch.
    Set one_per_class=True to sample exactly one image for each class.

    Set use_RK45 to True to use the adaptive RK45 solver from torchdiffeq.odeint
    - RK4: Takes 4x longer than euler.
    - RK45: Takes 6x longer than euler.
    """

    dt = 1 / n_steps

    if one_per_class:
        n_samples = config.n_classes
        c = torch.arange(config.n_classes, device=device, dtype=torch.long)
    elif classid is not None:
        c = torch.full((n_samples,), classid, device=device, dtype=torch.long)
    else:
        c = torch.randint(low=0, high=config.n_classes, size=(n_samples,), device=device, dtype=torch.long)

    nullids = torch.full((n_samples,), config.n_classes, device=device, dtype=torch.long)
    
    x_t = torch.randn((n_samples, 3, config.input_spatial, config.input_spatial), device=device)

    # RK45 if selected
    if use_RK45: 
        # define continuous time span
        t_span = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)

        # torchdiffeq requires a function f(t, x) -> v
        def ode_func(t, x):
            ode_func.nfe += 1 # increase number of function evaluations counter

            t_expanded = t.expand(x.shape[0])
            return call_model(model, x, t_expanded, c, nullids, do_cfg, cfg_strength)

        # init nfe counter
        ode_func.nfe = 0

        # call torchdiffeq ode integration
        # it returns (2, B, C, H, W) since t_span has 2 elements
        x_1 = odeint(
            func=ode_func,
            y0=x_t,
            t=t_span,
            method="dopri5",    # Dormand-Prince (RK45)
            atol=tolerance,          # abslute error tolerance  
            rtol=tolerance,          # relative error tolerence 
            options={"max_num_steps": 1000}
        )[-1]

        total_calls = ode_func.nfe
        print(f"[RK45 SAMPLER]: The model was called a total of {total_calls} times. About {total_calls//6} timesteps were used.")
        return x_1
  

    if use_tqdm:
        loop = tqdm(range(n_steps))
    else:
        loop = range(n_steps)

    # pre-calculate t tensors
    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
    dt_half = dt_tensor / 2

    for i in loop:
        current_t = dt_tensor * i

        # using .expand() instead of .full() saves memory
        t_1 = current_t.expand(n_samples)

        if last_step_euler and i == n_steps - 1:
            # last step: Euler jump to x_1
            v = call_model(model, x_t, t_1, c, nullids, do_cfg, cfg_strength)
            x_t = x_t + dt_tensor * v
            continue


        t_2 = (current_t + dt_half).expand(n_samples)
        t_3 = t_2
        t_4 = (current_t + dt_tensor).expand(n_samples)

        # first velocity prediction k_1 (beginning)
        k_1 = call_model(model, x_t, t_1, c, nullids, do_cfg, cfg_strength)

        # second velocity prediction k_2 (middle, estimated using k_1)
        estim_2 = x_t + k_1 * dt_half
        k_2 = call_model(model, estim_2, t_2, c, nullids, do_cfg, cfg_strength)

        # third velocity prediction k_3 (middle, estimated using k_2)
        estim_3 = x_t + k_2 * dt_half
        k_3 = call_model(model, estim_3, t_3, c, nullids, do_cfg, cfg_strength)

        # forth velocity prediction k_4 (end, estimated using k_3)
        estim_4 = x_t + k_3 * dt_tensor
        k_4 = call_model(model, estim_4, t_4, c, nullids, do_cfg, cfg_strength)

        # next timestep:
        # use average velocity prediction. middle gets double weight
        x_t = x_t + (dt_tensor / 6) * (k_1 + 2*k_2 + 2*k_3 + k_4)
    
    return x_t

