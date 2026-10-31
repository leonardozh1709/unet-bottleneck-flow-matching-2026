import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle # only needed for type indications
import torch.fx as fx # torch fx symbolic tracer

import pynvml # NVIDIA Management Library

from datetime import datetime
from tqdm import tqdm
from collections import deque
import copy
import gc # python garbage collector access
import time
from typing import Type

import tkinter as tk

from .gui.tools import select_dict_items_gui
from .gui.tools import TensorLogger
from .gui.tools import analyzer_print
from .gui.app import tkinter_treeview

# =============================================================================
# fyi: attaching attributes to modules does not slow down their regular forward and backward passes 
# after training and using ModelAnalyzer each module has: (unless specified otherwise)

# static attributes:
# ._order                               | int
# ._is_leaf                             | bool
# ._runtime                             | float
# ._mem_used                            | string  e. g. "12 KB"
# ._inshape                             | tuple[tuple] e. g. ((1,2),(4,8))
# ._outshape                            | tuple[tuple] e. g. ((1,2),(4,8))
# ._n_trainable_params                  | int
# ._n_not_trainable_params              | int

# dynamic attributes
# ._param_cache                         | list with pointers to all parameters in a module (for faster bwd hooks)                         
# ._fwd_stats, ._bwd_stats              | dict[str, list | deque]
# ._ema_fwd_stats, ._ema_bwd_stats      | dict[str, torch.Tensor]
# ._gradients                           | module._gradients[label] = {"influence": influence_ema, "scale_aware_influence": scale_aware_influence_ema}, label is the parameter name
# ._additional_attributes (ema)         | dict[str, torch.Tensor]
# =============================================================================


# runtime and memory tracking
def runtime_memory_pre_hook(module, _input):
    module._start_time = time.perf_counter() # time before this module's computation is done (pre-hook)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize() # block the CPU until the GPU has finished all queued work
        module._mem_before = torch.cuda.memory_allocated() # memory allocated on cuda before this modules computation

def runtime_memory_hook(module, _input, _output):
    if torch.cuda.is_available(): # mem used only avaiable with cuda
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() # memory allocated on cuda after this module's computation
        module._mem_used = mem_after - module._mem_before 
        # formatting: convert into B, KB or MB depending on the size
        if module._mem_used < 1024:
            module._mem_used = f"{module._mem_used} B"
        elif module._mem_used < 1024 ** 2:
            module._mem_used = f"{module._mem_used / 1024:.2f} KB"
        else:
            module._mem_used = f"{module._mem_used / (1024 ** 2):.2f} MB"
    else:
        module._mem_used = "unavaiable"

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter() # time after this module's computation
    runtime = end_time - module._start_time
    module._runtime = runtime # final runtime estimate

def register_runtime_memory_hooks(model: nn.Module) -> list[RemovableHandle]: # torch.utils.hooks.RemovableHandle is pytorch's hook class
    """
    registers all runtime_memory hooks and pre_hooks
    returns a list containing the hooks
    """
    h = [] #store the hooks to prevent them from being discarded by python garbage collector
    for name, module in model.named_modules():
        if isinstance(module, nn.Module):
            try:
                h.append(module.register_forward_hook(runtime_memory_hook))
                h.append(module.register_forward_pre_hook(runtime_memory_pre_hook))
            except TypeError as e:
                analyzer_print(f"Error occurred while registering runtime/memory hooks for module {name}: {e}")
    return h

# Track GPU power consumption of the full model per forward pass (NVIDIA GPU)
def safe_nvml_init() -> bool: # check if pynvml is avaiable
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError:
        return False


def get_estimated_gpu_energy(model: nn.Module, dummy: list[torch.Tensor], iterations: int = 50) -> tuple[float, float]:
    """
    Only possible if using NVIDIA GPU and NVML is installed.
    This function only looks at the GPU at index 0. 
    Returns estimates energy used on GPU per fwd pass [J] and duration per fwd pass [s]. 
    Warmup iterations are half of the actual 'iterations' argument.
    Dummy must be a valid input to the model.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # GPU 0

    device = next(model.parameters()).device
    dummy = [d.to(device) for d in dummy] # ensure dummy inputs are on device

    # Warmup
    with torch.no_grad():
        for _ in range(iterations//2):
            _ = model(*dummy)

    powers = []
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    unsupported = False
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(*dummy)
            torch.cuda.synchronize() # wait for GPU to finish the computation before measuring power

            try:
                powers.append(pynvml.nvmlDeviceGetPowerUsage(handle)) # milliwatts
            except (pynvml.NVMLError_NotSupported, pynvml.NVMLError_Unknown):
                unsupported = True
                powers.append(0) # append 0 if power measurement is not supported

    if unsupported:
        analyzer_print("PYNVML power measurement not supported on this GPU or driver version.")

    end_time = time.perf_counter()
    duration = end_time - start_time

    avg_power_watts = sum(powers) / len(powers) / 1000 # mW -> W, watt is J/s
    energy_joules_per_pass = avg_power_watts * (duration / iterations) # W * s -> J

    duration_per_pass = duration / iterations # in s

    return energy_joules_per_pass, duration_per_pass

def get_execution_order_fx(model: nn.Module, modules_to_skip: list[Type] | tuple[Type, ...] | None = None) -> list[nn.Module]: # use the torch fx symbolic tracer to get the forward pass operations in order
    """
    Does a torch fx symbolic trace of the module, 
    gives modules the ._order attribute,
    returns list of the modules in their call order.
    """

    # Create a deep copy on CPU to ensure zero interference with GPU training
    model_copy = copy.deepcopy(model).cpu()

    # trace the copy
    gm = fx.symbolic_trace(model_copy)

    skip_types = tuple(modules_to_skip or ())
    modules_in_order = []

    for node in gm.graph.nodes:
        if node.op == "call_module":
            # Find the match in the ORIGINAL model
            target_name = node.target
            original_submodule = model.get_submodule(target_name)
            if isinstance(original_submodule, skip_types):
                continue
            modules_in_order.append(original_submodule)

    # give modules the ._order attribute
    for i, module in enumerate(modules_in_order):
        module._order = i

    # EXPLICIT CLEANUP
    del gm
    del model_copy
    # force python grabage collector to activate
    gc.collect()
    # if anything accidentally touched the GPU, clear it
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return modules_in_order

def fwd_quick_hook() -> RemovableHandle: # hook for the dummy forward pass, these are removed for training

    def hook(module, _input, _output):

        # check input and output shapes
        module._inshape = tuple(tuple(i.size()) for i in _input if i is not None)
        module._outshape = tuple(tuple(o.size()) for o in _output if o is not None)

        # number of parameters (trainable & not trainable)
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module._n_trainable_params = trainable
        module._n_not_trainable_params = total - trainable
    
    return hook

def register_fwd_quick_hooks(model: nn.Module) -> list[RemovableHandle]:
    h = [] #store the hooks to prevent them from being discarded by python garbage collector
    for name, module in model.named_modules():
        if isinstance(module, nn.Module):
            try:
                h.append(module.register_forward_hook( fwd_quick_hook() ))
            except TypeError as e:
                analyzer_print(f"Error occurred while registering runtime/memory hooks for module {name}: {e}")
    return h

# initialisation
def initiate_static_attributes(model: nn.Module):
    for module in model.modules(): # this walks the entire module tree (all model nested submodules)

        module._is_leaf = len(list(module.children())) == 0 # is this module a leaf node? = is this module not a container like nn.Sequential etc. ?
        
        # make sure attribute always exists
        module._order = None
        module._runtime = None
        module._mem_used = None
        module._inshape = None
        module._outshape = None
        module._n_trainable_params = None
        module._n_not_trainable_params = None

def initiate_dynamic_attributes(model: nn.Module, tracking_funcs: dict, max_history_size: int | None = 1000):
    for module in model.modules():
    # store history over iterations
        module._fwd_stats = {}
        module._bwd_stats = {}
        for stat_name in tracking_funcs.keys():
            module._fwd_stats[stat_name] = [] if max_history_size is None else deque(maxlen=max_history_size) 
            module._bwd_stats[stat_name] = [] if max_history_size is None else deque(maxlen=max_history_size)

        module._ema_fwd_stats = {} # dict to store the EMA of all stats in fwd_stats, updated in the hook function with a decay factor
        module._ema_bwd_stats = {} 

        module._gradients = {} # dict to store the gradients of all parameters in the module

        # cache params to use in bwd hooks -> speedup
        module._param_cache = []
        for name, p in module.named_parameters():
            module._param_cache.append((name, p))

        module._additional_attributes = {}

# main fwd/bwd hook
# Hook execution is now manual: the caller attaches and detaches hooks explicitly.
def main_hook(analyzer, fwd_or_bwd: str = "fwd", ema_decay: float = 0.9, tracking_funcs: dict = None, track_history: bool = True, track_gradients: bool = True, additional_attributes: list[str] = None, additional_attributes_ema_decay: float = 0.5):

    if tracking_funcs is None:
        tracking_funcs = {}

    if additional_attributes is None:
        additional_attributes = []

    def hook(module, _input, _output):

        if isinstance(_output, torch.Tensor):
            output_tensor = _output
        elif isinstance(_output, tuple):
            output_tensor = next((item for item in _output if isinstance(item, torch.Tensor)), None)
        else:
            return

        if output_tensor is None:
            return

        _output_detached = output_tensor.detach()

        # VALS
        vals: dict[torch.Tensor] = {name : (f(_output_detached)).to(torch.float16) for name,f in tracking_funcs.items()} # float16 to save memory at the cost of precision


        # HISTORY
        if track_history:
            stats = module._fwd_stats if fwd_or_bwd == "fwd" else module._bwd_stats

            for name, val in vals.items(): # apply given funcs
                stats[name].append(val)
                # the max history size is automatically enforced because we used deque in init (if the max history size is not None)


        # SPECIFIC GRADIENTS
        # bwd hooks update _gradients only on leaf modules (modules without children)
        if track_gradients and fwd_or_bwd == "bwd":
            if module._is_leaf:
                for name, p in module._param_cache: # use param cache, a reference to the model params
                    if p.grad is not None:
                        label = f"{name} {tuple(p.shape)} | n_params: {p.numel()} {'(non_trainable)' if not p.requires_grad else ''}"
                    
                        grad = p.grad.detach()
                        param = p.detach()

                        influence = grad.abs().mean()
                        scale_aware_influence = (param * grad).abs().mean()

                        prev = module._gradients.get(label, {"influence": 0, "scale_aware_influence": 0})

                        influence_ema = prev.get("influence", 0) * ema_decay + (1 - ema_decay) * influence
                        scale_aware_influence_ema = prev.get("scale_aware_influence", 0) * ema_decay + (1 - ema_decay) * scale_aware_influence

                        # store a copy of the gradient in the module's _gradients dict
                        module._gradients[label] = {"influence": influence_ema, "scale_aware_influence": scale_aware_influence_ema}
        

        # EMA STATS
        # ema_new = ema_old * decay + (1 - decay) * new_value 
        stats_dict = module._ema_fwd_stats if fwd_or_bwd == "fwd" else module._ema_bwd_stats
        for name, val in vals.items():
            ema_stat = stats_dict.get(name, 0)
            stats_dict[name] = ema_stat * ema_decay + (1 - ema_decay) * val # dicts are mutable -> stats_dict is a pointer to fwd_stats or bwd_stats


        # ADDITIONAL ATTRIBUTES (EMA)
        for attname in additional_attributes:
            if hasattr(module, attname):

                att = getattr(module, attname)
                if isinstance(att, torch.Tensor):
                    att = att.detach()

                prev = module._additional_attributes.get(attname, 0)
                module._additional_attributes[attname] = prev * ema_decay + (1 - ema_decay) * att

    return hook

def history_to_ema_value(history: list[torch.Tensor], decay: float) -> torch.Tensor:
    """Calculates exponential moving average given history and decay factor"""
    x = torch.stack(history).flatten()
    T = x.numel()

    powers = torch.arange(T, device=x.device)
    weights = decay**(T - 1 - powers)

    weights[1:] *= (1 - decay)  # apply (1 - decay) to all except first

    ema = torch.sum(weights * x) # scalar tensor

    return ema


# region | Tracking funcs
# - define all avaiable metrics
# in ModelAnalyzer.__init__ :
# - user config using select_dict_items_gui

epsilon = 1e-6
epsilon_2 = 1e-3

from torch.distributions import Normal, kl_divergence
def KL_to_unit_gaussian(mu, std) -> torch.Tensor:

    std = std.clamp(min=1e-6) 

    p = Normal(mu, std)
    q = Normal(torch.tensor(0.0), torch.tensor(1.0))
    return kl_divergence(p, q)

def excess_kurtosis(x) -> torch.Tensor: 
    if x.numel() < 2:
        return 0.0 # return 0 for 1-element tensors

    x = x.float()

    mu = x.mean()
    std = x.std(unbiased=False).clamp(min=1e-6) 

    kurtosis = ((x - mu)**4).mean() / std**4
    
    return (kurtosis - 3) # kurtosis - 3 because a unit gaussian dist. has kurtosis 3

def skewness(x) -> torch.Tensor:
    if x.numel() < 2:
        return 0.0 # return 0 for 1-element tensors
    
    x = x.float()

    mu = x.mean()
    std = x.std(unbiased=False).clamp(min=1e-6) 

    skew = ((x - mu)**3).mean() / std**3
    
    return skew

ALL_TRACKING_FUNCS = {
    "mean": lambda x: x.mean(),                                                     # (1st central moment)
    "std": lambda x: x.std(),                                                       # sqrt(variance)
    "max": lambda x: x.max(),
    "min": lambda x: x.min(),
    "zero_frac": lambda x: (x.abs() < epsilon).float().mean(),                      # count of values that are effectively zero (< epsilon)
    "small_frac": lambda x: (x.abs() < epsilon_2).float().mean(),                   # count of values that are very small (< epsilon_2)
    "mean_abs": lambda x: x.abs().mean(),                                           # average magnitude
    "skewness": skewness,                                                           # (3rd central moment)
    "excess_kurtosis": excess_kurtosis                                              # (excess 4th central moment in comparassion to unit gaussian)
} 
# endregion

class ModelAnalyzer:

    def __init__(self, root, model: nn.Module, ema_decay: float = 0.9, max_history_size: int | None = None, tensorlogger_interval: int = 100, tracking_funcs: dict = None, disable_prints: bool = False):

        self.model = model
        self.ema_decay = ema_decay
        self.print_fn = (lambda *args, **kwargs: None) if disable_prints else analyzer_print

        # select which tracking funcs to use using the tkinter gui
        if tracking_funcs is None:
            active_tracking_funcs = select_dict_items_gui(root=root, input_dict= ALL_TRACKING_FUNCS, title= "Select tracking funcs to use")
        else:
            active_tracking_funcs = tracking_funcs
        self.tracking_funcs = active_tracking_funcs

        self.print_fn("Selected tracking funcs:", list(self.tracking_funcs.keys()))

        initiate_static_attributes(model = self.model)
        initiate_dynamic_attributes(model = self.model, tracking_funcs = self.tracking_funcs, max_history_size = max_history_size)

        # create the TensorLogger
        self.tensorlogger = TensorLogger(funcs = self.tracking_funcs, interval = tensorlogger_interval, max_log_length = None)


    def enable_prints(self):
        self.print_fn = analyzer_print


    def get_state_dict(self):

        def getattributes(attributes: list[str], module: nn.Module) -> dict[str, dict]:
            attr_dict = {}
            for att in attributes:
                if hasattr(module, att):
                    attr_dict[att] = getattr(module, att)
            return attr_dict
        
        attributes_to_save = ["_order", "_is_leaf", "_runtime", "_mem_used", "_inshape", "_outshape", "_n_trainable_params", "_n_not_trainable_params",# static attributes
                            "_fwd_stats", "_bwd_stats", "_ema_fwd_stats", "_ema_bwd_stats", "_gradients", "_additional_attributes" # dynamic attributes
                            ]
        
        def _history_to_floats(stats: dict) -> dict:
            """Batched conversion of dict[stat_name -> deque/list of 0-dim tensors] to dict[stat_name -> list[float]].
            One .tolist() call per stat (not per value) — cheap, and only runs at save time.
            Safe to call on histories that are already floats (e.g. re-saving an already-migrated file)."""
            out = {}
            for stat_name, history in stats.items():
                history = list(history)
                if len(history) == 0:
                    out[stat_name] = []
                elif isinstance(history[0], torch.Tensor):
                    out[stat_name] = torch.stack(history).flatten().tolist()
                else:
                    out[stat_name] = history  # already plain floats
            return out

        state_dict: dict[str, dict] = {} # for every submodule, save a dict with the attributes in attributes_to_save that exist for this module
        for name, module in self.model.named_modules():
            attrs = getattributes(attributes_to_save, module)
            # convert only the big per-iteration histories; this reassigns keys in the
            # local `attrs` dict, so the live module's own ._fwd_stats/._bwd_stats
            # (and thus training speed) are completely unaffected
            if "_fwd_stats" in attrs:
                attrs["_fwd_stats"] = _history_to_floats(attrs["_fwd_stats"])
            if "_bwd_stats" in attrs:
                attrs["_bwd_stats"] = _history_to_floats(attrs["_bwd_stats"])
            state_dict[name] = attrs

        # save this modelanalyzers attributes
        modules_in_order = getattr(self, "modules_in_order", None)
        if modules_in_order is not None:
            # save only names, not the actual nn.Module objects — avoids re-pickling
            # the whole model's weights a second time inside the analyzer state dict
            module_to_name = {module: name for name, module in self.model.named_modules()}
            modules_in_order_names = [module_to_name[m] for m in modules_in_order]
        else:
            modules_in_order_names = None

        state_dict["ModelAnalyzer"] = {
            "tensorlogger": self.tensorlogger,
            "joules_per_pass": getattr(self, "joules_per_pass", None),
            "duration_per_pass": getattr(self, "duration_per_pass", None),
            "modules_in_order": modules_in_order_names,
        }



        return state_dict
    
    def load_state_dict(self, state_dict: dict[str, dict], modules_to_skip: list[Type] | tuple[Type, ...] | None = None) -> None:

        def setattributes(attributes: dict[str, object], module: nn.Module):
            for att, value in attributes.items():
                if hasattr(module, att):
                    setattr(module, att, value)

        skip_types = tuple(modules_to_skip or ())
        for name, module in tqdm(self.model.named_modules(), desc="Loading Modelanalyzer State Dict"):
            if isinstance(module, skip_types):
                continue
            if name in state_dict:
                setattributes(state_dict[name], module)
        
        # load this modelanalyzers attributes
        ma_attributes = state_dict.get("ModelAnalyzer", {})
        self.tensorlogger = ma_attributes.get("tensorlogger", None)
        self.joules_per_pass = ma_attributes.get("joules_per_pass", None)
        self.duration_per_pass = ma_attributes.get("duration_per_pass", None)

        # reassemble modules_in_order by looking names up against the live model
        modules_in_order_names = ma_attributes.get("modules_in_order", None)
        if modules_in_order_names is not None:
            self.modules_in_order = []
            for name in modules_in_order_names:
                module = self.model.get_submodule(name)
                if isinstance(module, skip_types):
                    continue
                module._order = len(self.modules_in_order)
                self.modules_in_order.append(module)
        else:
            self.modules_in_order = None
    
    def log_tensor(self, label: str, value: torch.Tensor, force_log: bool = False):
        self.tensorlogger.log(value=value, label=label, force_log=force_log)
    
    def clear_tensorlogger(self):
        self.tensorlogger.clear()
    
    def attach_main_hooks(self, modules_to_ignore: tuple[type] = (nn.Dropout, nn.Identity), skip_containers: bool = False, track_history: bool = True, track_gradients: bool = True, additional_attributes: list[str] = None, verbose: bool = False):
        """Attaches the main fwd and bwd hooks to the model. Should be done before training."""

        if modules_to_ignore is None:
            modules_to_ignore = tuple()

        h = [] # store the hooks to prevent them from being discarded by python garbage collector

        base_args = {
            "analyzer": self,
            "ema_decay": self.ema_decay,
            "tracking_funcs": self.tracking_funcs,
            "track_history": track_history,
            "track_gradients": track_gradients,
            "additional_attributes": additional_attributes,
            "additional_attributes_ema_decay": self.ema_decay,
        }

        # create hook function once, then assign them to the modules
        fwd_hook = main_hook(**{**base_args, "fwd_or_bwd": "fwd"})
        bwd_hook = main_hook(**{**base_args, "fwd_or_bwd": "bwd"})

        if verbose:
            self.print_fn("Base arguments for fwd and bwd hooks:")
            for k, v in base_args.items():
                self.print_fn(f"{k} = {v}")

        for name, module in self.model.named_modules():

            if isinstance(module, modules_to_ignore): # skip modules of types that are chosen to be ignored
                continue

            if skip_containers: # option to not add hooks for nn.Sequential, nn.Modulelist etc.
                if not module._is_leaf:
                    continue

            try:
                h.append(module.register_forward_hook(fwd_hook))
                h.append(module.register_full_backward_hook(bwd_hook))
            except Exception as e:
                self.print_fn(f"Could not register hooks for module {name}, {e}")

        self.main_hooks = h

        self.print_fn("Successfully attached main hooks")


    def add_kl_to_unit_gaussian(self):
        """
        Can be used after training to get the kl divergence to a unit gaussian if mean and std were tracked and kl_to_unit_gaussian was not tracked yet. 
        Caution: Takes some time.
        """
        if "mean" in self.tracking_funcs.keys() and "std" in self.tracking_funcs.keys() and "kl_to_unit_gaussian" not in self.tracking_funcs.keys():
            for module in tqdm(self.model.modules(), desc=f'Adding KL to unit gaussian. Number of modules: {len(list(self.model.modules()))} '):

                means = module._fwd_stats.get("mean", [])
                stds  = module._fwd_stats.get("std", [])

                if len(means) == 0 or len(stds) == 0:
                    continue

                if "mean" in module._fwd_stats and "std" in module._fwd_stats and "mean" in module._bwd_stats and "std" in module._bwd_stats:
                    module._fwd_stats["kl_to_unit_gaussian"] = []
                    for mu, sigma in zip(module._fwd_stats["mean"], module._fwd_stats["std"]):
                        KL = KL_to_unit_gaussian(mu, sigma)
                        if KL is not None:
                            module._fwd_stats["kl_to_unit_gaussian"].append(KL)

                    module._bwd_stats["kl_to_unit_gaussian"] = []
                    for mu, sigma in zip(module._bwd_stats["mean"], module._bwd_stats["std"]):
                        KL = KL_to_unit_gaussian(mu, sigma)
                        if KL is not None:
                            module._bwd_stats["kl_to_unit_gaussian"].append(KL)

                    # EMA
                    ema = history_to_ema_value(history = module._bwd_stats["kl_to_unit_gaussian"], decay = self.ema_decay)
                    module._ema_bwd_stats["kl_to_unit_gaussian"] = ema

                    ema = history_to_ema_value(history = module._fwd_stats["kl_to_unit_gaussian"], decay = self.ema_decay)
                    module._ema_fwd_stats["kl_to_unit_gaussian"] = ema
            
            self.print_fn("Succesfully added kl_to_unit_gaussian histories and ema computed from tracked mean and variances")

            
    def remove_main_hooks(self):
        for h in self.main_hooks:
            h.remove()
        self.main_hooks = []

        self.print_fn("Successfully removed main hooks")

    def dummy_forward_pass(self, dummyinputs: list[torch.Tensor], gpu_estimate_iterations: int = 5, runtime_memory_iterations: int = 5, quick_hook_iterations: int = 1, modules_to_skip: list[Type] | tuple[Type, ...] | None = None):
        """runs the model with a dummy input to determine tensor sizes, runtime, memory, execution order"""
        start_time = datetime.now()

        # get execution order
        self.modules_in_order = get_execution_order_fx(self.model, modules_to_skip=modules_to_skip) # save pointers to modules in order, also gives each module the _order attribute
        
        # get joules per pass, duration per pass only if pynvml avaiable
        if safe_nvml_init():
            self.joules_per_pass, self.duration_per_pass = get_estimated_gpu_energy(self.model, dummy= dummyinputs, iterations= gpu_estimate_iterations)
        else:
            self.print_fn("NVML not available. Skipping GPU energy estimation.")
            self.joules_per_pass = "unavaiable"
            self.duration_per_pass = "unavaiable"

        # get runtime & memory usage per module
        runtime_memory_hooks = register_runtime_memory_hooks(self.model)
        for i in range(runtime_memory_iterations):
            _ = self.model(*dummyinputs)
        for h in runtime_memory_hooks:
            h.remove()
        
        # get tensor sizes, parameter counts per module
        fwd_quick_hooks = register_fwd_quick_hooks(self.model)
        for i in range(quick_hook_iterations):
            _ = self.model(*dummyinputs)
        for h in fwd_quick_hooks:
            h.remove()
        
        self.print_fn(f"Successfully ran dummy forward pass, elapsed: {(datetime.now() - start_time).total_seconds()}s")

    def open_interface(self, root):
        """Opens the tkinter UI for this ModelAnalyzer. Access plots from there."""

        extradata = {"Estimated duration per forward pass": self.duration_per_pass              if hasattr(self, "duration_per_pass") else "unavaiable",
                     "Estimated energy [joules] used per forward pass": self.joules_per_pass    if hasattr(self, "joules_per_pass") else "unavaiable",
                     }
        root.deiconify()
        tkinter_treeview(analyzer=self, module=self.model, extradata=extradata, root=root)


def test():
    from .test import ExampleNet

    root1 = tk.Tk()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    analyzer_print(f"Using device: {device}")

    dummy = torch.randn((2,4), device=device) # B, C
    net = ExampleNet(4).to(device)
    main_analyzer = ModelAnalyzer(
                                root=root1,
                                model=net,
                                ema_decay=0.9,
                                tensorlogger_interval=1)

    # run once
    main_analyzer.dummy_forward_pass(dummyinputs= [dummy], runtime_memory_iterations=10, quick_hook_iterations=10)

    for i in range(10):

        main_analyzer.attach_main_hooks(verbose= False, track_gradients= True, track_history= True, additional_attributes= ["test123"])


        net.zero_grad()

        y = net(dummy)
        loss = y.mean()
        loss.backward()

        # after fwd + bwd pass: remove main hooks
        main_analyzer.remove_main_hooks()

        for module in net.modules():
            module.test123 = torch.randn((2,2))

        main_analyzer.log_tensor("MyFavouriteTensor", torch.randn((2,2))*i)


    main_analyzer.add_kl_to_unit_gaussian() # computes kl to unit gaussian history from mean and std histories
    main_analyzer.remove_main_hooks()
    main_analyzer.open_interface(root=root1)



if __name__ == "__main__":
    test()