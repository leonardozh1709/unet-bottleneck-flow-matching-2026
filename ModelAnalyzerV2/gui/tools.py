import tkinter as tk
from tkinter import ttk

import torch
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt
import math
from typing import Callable

from colorama import Fore, Style



analyzer_prefix = Fore.GREEN + "[MA]:" + Style.RESET_ALL
def analyzer_print(*args, **kwargs):
    print(analyzer_prefix, *args, **kwargs)


def get_leafs_in_order(module: nn.Module) -> list[nn.Module]:
    """assumes modules have ._order and ._is_leaf to return all leaf submodules ordered my their ._order"""

    leafs: dict[int, nn.Module] = {} # order: module
    
    # get all leaf modules
    def walk_tree(m: nn.Module):
        if getattr(m, "_is_leaf"):
            leafs[m._order] = m
        else:
            for child in m.children():
                walk_tree(child)

    walk_tree(module) # populates leafs dict

    # return modules ordered by their ._order
    ordered_leafs = [leafs[k] for k in sorted(leafs)]

    return ordered_leafs 



def select_dict_items_gui(root: tk.Tk, input_dict: dict, title: str = "Select items"):
    if root is None:
        return

    # create the window
    win = tk.Toplevel(root)
    win.title(title)
    win.geometry("200x300")

    # the checkbutton variables
    vars_dict = {}

    frame = ttk.Frame(win, padding=10)
    frame.pack(fill="both", expand=True)

    for i, (k, v) in enumerate(input_dict.items()):
        var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text=str(k), variable=var).grid(row=i, column=0, sticky="w")
        vars_dict[k] = var

    result = {}

    def submit():
        nonlocal result
        result = {k: input_dict[k] for k, var in vars_dict.items() if var.get()} # result dict constructed from only the selected checkbuttons (if var.get() is True if the box is checked)
        win.destroy()

    ttk.Button(frame, text="Confirm", command=submit).grid(row=len(input_dict), column=0, pady=10)

    win.wait_window()

    return result



class TensorLogger(): 

    def __init__(self, funcs: dict[str, Callable[..., torch.Tensor]], interval: int = 100, max_log_length: int | None = None):
        # logging funcs must return a scalar tensor 

        if funcs is None or len(funcs) == 0:
            raise ValueError(f"Invalid functions dict")
        
        # save config
        self.max_log_length = max_log_length
        self.funcs = funcs
        self.interval = interval
        self.iterationcounter: dict[str, int] = {}
        self.logs: dict[str, dict[str, list[float] | deque]] = {} # label >> funcs >> values
    
    def clear(self):
        self.iterationcounter: dict[str, int] = {}
        self.logs: dict[str, dict[str, list[float] | deque]] = {} # label >> funcs >> values
    
    def log(self, value: torch.Tensor, label: str, force_log: bool = False):
        
        # augment iteration by one for this label, only actually log every self.interval calls
        self.iterationcounter[label] = self.iterationcounter.get(label, 0) + 1 
        if not force_log and self.iterationcounter[label] % self.interval != 0:
            return

        if label not in self.logs:
            # create data structure: dict[str, dict[str, list[float] | deque]]
            if self.max_log_length is None:
                self.logs[label] = {name: [] for name in self.funcs.keys()}
            else:
                self.logs[label] = {name: deque(maxlen=self.max_log_length) for name in self.funcs.keys()}
        
        for name, f in self.funcs.items():
            out = f(value)
            if out.numel() != 1: # check if scalar tensor
                raise ValueError(f"Function {name} must return scalar tensor, got {out.shape}")
            floatvalue = out.item() # scalar-tensor.item() gives float
            self.logs[label][name].append(floatvalue)

    def plot(self, labels: list[str], selected_func_names: list[str], plot_title: str):
        analyzer_print("Plotting TensorLogger data with labels", labels, "and funcs", selected_func_names)

        n = len(labels)

        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        
        fig, ax = plt.subplots(nrows, ncols, figsize=(10, 8)) # ax is a np array of subplots or just a subplot
        subplots = ax.flatten() if hasattr(ax, "flatten") else [ax] 

        # populate subplots
        for i, label in enumerate(labels):
            if label not in self.logs:
                raise ValueError(f"The label {label} does not exist as a key in the log")

            p = subplots[i]
            for func_name in selected_func_names:
                if func_name not in self.logs[label]:
                    raise ValueError(f"The function {func_name} does not exist for label {label}")

                y = self.logs[label][func_name]
                p.plot(y, label=func_name)

            p.set_title(label)
            p.grid(True)
            p.legend()

        # delete empty subplots
        for j in range(n, len(subplots)):
            fig.delaxes(subplots[j])

        plt.suptitle(plot_title)
        plt.show()



def test_tensorlogger():
    testfuncs = {
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        'absmax': lambda x: x.abs().max()
    }

    logger = TensorLogger(interval=5, funcs=testfuncs, max_log_length=None)

    for i in range(500):

        # arbitrary variables
        a = torch.randn((4,4), dtype=float)
        b = torch.tensor(i, dtype=float)
        c = torch.tensor(i**2, dtype=float)
        d = torch.tensor(math.sin(i/10), dtype=float)
        e = torch.randn((4,4), dtype=float) * i

        for i, var in enumerate([a,b,c,d,e]):
            logger.log(var, f"test{i}")

    allfuncnames = list(testfuncs.keys())
    logger.plot(labels=["test0","test1","test2","test3","test4"], selected_func_names=allfuncnames, plot_title="This is a test")
    


if __name__ == "__main__": # Only runs if this file is ran (not on import)
    test_tensorlogger()
        