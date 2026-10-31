import tkinter as tk
from tkinter import ttk
import torch
import torch.nn as nn



# . means cuurent dir, .. means parent dir
from .tools import select_dict_items_gui


from ..plots import plot_stats, plot_emastat_vs_order, plot_gradient_magnitude_in_module_leafs, plot_gradient_flow, plot_gradient_flow_by_module_type, plot_stat_vs_time_vs_depth


module_refs = {} # module references for plotting
label_refs = {}  # label references for plotting

# Recursive function to add nodes
def model_tree(root, treeview, parent, module, parent_path: str = "", extradata: dict = None):

    # TAGS
    # module : module label
    # meta, meta_child : various metadata
    # tech, tech_child : runtime and mem used
    # gradients, gradients_child : specific gradients for this modules parameters
    # gradients_button: plot_gradient_magnitude_in_module_leafs when clicked
    # attribute, attribute_child : additional attributes
    # stats


    if extradata is None:
        extradata = {}

    if parent_path == "":
        label = f"{module.__class__.__name__}"
        summary_id = treeview.insert(parent,"end",text=label,tags=("meta",))

        # n params, n submodules
        treeview.insert(summary_id,"end",text=f"Total trainable params: {sum(p.numel() for p in module.parameters() if p.requires_grad)}",tags=("meta_child",))
        treeview.insert(summary_id,"end",text=f"Total non-trainable params: {sum(p.numel() for p in module.parameters() if not p.requires_grad)}",tags=("meta_child",))
        treeview.insert(summary_id,"end",text=f"Total submodules: {len(list(module.modules()))}",tags=("meta_child",))

        for k, v in extradata.items():
            treeview.insert(summary_id,"end",text=f"{k}: {v}",tags=("meta_child",))

    for name, child in module.named_children():
        
        # Full path for uniqueness
        if parent_path:
            full_name = f"{parent_path}.{name}"
        else:
            full_name = name

        # label for this module 
        number_label = f'[operation #{child._order}]' if hasattr(child, '_order') and child._order is not None else ''
        label = f"{name}:{child.__class__.__name__} {number_label}"

        # insert module node
        item_id = treeview.insert(parent,"end",text = label, tags=("module",)) # FOLDER



        # -------- info folder -------- (has metadata, gradients and stats and technical info inside it)
        info_id = treeview.insert(item_id, "end", text="info", tags=("info",)) # FOLDER



        # -------- technical info --------
        tech_id = treeview.insert(info_id, "end", text="Technical info:", tags=("tech",)) # FOLDER
        treeview.insert(tech_id, "end", text=f"Runtime: {child._runtime}s", tags=("tech_child",))
        treeview.insert(tech_id, "end", text=f"Memory used: {child._mem_used}", tags=("tech_child",))



        # -------- metadata --------
        # input & output tensor shapes
        meta_id = treeview.insert(info_id,"end",text="metadata:",tags=("meta",)) # FOLDER
        treeview.insert(meta_id, "end", text=f"Input shape: {child._inshape}", tags=("meta_child",))
        treeview.insert(meta_id, "end", text=f"Output shape: {child._outshape}", tags=("meta_child",))

        # number of parameters
        treeview.insert(meta_id, "end", text=f"Trainable params: {child._n_trainable_params}", tags=("meta_child",))
        treeview.insert(meta_id, "end", text=f"Non-trainable params: {child._n_not_trainable_params}", tags=("meta_child",))

        # ema stats
        ema_fwd_stats_id = treeview.insert(meta_id,"end",text=f"ema_fwd_stats (Forward signal)",tags=("meta",)) # FOLDER
        for k, v in child._ema_fwd_stats.items():
            treeview.insert(ema_fwd_stats_id,"end",text=f"{k}: {(30-len(k))*' '} {v:.3e}",tags=("meta_child",)) # for example "mean: 0.1e-2"
    
        ema_bwd_stats_id = treeview.insert(meta_id,"end",text=f"ema_bwd_stats (Backward signal)",tags=("meta",)) # FOLDER
        for k, v in child._ema_bwd_stats.items():
            treeview.insert(ema_bwd_stats_id,"end",text=f"{k}: {(30-len(k))*' '} {v:.3e}",tags=("meta_child",)) # for example "mean: 0.1e-2"
        


        # -------- gradients --------
        if child._is_leaf and len(child._gradients) > 0:
            grad_id = treeview.insert(info_id, "end", text="Gradients:", tags=("gradients",)) # FOLDER
            for k, v in child._gradients.items():
                treeview.insert(grad_id, "end", text=f"{k} | {' | '.join(f'{k1}: {v1:.3e}' for k1, v1 in v.items())}",tags=("gradients_child",))
        gradients_button_id = treeview.insert(info_id, "end", text="[ Gradient plots (right click or Ctrl+click) ]", tags=("gradients_button",))
        module_refs[gradients_button_id] = child
        label_refs[gradients_button_id] = label

        # ------- attributes (inside metadata) ------
        if child._additional_attributes != {}:
            attributes_id = treeview.insert(meta_id,"end",text="Attributes:",tags=("attribute",)) # FOLDER
            for k, v in child._additional_attributes.items():
                if isinstance(v, torch.Tensor):
                    treeview.insert(attributes_id,"end",text=f"{k}: size={tuple(v.shape)}, absmax={round(v.float().abs().max().item(),5)}, std={round(v.float().std().item(),5)}, mean={round(v.float().mean().item(),5)}",tags=("attribute_child",))



        # ------- stats ------
        stats_id = treeview.insert(info_id,"end",text="[ Stat plots (right click or Ctrl+click) ]",tags=("stats",)) # BUTTON -> opens plots when right clicked
        if child._fwd_stats != {} or child._bwd_stats != {}:
            module_refs[stats_id] = child
            label_refs[stats_id] = label



        # -------- recurse --------
        model_tree(
            root = root,
            treeview = treeview,
            parent = item_id,
            module = child,
            parent_path = full_name,
            extradata = None,
        )




def tkinter_treeview(analyzer, module: nn.Module, extradata: dict, root):

    # function that's called when fwd_stats or bwd_stats is clicked
    def on_right_click1(event):
        item = event.widget.identify_row(event.y)
        if not item:
            return
        
        tags = event.widget.item(item, "tags")

        if "stats" not in tags:
            return
        
        module = module_refs.get(item, None)
        
        treeview = event.widget
        treeview.selection_set(item)
        treeview.focus(item)

        options = list(module._fwd_stats.keys())
        options_dict = {k:"" for k in options}
        selected_stats = select_dict_items_gui(root = root, input_dict = options_dict,  title = label_refs.get(item, "Stats"))

        plot_stats(module, title = label_refs.get(item, "Stats"), stats_to_plot = selected_stats)

    def on_right_click2(event): # when a gradients button row is right clicked
        item = event.widget.identify_row(event.y)
        if not item:
            return
        
        tags = event.widget.item(item, "tags")

        if "gradients_button" not in tags:
            return

        module = module_refs[item]
        title = label_refs[item]

        tempdict = select_dict_items_gui(root = root, input_dict = {"scale_aware":"", "delete_empty":""})
        scale_aware = True if "scale_aware" in tempdict else False
        delete_empty = True if "delete_empty" in tempdict else False

        plot_gradient_magnitude_in_module_leafs(module, title, scale_aware, delete_empty)
 
    def on_right_click(event):
        on_right_click1(event)
        on_right_click2(event)

    if extradata is None:
        extradata = {}

    root.title("PyTorch Module Explorer")
    root.geometry("1000x700")   # width x height (pixels)

    # region | cosmetics, creation of treeview object
    style = ttk.Style()
    style.theme_use("default")

    style.configure(
        "custom.Treeview",
        rowheight=26,
        font=("consolas", 12),
        background="black",
        foreground="white",
        fieldbackground="black",
        indent = 30,
    )

    style.map(
        "custom.Treeview",
        background=[("selected", "#347083")],
        foreground=[("selected", "white")]
    )

    # treeview widget
    treeview = ttk.Treeview(root, style="custom.Treeview")
    treeview.pack(expand=True, fill="both")

    # color tags
    smallfont = ("consolas",9)
    treeview.tag_configure("module", foreground="dodgerblue")
    treeview.tag_configure("meta", foreground="lightseagreen")
    treeview.tag_configure("meta_child", foreground="lightgray", font=smallfont)
    treeview.tag_configure("stats", foreground="orange")
    treeview.tag_configure("gradients", foreground="blueviolet")
    treeview.tag_configure("gradients_child", foreground="violet", font=smallfont)
    treeview.tag_configure("gradients_button", foreground="violet")
    treeview.tag_configure("info", foreground="pink")
    treeview.tag_configure("tech", foreground="darkred")
    treeview.tag_configure("tech_child", foreground="red", font=smallfont)
    treeview.tag_configure("attribute", foreground="yellow")
    treeview.tag_configure("attribute_child", foreground="yellow", font=smallfont)
    treeview.tag_configure("attention_heatmap", foreground="lightgreen")
    
    # endregion

    # populate treeview
    model_tree(root= root, treeview= treeview, module= module, parent= "", parent_path= "", extradata= extradata)

    # bind 
    treeview.bind("<Button-3>", on_right_click) #standard right-click (works on mouse)
    treeview.bind("<Control-Button-1>", on_right_click) #macOS fallback (Ctrl + click)

    # SECOND WINDOW - PLOTTING STATS VS MODEL DEPTH
    def open_second_window():

        options = list(analyzer.tracking_funcs.keys())
        if any("kl_to_unit_gaussian" in m._fwd_stats for m in analyzer.model.modules()):
            options.append("kl_to_unit_gaussian")

        selected = select_dict_items_gui(root = root, input_dict = {a:"" for a in options})
        selected = list(selected.keys())

        title="Forward and backward stats (exponential moving averages) for modules vs. module operation order ('depth')"
        plot_emastat_vs_order(ordered_modules= analyzer.modules_in_order, selected_stats=selected, title=title)



    # THIRD WINDOW - TENSORLOGGER
    def open_third_window():
        style3 = ttk.Style()
        style3.theme_use("default")

        # configure style per widget type
        style3.configure("Custom.TCheckbutton",font=("consolas", 12),background="black",foreground="white",indicatorcolor="black")
        style3.configure("Custom.TLabel",font=("consolas", 12),background="black",foreground="white")
        style3.configure("Custom.TButton",font=("consolas", 12),background="black",foreground="white",padding=6)

        root3 = tk.Toplevel(root) # third window
        root3.geometry("300x450")
        root3.configure(bg="black")

        ttk.Label(root3, text="Plots for additional logged tensors (TensorLogger)", style="Custom.TLabel").pack()

        # selectable buttons
        options = list(analyzer.tensorlogger.logs.keys())
        checkbutton_vars = {}

        for k in options:
            var = tk.IntVar(value=0)
            checkbutton_vars[k] = var
            
            cb = ttk.Checkbutton(root3, text=str(k), variable=var, style="Custom.TCheckbutton")
            cb.pack(anchor="w")

        def plot_button_func():
            title = "TensorLogger"
            selected = [k for k, v in checkbutton_vars.items() if v.get() == 1]
            selected_funcs = select_dict_items_gui(root = root3, input_dict = analyzer.tracking_funcs)
            analyzer.tensorlogger.plot(labels = selected, selected_func_names = selected_funcs, plot_title = title)

        plot_button = ttk.Button(root3, text="Plot", command=plot_button_func, style="Custom.TButton")
        plot_button.pack(anchor="s")

    
    # FORTH WINDOW - GRADIENT FLOW PLOT
    def open_forth_window():
        style4 = ttk.Style()
        style4.theme_use("default")

        # configure style per widget type
        style4.configure("Custom.TCheckbutton",font=("consolas", 12),background="black",foreground="white",indicatorcolor="black")
        style4.configure("Custom.TLabel",font=("consolas", 12),background="black",foreground="white")
        style4.configure("Custom.TButton",font=("consolas", 12),background="black",foreground="white",padding=6)

        root4 = tk.Toplevel(root) # fourth window
        root4.geometry("300x450")
        root4.configure(bg="black")

        ttk.Label(root4, text="Gradient Flow Plots", style="Custom.TLabel").pack()

        # button for plot_gradient_flow
        def plot_gradient_flow_func():
            selected_metric = select_dict_items_gui(root = root4, input_dict = {"grad_weight_ratio":"", "grad_norm":"", "weight_norm":"", "mean_abs_grad":""})
            selected_metric = list(selected_metric.keys())[0] if selected_metric else "grad_weight_ratio"
            plot_gradient_flow(modules_in_order=analyzer.modules_in_order, metric=selected_metric, log_scale=True, title=f"Gradient Flow ({selected_metric}, EMA cached gradients, log scale)")
        
        plot_button = ttk.Button(root4, text="Plot gradient flow (all modules)", command=plot_gradient_flow_func, style="Custom.TButton")
        plot_button.pack(anchor="s")

        # button for plot_gradient_flow_by_module_type
        def plot_gradient_flow_by_module_type_func():
            selected_metric = select_dict_items_gui(root = root4, input_dict = {"grad_weight_ratio":"", "grad_norm":"", "weight_norm":"", "mean_abs_grad":""})
            selected_metric = list(selected_metric.keys())[0] if selected_metric else "grad_weight_ratio"
            plot_gradient_flow_by_module_type(modules_in_order=analyzer.modules_in_order, metric=selected_metric, log_scale=True, title=f"Gradient Flow by module type ({selected_metric}, EMA cached gradients, log scale)")
        
        plot_button2 = ttk.Button(root4, text="Plot gradient flow (by module type)", command=plot_gradient_flow_by_module_type_func, style="Custom.TButton")
        plot_button2.pack(anchor="s")


    # FIFTH WINDOW - STAT VS TIME VS DEPTH PLOT
    def open_fifth_window():
        style5 = ttk.Style()
        style5.theme_use("default")

        # configure style per widget type
        style5.configure("Custom.TCheckbutton",font=("consolas", 12),background="black",foreground="white",indicatorcolor="black")
        style5.configure("Custom.TLabel",font=("consolas", 12),background="black",foreground="white")
        style5.configure("Custom.TButton",font=("consolas", 12),background="black",foreground="white",padding=6)

        root5 = tk.Toplevel(root) # fifth window
        root5.geometry("300x450")
        root5.configure(bg="black")

        ttk.Label(root5, text="Stat vs Time vs Depth Plot", style="Custom.TLabel").pack()

        # dropdown for plot mode
        plot_mode_var = tk.StringVar(value="heatmap")
        plot_mode_label = ttk.Label(root5, text="Plot mode:", style="Custom.TLabel")
        plot_mode_label.pack()
        plot_mode_dropdown = ttk.Combobox(root5, textvariable=plot_mode_var, values=["heatmap", "3d"], style="Custom.TCombobox")
        plot_mode_dropdown.pack()

        # dropdown for stat mode
        stat_mode_var = tk.StringVar(value="fwd")
        stat_mode_label = ttk.Label(root5, text="Stat mode:", style="Custom.TLabel")
        stat_mode_label.pack()
        stat_mode_dropdown = ttk.Combobox(root5, textvariable=stat_mode_var, values=["fwd", "bwd"], style="Custom.TCombobox")
        stat_mode_dropdown.pack()

        # dropdown for display normalization
        display_mode_var = tk.StringVar(value="raw")
        display_mode_label = ttk.Label(root5, text="Display mode:", style="Custom.TLabel")
        display_mode_label.pack()
        display_mode_dropdown = ttk.Combobox(root5, textvariable=display_mode_var, values=["raw", "per_module_normalized"], style="Custom.TCombobox")
        display_mode_dropdown.pack()

        # dropdown for module selection
        module_var = tk.StringVar(value="all")
        module_label = ttk.Label(root5, text="Module selection:", style="Custom.TLabel")
        module_label.pack()
        module_options = ["all"] + list(dict.fromkeys(
            m.__class__.__name__ for m in analyzer.modules_in_order
        ))
        module_dropdown = ttk.Combobox(root5, textvariable=module_var, values=module_options, style="Custom.TCombobox")
        module_dropdown.pack()

        # checkbox for log scale
        log_scale_var = tk.IntVar(value=1)
        log_scale_checkbox = ttk.Checkbutton(root5, text="Use log scale for heatmap", variable=log_scale_var, style="Custom.TCheckbutton")
        log_scale_checkbox.pack()

        window_size_var = tk.StringVar(value="1")
        window_size_label = ttk.Label(root5, text="Timesteps per mean window:", style="Custom.TLabel")
        window_size_label.pack()
        window_size_entry = ttk.Entry(root5, textvariable=window_size_var, style="Custom.TEntry")
        window_size_entry.pack()

        def plot_button_func():

            options = list(analyzer.tracking_funcs.keys())
            if any("kl_to_unit_gaussian" in m._fwd_stats for m in analyzer.model.modules()):
                options.append("kl_to_unit_gaussian")
    
            selected = select_dict_items_gui(root = root, input_dict = {a:"" for a in options})
            selected = list(selected.keys())

            plot_stat_vs_time_vs_depth(
                modules_in_order=analyzer.modules_in_order,
                selected_stats=selected,
                plot_mode=plot_mode_var.get(),
                stat_mode=stat_mode_var.get(),
                use_log_scale=bool(log_scale_var.get()),
                module=module_var.get(),
                display_mode=display_mode_var.get(),
                window_size=int(window_size_var.get()),
                model=analyzer.model,
            )

        plot_button = ttk.Button(root5, text="Plot", command=plot_button_func, style="Custom.TButton")
        plot_button.pack(anchor="s")




    # button to open second window
    style.configure("Custom.TButton",font=("consolas", 12),background="black",foreground="white",padding=6)
    button = ttk.Button(root, text="Stats vs module operation order plots", command=open_second_window, style="Custom.TButton")
    button.pack(anchor="s")

    # button to open third window
    button = ttk.Button(root, text="TensorLogger", command=open_third_window, style="Custom.TButton")
    button.pack(anchor="s")

    # button to open forth window
    button = ttk.Button(root, text="Gradient Flow Plots", command=open_forth_window, style="Custom.TButton")
    button.pack(anchor="s")


    # button to open fifth window
    button = ttk.Button(root, text="Stat vs Time vs Depth Plot", command=open_fifth_window, style="Custom.TButton")
    button.pack(anchor="s")





    # Start GUI
    root.mainloop()
