# CODE THAT I USED TO FIND OUT OPTIMAL HYPERPARAMS FOR THE DATALOADER TO MAXIMIZE TRAINING SPEED. THIS IS NOT CALLED DURING NORMAL TRAINING, JUST USED ONCE TO FIND GOOD HYPERPARAMS.


from torch.utils.data import DataLoader

from colorama import Fore, Style

import torch
from tqdm import tqdm
import gc

import random
import math
import time
from contextlib import nullcontext # "do nothing" context

from Utils.plotimages import plot_images

# ==================================
@torch.no_grad()
def update_EMA_model(model, ema_model, decay: float = 0.999):
    """
    Updates EMA model using parameter iteration rather than name lookups.
    This bypasses name-prefix issues caused by torch.compile(model)
    When you wrap a model in torch.compile(), PyTorch creates an optimized wrapper. 
    By default, this wrapper prefixes all original parameter names with _orig_mod.
    """
    # Use .parameters() to ensure we align tensors by order
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

# ==================================


def train(config, model, ema_model, optimizer, diffusion, loader, analyzer = None):

    from Utils.data import get_full_dataset
    full_dataset = get_full_dataset(config)

    cuda = True if config.device == torch.device("cuda") else False

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) # autocast to bfloat16 (only for forward pass and loss calc)
        if cuda and config.training_autocasting
        else nullcontext()
    )
    print(f"Autocasting (Bf16): {cuda and config.training_autocasting}")

    ## constant lr
    lr = config.lr_max
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    search_results = []


    ema_model.eval()

    it_s_history = []
    # this results in 64 different configs
    batch_size_options = [48,64,96,128,192,256]
    num_workers_options = [2]
    prefetch_factor_options = [4]

    i = 0

    # have 2 negative epochs for warmup and then start testing different dataloader configs
    for epoch in range(-2, config.epochs):
        # start of epoch
        it_s_history = []

        if epoch > -1:
            batch_size = batch_size_options[epoch % len(batch_size_options)]
            num_workers = num_workers_options[0]
            prefetch_factor = prefetch_factor_options[0]
        else:
            num_workers = 2
            prefetch_factor = 4
            batch_size = 128 # 384

        if "loader1" in locals():
            del loader1
        gc.collect()
        # big wait time 
        # -> old workers dead before we spawn new ones
        # -> time for the hardware to cool dowm, prevent overheating and throttling during the speed tests
        time.sleep(5) #

        loader1 = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, # dont shuffle here because we want to keep the same order of images for each dataloader config to make the speed comparison more fair 
                            num_workers=num_workers,
                            pin_memory=True, 
                            persistent_workers=True if num_workers > 0 else False, 
                            prefetch_factor=prefetch_factor)

        


        loop = tqdm(loader1)
        for images, labels in loop:


            i += 1


            images =  images.to(config.device, non_blocking=config.nonblocking_gpu_transfers)
            labels = labels.to(config.device, non_blocking=config.nonblocking_gpu_transfers)

            ## diffusion
            _t = torch.randint(0,diffusion.T, (images.shape[0],),device=config.device)
            x_t, epsilon = diffusion.q_sample(images, _t)

            classids = [l if random.random() > config.class_conditioning_drop_prob else -1 for l in labels]
            classids = torch.tensor(classids, device=config.device, dtype=torch.long)

            with autocast_context:
                ## forward pass (no self-conditioning)
                epsilon_theta = model(x_t, _t, classids)
                
                ## Loss
                loss = torch.nn.functional.mse_loss(epsilon_theta, epsilon)

            optimizer.zero_grad(set_to_none=True)

            ## backward pass
            loss.backward()
            if config.clip_grad_norm:
                norm: float = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0, error_if_nonfinite=True)

            optimizer.step()


            ## update EMA model
            update_EMA_model(model, ema_model, decay=config.ema_decay)




            
            # print loss in beginning to check if the starting loss is reasonable
            if i == 1:
               print("Initial loss:", loss.item())


            dt = loop.format_dict['rate']
            it_s_history.append(dt)
            # save average it/s over this epoch

        
        # end of epoch
        # first epoch: warmup
        # next epochs: test combination of different n_workers and prefetch_factor settings to find optimal dataloader config for training speed
        if epoch == -1:
            print("Warmup epoch completed. Starting training speed tests with different dataloader configs...")
            pass
        elif epoch > -1 and epoch < 65: # 64 different configs


            # convert it/s into images/s
            valid_rates = [r for r in it_s_history if r is not None and r > 0]
            avg_it_s = sum(valid_rates)/len(valid_rates) if valid_rates  and len(valid_rates) > 0 else 0
            avg_images_s = avg_it_s * batch_size

            # this epoch's report
            search_results.append({
                "epoch": epoch,
                "img_s": avg_images_s,
                "nw": num_workers,
                "pf": prefetch_factor,
                "bs": batch_size
            })

            print(Fore.BLUE + f"Completed Epoch {epoch}")
            print(Fore.GREEN + f"Average images/s this epoch: {avg_images_s:.4f}")
            print(f"This epochs config: num_workers={num_workers}, prefetch_factor={prefetch_factor}, batch_size={batch_size}" + Style.RESET_ALL)
        elif epoch == 65:
            best = max(search_results, key=lambda x: x['img_s'])
            print(f"WINNER: Epoch {best['epoch']} with {best['img_s']:.2f} img/s")
            print(f"Winner's config: num_workers={best['nw']}, prefetch_factor={best['pf']}, batch_size={best['bs']}")
            print("Dataloader config tests completed. ")
            break # stop training after the dataloader config tests
    


    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# NOTE:
# Final result:
# Winner's config: 
## num_workers=2, prefetch_factor=4, batch_size=96