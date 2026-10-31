import torch
import torch.nn as nn
from contextlib import nullcontext
from tqdm import tqdm
from Utils.config import config


# Calculate how much longer the model's generated path is than a perfect straight line. 
@torch.inference_mode()
def get_path_straightness(model, device, images: torch.Tensor, labels: torch.Tensor, noises_per_image: int, batch_size: int, n_ODE_steps: int=100) -> float:
    """recommended: 100 ODE steps, 32 images x 8 noises."""

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) # autocast to bfloat16 (only for forward pass and loss calc)
        if torch.cuda.is_available() else nullcontext()
    )

    # create a huge tensor with all the images repeated noises_per_image times
    repeated_images = images.repeat_interleave(noises_per_image, dim=0).to("cpu") # batches on cpu
    repeated_labels = labels.repeat_interleave(noises_per_image, dim=0).to("cpu")

    # chunk into batches
    img_batches = repeated_images.split(batch_size)
    label_batches = repeated_labels.split(batch_size)

    # save each instances relative trajectory length here
    global_relative_trajectory_lengths = []

    for img, label in tqdm(zip(img_batches, label_batches)):

        # move to device
        img = img.to(device)
        label = label.to(device).to(dtype=torch.long)
        B = img.shape[0]
  
        # sample starting point
        x_0 = torch.randn_like(img)
        x_t = x_0

        # to get the intergrated trajectory length
        model_trajectory = []

        # solve the ODE
        dt = 1 / n_ODE_steps
        for i in range(n_ODE_steps):
            t = torch.full((B,), dt*i, device=device)

            # fwd pass
            with autocast_context:
                v_output = model(x_t, t, label) # model takes in (B, C, H, W) x (B,) x (B,)

            # euler ODE step
            x_t = x_t + v_output * dt

            # save trajectory length (for each batch dim)
            step_length = torch.linalg.vector_norm(v_output, dim=(1, 2, 3)) * dt  # shape (B,)
            model_trajectory.append(step_length)

        x_1 = x_t

        # get the full length for each image/noise instance by summing
        trajectory_lengths = torch.stack(model_trajectory, dim = 0).sum(dim=0) # shape (B,)
        linear_trajectory_lengths = torch.linalg.vector_norm(x_1 - x_0, dim=(1, 2, 3)) # shape (B,)
        relative_trajectory_lengths = trajectory_lengths / linear_trajectory_lengths
        global_relative_trajectory_lengths.append(relative_trajectory_lengths.cpu())

    # get the mean relative trajectory length across all instances
    # first, stack batches
    stack = torch.cat(global_relative_trajectory_lengths, dim=0) # (B*len(img_batches),)
    # return mean as a float
    return stack.mean().item()


             
        

if __name__ == "__main__":
    from Utils.loadmodel import load_models
    from Utils.data import get_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ema_model = load_models("base_mini64", device=device)
    print("Successfully loaded EMA model weights!")

    loader = get_dataloader(config, split="train")

    # get a batch of images and labels
    images, labels = next(iter(loader))

    # get first 32
    images = images[:32]
    labels = labels[:32]

    # get path straightness
    straightness = get_path_straightness(ema_model, device, images, labels, noises_per_image=8, batch_size=32, n_ODE_steps=100)
    straightness = round(straightness, 4)
    print(f"Path straightness for trained model (mean relative trajectory length): {straightness}")

