# Get Cifar-10 dataset from pytorch datasets and create a dataloader object

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Utils.paths import DATA_DIR

# ===============================

def get_dataloader(config, split="train"):
    """split: 'train' or 'test'"""
    # Define transforms
    # For diffusion model normalize to [-1, 1] and convert to tensor 
    # for visualisation convert back to [0,1]
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)), # resize 
        transforms.ToTensor(), # Converts HxWxC in [0,255] to [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1,1]
    ])

    # ===============================

    # Load dataset
    if split == "train":
        print("creating dataloader for TRAIN set")
        do_shuffle = True
        dataset = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform)
    else:
        print("creating dataloader for TEST set")
        do_shuffle = False
        dataset   = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)

    # ===============================

    # Create dataloaders
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=do_shuffle, 
                        num_workers=2,
                        pin_memory=True, 
                        persistent_workers=True, 
                        prefetch_factor=4) 

    return loader

def get_dataset(config, split="train"):
    """split: 'train' or 'test'"""
    # Define transforms
    # For diffusion model normalize to [-1, 1] and convert to tensor 
    # for visualisation convert back to [0,1]
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)), # resize 
        transforms.ToTensor(), # Converts HxWxC in [0,255] to [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1,1]
    ])

    # ===============================

    # Load datasets
    if split == "train":
        print("creating dataset for TRAIN set")
        dataset = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform)
    else:
        print("creating dataset for TEST set")
        dataset = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)


    return dataset
