import random
import warnings

import numpy as np
import torch
import torch.utils.data

from .distributed import is_distributed


def reset_seed_worker_init_fn(worker_id):
    """Reset NumPy and Python seed for data loader worker."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=None,
    collate_fn=None,
    sampler=None,
    pin_memory=True,
    drop_last=False,
):
    if is_distributed():
        if sampler is None:
            sampler = torch.utils.data.DistributedSampler(dataset)
            shuffle = False
        else:
            warnings.warn("Custom sampler is used in DDP mode. Make sure your sampler correctly handles DDP sampling.")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader
