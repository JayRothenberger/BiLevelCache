import os
import shutil

import torch
import torchvision
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from bicache.datasets import BiLevelCachedDataset

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)

    path = '/local/cache'
    # the data we will be caching
    data = torchvision.datasets.ImageNet('/somewhere/on/distributed/filesystem')
    # Building the object that will handle the caching.  Upon access it will retrieve from the filesystem and cache the example.
    # When the cache would exceed memory_cache_size bytes in RAM the element is evicted to disk.  
    # When both are full, elements become uncached in LRU order.
    cached_dataset = BiLevelCachedDataset(data, memory_cache_size=100_000_000_000, disk_cache_size=300_000_000_000, disk_cache_path=path)

    # this is an indexed dataset that can be used like any other in pytorch with a dataloader
    loader_kwargs = {'batch_size': 4, 
                     'num_workers': 2, #  We can even use multiple workers.  The cache is thread-safe.
                     'pin_memory': True,
                     'shuffle': True}

    loader = DataLoader(cached_dataset, **loader_kwargs)

    memory_hits = []
    memory_miss = []

    for _ in range(4):
        for i in loader:
            memory_miss.append(cached_dataset.memory_cache.misses.value - max(memory_miss + [0]))
            memory_hits.append(cached_dataset.memory_cache.hits.value - max(memory_hits + [0]))

    print(len([cached_dataset.memory_cache._values_cache.get(i) for i in range(8) if cached_dataset.memory_cache._values_cache.get(i) is not None]))

    with open('out.txt', 'a') as fp:
        fp.writelines(f'\n{rank} hit' + str(memory_hits) + '\n')
        fp.writelines(f'\n{rank} miss' + str(memory_miss))

    torch.distributed.barrier()

    if rank == 0:
        shutil.rmtree(path)

    cleanup()