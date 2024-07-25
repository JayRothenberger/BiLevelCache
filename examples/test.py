import os
from util import setup, cleanup
from torch.utils.data import DataLoader, DistributedSampler
import shutil
from bicache.datasets import BiLevelCachedDataset, IndexSubsetDataset
from logistics.util.data import ConcatZarrDataset, CompiledZarrDataset
import torch


if __name__ == "__main__":

    path = 'lmdbm.db'
    # the store we will be caching
    compiled_train = CompiledZarrDataset(
                            os.path.join('/home/jroth/lumenus/emit/emit_latent', 'compiled_train'), 
                            {'radiance': 'radiance', 'xch4': 'xch4', 'elev': 'elev', 'latents': 'latents'}, 
                            )

    # Shuffling can occur in the train loader as usual for a single rank.
    # For multiple ranks per machine, the normal distributed sampler can be used, except with only 4 replicas.  We should make an example for this case.

    data = IndexSubsetDataset(compiled_train, list(range(8)))

    cached_dataset = BiLevelCachedDataset(data, 10_000_000_000, 10_000_000_000, path)

    rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)

    sampler = DistributedSampler(cached_dataset, rank=rank, num_replicas=world_size, shuffle=False)

    loader_kwargs = {'batch_size': 4, 
                     'sampler': sampler, 
                     'num_workers': 2,
                     'pin_memory': True,
                     'shuffle': False}

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