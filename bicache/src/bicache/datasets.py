from bicache.caches import DiskStoreCache, RAMStoreCache
import torch
import psutil
import shutil
import pickle
import random
import warnings
import os

def max_cache_elements(cache):
    return cache._max_size // cache.buffer_size


class IndexSubsetDataset:
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        return self.ds[self.indices[i]]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class LazyShardDataset:
    """
    Parameters
    ---------------
    disk_size: Union[float|int]
        percent of free disk space to use for the cache, or size in bytes.  If 0, no disk cache will be created.
    memory_size: Union[float|int]
        percent of free memory to use for the cache, or size in bytes.
    overflow: float
        (shard_size - cache_size) / cache_size, use None for no overflow limit.  Default is 0.  If 0, no RAM cache will be created.
    min_elements: int
        The minimum number of elements to cache.  Otherwise, the shards will be of equal size and uniformly sampled without replacement from the dataset.
    rank: int
        unique rank id of the current process
    num_replicas: int
        size of the process pool
    seed: int
        random seed for shuffling
    mode: str
        one of ['rank', 'node', 'disk'].
        'rank': Each rank in the process group gets its own shard to cache.  Caches to both RAM and local disk.  Disk cache space is shared across ranks, and
            RAM cache space is allocated per rank.  Shuffling will not occur across rank partitions, only within a single rank partition.
        'node': Each node in the training gets its own shard to cache.  Caches to both RAM and local disk.  Disk cache space is shared across ranks, and
            RAM cache space is allocated per rank.  Shuffling across ranks occurs which may impact RAM cache performance.
        'disk': Each node in the training gets its own shard to cache.  Caches only to local disk.  Disk cache space is shared across ranks.  Shuffling across 
            ranks occurs which does not impact disk cache performance.
    """
    # TODO: automatically create shards that will fill the common cache size (max out the minimum cache size across all ranks)
    def __init__(self, ds, shuffle=False, disk_size=0.8, memory_size=0.8, overflow=0.0, min_elements=0, rank=None, num_replicas=None, seed=13, mode='rank', disk_cache_path='./bicache'):
        self.ds = ds
        self.indices = []
        self.shuffle = shuffle

        if isinstance(disk_size, float): 
            assert disk_size <= 1.0, 'if disk_size is provided as a float it must be in the range [0, 1.0]'
            self.disk_size = int(psutil.disk_usage(disk_cache_path).free * disk_size)
        else:
            assert isinstance(disk_size, int), f'disk_size must be either float or int, found: {type(disk_size)}'
            assert disk_size >= 0, f'disk_size cannot be negative'
            self.disk_size = disk_size

        if isinstance(memory_size, float): 
            assert memory_size <= 1.0, 'if memory_size is provided as a float it must be in the range [0, 1.0]'
            self.memory_size = int(psutil.virtual_memory().free * memory_size)
        else:
            assert isinstance(memory_size, int), f'memory_size must be either float or int, found: {type(memory_size)}'
            assert memory_size >= 0, f'memory_size cannot be negative'
            self.memory_size = memory_size

        self.overflow=overflow
        self.min_elements = min_elements
        self.rank = rank
        self.num_replicas = num_replicas
        self.seed = seed
        self.mode = mode
        self.cache_path = disk_cache_path

        # TODO: make the shard IndexDataset
        self.buffer_size = len(pickle.dumps(ds[0]))

        cache_size = self.disk_size + self.memory_size
        max_shard_size = (1 + overflow) * cache_size

        assert min_elements <= max_shard_size, f'number of requested minimum elements does not fit in this cache + overflow allowance (rank {rank})'

        if os.environ.get('LOCAL_WORLD_SIZE') is None:
            warnings.warn('Environment variable LOCAL_WORLD_SIZE has not been set.  It is reccommended to run the process group with torchrun or set this variable manually, otherwise the local world size is assumed to be 1.')
            self.local_world_size = 1
        else:
            self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE'))

        inds = list(range(len(ds)))
        random.seed(self.seed)
        random.shuffle(inds)

        first_group_worker = self.rank - (self.num_replicas % self.local_world_size)

        self.disk_inds = sum([inds[first_group_worker + i:len(ds):self.num_replicas][:max_shard_size] for i in range(self.local_world_size)], [])
        self.rank_inds = inds[rank:len(ds):self.num_replicas][:max_shard_size]

        self.disk_ds = IndexSubsetDataset(ds, self.disk_inds)
        self.rank_ds = IndexSubsetDataset(ds, self.rank_inds)

        assert mode in ['rank', 'node', 'disk']

        assert self.disk_size + self.memory_size > 0, 'cannot have sum cache size equal to 0'

        if mode == 'rank':
            # a dataset for bi-level partition caching (each rank gets a partition)
            self.ssd_cache = DiskStoreCache(self.rank_ds, self.disk_size, self.cache_path, self.memory_size > 0) if self.disk_size > 0 else self.ds
            self.cache = RAMStoreCache(self.ssd_cache, self.memory_size) if self.memory_size > 0 else self.ssd_cache
        elif mode == 'node':
            # a dataset for bi-level unpartitioned caching (each machine gets a partition, ram caches are a free-for-all)
            self.ssd_cache = DiskStoreCache(self.rank_ds, self.disk_size, self.cache_path, self.memory_size > 0) if self.disk_size > 0 else self.ds
            self.cache = RAMStoreCache(self.ssd_cache, self.memory_size) if self.memory_size > 0 else self.ssd_cache
        elif mode == 'disk':
            # a dataset for disk caching (each machine gets a partition, no ram cache)
            assert self.disk_size > 0, 'cannot have only disk cache with size 0'
            self.cache = DiskStoreCache(self.rank_ds, self.disk_size, self.cache_path)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        return self.cache[self.indices[i]]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
        else:
            if self.shuffle:
                random.shuffle(self.inds)

    def __del__(self):
        shutil.rmtree(self.cache_path)


class BiLevelCachedDataset:
    def __init__(self, ds, memory_cache_size=None, disk_cache_size=None, disk_cache_path="./lmdbm.db"):
        self.ds = ds
        self.ssd_cache = DiskStoreCache(ds, disk_cache_size, disk_cache_path, memory_cache_size > 0)
        self.memory_cache = RAMStoreCache(self.ssd_cache, memory_cache_size)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, i):
        return self.memory_cache[i]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

