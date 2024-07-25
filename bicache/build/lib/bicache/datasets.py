from bicache.caches import DiskStoreCache, RAMStoreCache

class IndexSubsetDataset:
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = indices
        self.dims = None
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return self.ds[self.indices[i]]
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class BiLevelCachedDataset:
    def __init__(self, ds, memory_cache_size=None, disk_cache_size=None, disk_cache_path="./lmdbm.db"):
        self.ds = ds
        self.ssd_cache = DiskStoreCache(ds, disk_cache_size, disk_cache_path)
        self.memory_cache = RAMStoreCache(self.ssd_cache, memory_cache_size)

    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        return self.memory_cache[i]
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]