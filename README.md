# BiLevelCache

A two level cache for machine learning model training and parallel distributed data processing.  This project was created to make effective use of RAM and local SSD storage on distributed systems during training models whose data is too large to fit in either, but too slow to constantly fetch from disk.  There are three use cases envisioned for this project:

1. Caching data in memory and on local SSD for fast (re)-access
2. Lazy sharding of large datasets

    By automatically determining disjoint subsets of indices for each worker and worker group, and caching those examples for that worker or worker group a shard can be sent to a node 'lazily' during training.  This is particularly helpful for [Zarr](https://zarr.readthedocs.io/en/stable/)  v2 arrays that do not have sharding support.  However, any integer-indexed structure can be sharded with this library.
3. Parallel, distributed data processing


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
    + [Caching](#caching-datasets)
    + [Lazy Sharding](#lazy-sharding)
- [Features](#features)
- [Roadmap](#roadmap)
- [License](#license)

## Installation 
First, clone this repository

`git clone https://github.com/JayRothenberger/BiLevelCache`

Then, install the package in the `bicache` directory with pip

`pip install ./BiLevelCache/bicache`

## Usage

### Caching Datasets
```python 
import torch
import torchvision
from bicache import BiLevelCachedDataset
# a directory on the local ssd
path = '/local/cache'

# the data we will be caching
data = torchvision.datasets.ImageNet('/somewhere/on/distributed/filesystem')

"""
Examples are cached when accessed.  When the cache would exceed memory_cache_size bytes in 
RAM the element is evicted to disk.  When both are full, elements become uncached in LRU 
order.

Cache sizes are in bytes.
"""
cached_dataset = BiLevelCachedDataset(
                        data, 
                        memory_cache_size=100_000_000_000, 
                        disk_cache_size=300_000_000_000, 
                        disk_cache_path=path
                        )

# This is a dataset that can be used like any other in pytorch with a dataloader.
# We can even use multiple workers.  The cache is thread-safe.
loader_kwargs = {'batch_size': 4, 
                    'num_workers': 2, 
                    'pin_memory': True,
                    'shuffle': True}

loader = DataLoader(cached_dataset, **loader_kwargs)

...

# at the end of training make sure to remove the cache directory

shutil.rmtree(path)
```

### Lazy Sharding

```python
import torch
import torchvision
from bicache import LazyShardDataset

rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
# a directory on the local ssd
path = '/local/cache'
# the data we will be caching
data = torchvision.datasets.ImageNet('/somewhere/on/distributed/filesystem')

"""
Examples are cached when accessed.  When the cache would exceed memory_cache_size bytes in 
RAM the element is evicted to disk.  When both are full, elements become uncached in LRU 
order.

Cache sizes are in bytes.
"""

cached_dataset = LazyShardDataset(
                                data,
                                shuffle=True,
                                disk_size=0.95, # use at most 95% free disk space
                                memory_size=0.5, # use at most 50% free RAM
                                overflow=0.0, # shard will not exceed cache size
                                min_elements=len(data) // world_size, # shards are at least 
                                rank=rank,
                                num_replicas=world_size,
                                seed=13,
                                mode='rank', # each rank gets a partition of indices
                                disk_cache_path='./bicache'
                                )

# this is an indexed dataset that can be used like any other in pytorch with a dataloader
loader_kwargs = {'batch_size': 4, 
                    'num_workers': 8, #  We can even use multiple workers.
                    'pin_memory': True,
                    'shuffle': True}

loader = DataLoader(cached_dataset, **loader_kwargs)

memory_hits = []
memory_miss = []

for _ in range(2):
    for i in loader:
        memory_miss.append(cached_dataset.memory_cache.misses.value - max(memory_miss + [0]))
        memory_hits.append(cached_dataset.memory_cache.hits.value - max(memory_hits + [0]))

print(len([cached_dataset.memory_cache._values_cache.get(i) for i in range(8) if cached_dataset.memory_cache._values_cache.get(i) is not None]))

with open('out.txt', 'a') as fp:
    fp.writelines(f'\n{rank} hit' + str(memory_hits) + '\n')
    fp.writelines(f'\n{rank} miss' + str(memory_miss))

torch.distributed.barrier()

...

shutil.rmtree(path)
```

## Roadmap
✅ Caching to a Directory\
&nbsp;&nbsp;&nbsp;&nbsp;✅ Evaluate database options\
&nbsp;&nbsp;&nbsp;&nbsp;✅ Manage cache promotions in the case of an existing RAM cache\
✅ Caching to RAM\
&nbsp;&nbsp;&nbsp;&nbsp;✅ Thread safe OrderedDict\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ ~~Cross-Process OrderedDict Cache~~ (out of scope)\
&nbsp;&nbsp;&nbsp;&nbsp;✅ Thread safe hits and misses counters\
✅ Convenience Objects\
&nbsp;&nbsp;&nbsp;&nbsp;✅ Bilevel Cache Dataset\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ Lazy Sharding Dataset\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ Automatic cache size discovery\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ Shuffling\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ rank mode caching\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ node mode caching\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✅ disk mode caching\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⬜️ add option to ensure shards are of similar but automatically determined size.\
⬜️ Benchmarking\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ Bilevel Cache Dataset hit and miss latency\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ Lazy Sharding Dataset hit and miss latency\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ Wall time comparison to uncached data\
⬜️ Testing\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ Write unit tests for the library\
&nbsp;&nbsp;&nbsp;&nbsp;⬜️ Write integration tests for the library

## License
[def]: #license
Copyright 2024 Jay Rothenberger (jay.c.rothenberger@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.