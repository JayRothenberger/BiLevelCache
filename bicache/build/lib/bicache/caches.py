from zarr._storage.store import Store, BaseStore, StoreLike
from typing import Sequence, Mapping, Optional, Union, List, Tuple, Dict, Any
from zarr.types import PathLike as Path, DIMENSION_SEPARATOR
from collections import OrderedDict
from zarr.storage import listdir, getsize
from threading import Lock, RLock
from lmdbm import Lmdb
import pickle

from multiprocessing.managers import BaseManager
from multiprocessing import Manager, Value
from collections import OrderedDict


def buffer_size(array):
    return len(pickle.dumps(array))

class DictDB:
    def __init__(self, path, item_keys):
        self.db = Lmdb.open(path, "c")
        self.item_keys = item_keys

    def __getitem__(self, index):
        return {key: self.db[str(index)+key] for key in self.item_keys}

    def __setitem__(self, index, value):
        self.db.update({str(index)+key: value[key] for key in self.item_keys})

    def __delitem__(self, index):
        for key in self.item_keys:
            del self.db[str(index)+key]


class DiskStoreCache(Store):
    """Storage class that implements a least-recently-used (LRU) cache layer over
    some other store. Intended primarily for use with stores that can be slow to
    access, e.g., remote stores that require network communication to store and
    retrieve data.

    All data entries are assumed to be of the same size

    Parameters
    ----------
    store : Iterable
        The dataset containing the actual data to be cached.
    max_size : int
        The maximum size that the cache may grow to, in number of bytes. Provide `None`
        if you would like the cache to have unlimited size.

    """

    def __init__(self, store, max_size: int, path="lmdbm.db"):
        self._store = store
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache = OrderedDict()
        self._contains_cache: Dict[Any, Any] = {}
        self._listdir_cache: Dict[Path, Any] = dict()
        self.hits = self.misses = 0

        single_item = self._store[0]
        self.buffer_size = buffer_size(single_item)

        self.item_keys = [key for key in single_item]
        self.db_path = path
        self._db = DictDB(self.db_path, self.item_keys)

    def __getstate__(self):
        return (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._listdir_cache,
            self.hits,
            self.misses,
        )

    def __setstate__(self, state):
        (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._listdir_cache,
            self.hits,
            self.misses,
        ) = state

    def __len__(self):
        return len(self._keys())

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        return key in self._db

    def clear(self):
        self._store.clear()
        self.invalidate()

    def keys(self):
        return iter(self._keys())

    def _keys(self):
        if self._keys_cache is None:
            self._keys_cache = list(self._store.keys())
        return self._keys_cache

    def listdir(self, path: Path = None):
        # TODO: should warn not implemented or such
        return None
    
    def getsize(self, path=None) -> int:
        return self._current_size

    def _pop_key(self):
        # remove the first value from the cache, as this will be the least recently
        # used value
        k, _ = self._values_cache.popitem(last=False)
        return k

    def _accommodate_value(self, value_size):
        if self._max_size is None:
            return
        # ensure there is enough space in the cache for a new value
        while self._current_size + value_size > self._max_size:
            k = self._pop_key()
            del self._db[k]
            self._current_size -= self.buffer_size

    def _cache_key(self, key: Path, value):
        # cache a value
        value_size = self.buffer_size
        # check size of the value against max size, as if the value itself exceeds max
        # size then we are never going to cache it
        if self._max_size is None or value_size <= self._max_size:
            self._accommodate_value(value_size)
            self._db[key] = value
            self._current_size += value_size

    def _invalidate_keys(self):
        self._keys_cache = OrderedDict()
        self._listdir_cache.clear()

    def _invalidate_value(self, key):
        if key in self._db:
            del self._db[key]
            self._current_size -= self.buffer_size

    def __getitem__(self, key):
        # TODO: this is bad, should check membership and then retrieve
        try:
            # first try to obtain the value from the cache
            value = self._db[key]
            # cache hit if no KeyError is raised
            self.hits += 1
            # treat the end as most recently used
            self._keys_cache.move_to_end(key)

        except KeyError as e:
            # cache miss, retrieve value from the store
            value = self._store[key]

            self.misses += 1

        return value

    def __setitem__(self, key, value):
        self._db[key] = value

    def __delitem__(self, key):
        del self._db[key]


class RAMStoreCache(Store):
    """Storage class that implements a least-recently-used (LRU) cache layer over
    some other store. Intended primarily for use with stores that can be slow to
    access, e.g., remote stores that require network communication to store and
    retrieve data.

    Parameters
    ----------
    store : Store
        The store containing the actual data to be cached.
    max_size : int
        The maximum size that the cache may grow to, in number of bytes. Provide `None`
        if you would like the cache to have unlimited size.

    """

    def __init__(self, store: StoreLike, max_size: int):
        BaseManager.register('OrderedDict', OrderedDict)
        manager = BaseManager()
        manager.start()

        self._store: BaseStore = BaseStore._ensure_store(store)
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache = None
        self._contains_cache: Dict[Any, Any] = {}
        self._listdir_cache: Dict[Path, Any] = dict()
        self._values_cache: Dict[Path, Any] = manager.OrderedDict()
        self._mutex = Lock()
        self.hits, self.misses = Value('i', 0), Value('i', 0)

    def __getstate__(self):
        return (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._contains_cache,
            self._listdir_cache,
            self._values_cache,
            self.hits,
            self.misses,
        )

    def __setstate__(self, state):
        (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._contains_cache,
            self._listdir_cache,
            self._values_cache,
            self.hits,
            self.misses,
        ) = state
        self._mutex = Lock()

    def __len__(self):
        return len(self._keys())

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        with self._mutex:
            if key not in self._contains_cache:
                self._contains_cache[key] = key in self._store
            return self._contains_cache[key]

    def clear(self):
        self._store.clear()
        self.invalidate()

    def keys(self):
        with self._mutex:
            return iter(self._keys())

    def _keys(self):
        if self._keys_cache is None:
            self._keys_cache = list(self._store.keys())
        return self._keys_cache

    def listdir(self, path: Path = None):
        with self._mutex:
            try:
                return self._listdir_cache[path]
            except KeyError:
                listing = listdir(self._store, path)
                self._listdir_cache[path] = listing
                return listing

    def getsize(self, path=None) -> int:
        return getsize(self._store, path=path)

    def _pop_value(self):
        # remove the first value from the cache, as this will be the least recently
        # used value
        _, v = self._values_cache.popitem(last=False)
        return v
    
    def _accommodate_value(self, value_size):
        if self._max_size is None:
            return
        # ensure there is enough space in the cache for a new value
        while self._current_size + value_size > self._max_size:
            # In this case, we need to demote the value.  This is the only time a write happens in l2.
            k, v = self._values_cache.popitem(last=False)
            self._store[k] = v
            self._current_size -= buffer_size(v)

    def _cache_value(self, key: Path, value):
        # cache a value
        value_size = buffer_size(value)
        # check size of the value against max size, as if the value itself exceeds max
        # size then we are never going to cache it
        if self._max_size is None or value_size <= self._max_size:
            self._accommodate_value(value_size)
            self._values_cache.update({key: value})
            self._current_size += value_size


    def invalidate(self):
        """Completely clear the cache."""
        with self._mutex:
            self._values_cache.clear()
            self._invalidate_keys()
            self._current_size = 0


    def invalidate_values(self):
        """Clear the values cache."""
        with self._mutex:
            self._values_cache.clear()


    def invalidate_keys(self):
        """Clear the keys cache."""
        with self._mutex:
            self._invalidate_keys()



    def _invalidate_keys(self):
        self._keys_cache = None
        self._contains_cache.clear()
        self._listdir_cache.clear()

    def _invalidate_value(self, key):
        if key in self._values_cache:
            value = self._values_cache.pop(key)
            self._current_size -= buffer_size(value)

    def __getitem__(self, key):
        try:
            # first try to obtain the value from the cache
            with self._mutex:
                value = self._values_cache.get(key)
                # cache hit if no KeyError is raised
                with self.hits.get_lock():
                    self.hits.value += 1
                # treat the end as most recently used
                self._values_cache.move_to_end(key)

        except KeyError:
            # cache miss, retrieve value from the store
            value = self._store[key]
            with self._mutex:
                with self.misses.get_lock():
                    self.misses.value += 1
                # need to check if key is not in the cache, as it may have been cached
                # while we were retrieving the value from the store
                if self._values_cache.get(key) is None:
                    self._cache_value(key, value)

        return value

    def __setitem__(self, key, value):
        self._store[key] = value
        with self._mutex:
            self._invalidate_keys()
            self._invalidate_value(key)
            self._cache_value(key, value)

    def __delitem__(self, key):
        del self._store[key]
        with self._mutex:
            self._invalidate_keys()
            self._invalidate_value(key)
