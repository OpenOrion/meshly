"""Generic two-tier LRU cache for Packable objects.

Provides an in-memory LRU cache backed by on-disk storage via PackableStore.
Disk I/O is parallelised with ForkPool for batch reads and writes.

Usage:
    from meshly import PackableCache, PackableStore
    from my_module import MyPackable

    store = PackableStore(root_dir=Path("/data"))
    cache = PackableCache(store, decoder=MyPackable, prefix="my_cache")

    # batch get (memory -> disk -> miss)
    found = cache.get_many({"key1", "key2"})

    # batch put (writes to memory + disk)
    cache.put_many({"key1": obj1, "key2": obj2})
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Generic, TypeVar

from meshly.packable import Packable, PackableStore
from meshly.utils.fork_pool import ForkPool

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Packable)

# Module-level helpers for ForkPool (must be picklable)

def _load_one(args: tuple[str, str, type]) -> tuple[str, bytes | None]:
    """Read a single asset file from disk. Returns (key, bytes|None)."""
    asset_path_str, key, _ = args
    p = Path(asset_path_str)
    if p.exists():
        try:
            return key, p.read_bytes()
        except Exception:
            pass
    return key, None


def _save_one(args: tuple[str, bytes]) -> None:
    """Write a single asset file to disk."""
    asset_path_str, data = args
    p = Path(asset_path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


class PackableCache(Generic[T]):
    """Two-tier LRU cache: in-memory + disk via PackableStore.

    Lookup order: memory -> disk -> miss.
    New entries are written to both tiers.
    Disk I/O uses ForkPool for parallelism on batch operations.

    Args:
        store: PackableStore for disk persistence.
        decoder: Packable subclass used to decode bytes from disk.
        prefix: Key prefix for namespacing within the store's assets dir.
        max_memory: Maximum entries in the in-memory LRU cache.
    """

    def __init__(
        self,
        store: PackableStore,
        decoder: type[T],
        prefix: str = "",
        max_memory: int = 10_000,
    ):
        self._store = store
        self._decoder = decoder
        self._prefix = prefix
        self._max_memory = max_memory
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()

    def _store_key(self, key: str) -> str:
        if self._prefix:
            return f"{self._prefix}/{key}"
        return key

    # -- public API -----------------------------------------------------------

    def get(self, key: str) -> T | None:
        """Get a single item (memory -> disk -> None)."""
        result = self.get_many({key})
        return result.get(key)

    def put(self, key: str, value: T) -> None:
        """Put a single item into both tiers."""
        self.put_many({key: value})

    def get_many(self, keys: set[str]) -> dict[str, T]:
        """Batch get. Returns only the keys that were found."""
        found: dict[str, T] = {}

        # Tier 1: memory
        with self._lock:
            for k in keys:
                if k in self._cache:
                    self._cache.move_to_end(k)
                    found[k] = self._cache[k]

        # Tier 2: disk (parallel via ForkPool)
        missing = keys - found.keys()
        if not missing:
            return found

        disk_hits = self._load_many_disk(missing)

        # Promote disk hits to memory
        if disk_hits:
            found.update(disk_hits)
            with self._lock:
                for k, v in disk_hits.items():
                    self._cache[k] = v
                    self._cache.move_to_end(k)
                self._evict()

        return found

    def put_many(self, items: dict[str, T]) -> None:
        """Batch put into memory + disk."""
        # Memory
        with self._lock:
            for k, v in items.items():
                if k in self._cache:
                    self._cache.move_to_end(k)
                else:
                    self._cache[k] = v
            self._evict()

        # Disk (parallel via ForkPool)
        self._save_many_disk(items)

    def clear(self) -> None:
        """Clear in-memory cache (disk is not affected)."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    # -- disk I/O (parallelised) ----------------------------------------------

    def _load_many_disk(self, keys: set[str]) -> dict[str, T]:
        work: list[tuple[str, str, type]] = []
        for k in keys:
            store_key = self._store_key(k)
            path = str(self._store.asset_file(store_key))
            work.append((path, k, self._decoder))

        results = ForkPool.map(_load_one, work, min_items_for_parallel=4)

        hits: dict[str, T] = {}
        for key, raw in results:
            if raw is not None:
                try:
                    hits[key] = self._decoder.decode(raw)
                except Exception:
                    logger.debug(f"PackableCache: failed to decode {key}")
        return hits

    def _save_many_disk(self, items: dict[str, T]) -> None:
        work: list[tuple[str, bytes]] = []
        for k, v in items.items():
            store_key = self._store_key(k)
            path = str(self._store.asset_file(store_key))
            work.append((path, v.encode()))

        ForkPool.map(_save_one, work, min_items_for_parallel=4)

    # -- internal -------------------------------------------------------------

    def _evict(self) -> None:
        """Evict oldest entries to stay within max_memory. Call with lock held."""
        while len(self._cache) > self._max_memory:
            self._cache.popitem(last=False)
