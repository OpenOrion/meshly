"""Fork-based parallel processing utilities.

Provides a standardized fork-based multiprocessing pool that:
- Uses fork context on POSIX for COW memory sharing (fastest)
- Falls back to sequential execution when fork is unavailable
- Works safely with FastAPI and other async frameworks

The key insight is that fork() happens atomically before worker code runs,
so parent threads don't affect child safety as long as workers don't
interact with parent state. This is safe for OCC/CAD operations.
"""

from __future__ import annotations

import logging
import os
import threading
import multiprocessing as mp
from typing import TypeVar, Callable, Iterable, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ForkPool:
    """Fork-based parallel processing utilities.
    
    Provides static methods for parallel map operations using fork-based
    multiprocessing. Falls back to sequential execution when fork is unavailable.
    """

    @staticmethod
    def can_use_fork() -> bool:
        """Check if fork-based multiprocessing is available.
        
        Returns True on POSIX systems where fork is the default or available.
        """
        return (
            os.name == "posix"
            and mp.get_start_method(allow_none=True) in (None, "fork")
        )

    @staticmethod
    def map(
        worker_fn: Callable[[T], R],
        items: Iterable[T],
        num_workers: int | None = None,
        min_items_for_parallel: int = 2,
    ) -> list[R]:
        """Map a function over items using fork-based parallelism.
        
        Uses fork context on POSIX for copy-on-write memory sharing.
        Falls back to sequential execution when:
        - Running on non-POSIX systems
        - Too few items to benefit from parallelism
        - Fork fails for any reason
        
        Args:
            worker_fn: Function to apply to each item. Must be picklable
                (module-level function, not lambda or closure).
            items: Items to process.
            num_workers: Number of worker processes. Defaults to min(cpu_count, len(items), 8).
            min_items_for_parallel: Minimum items required to use parallel processing.
        
        Returns:
            List of results in same order as input items.
            
        Example:
            def process_item(idx: int) -> str:
                return SHARED_DATA[idx].upper()
            
            SHARED_DATA = ["a", "b", "c"]
            results = ForkPool.map(process_item, range(3))
        """
        items_list = list(items)
        
        if not items_list:
            return []
        
        if num_workers is None:
            num_workers = min(mp.cpu_count() or 1, len(items_list), 8)
        
        use_parallel = (
            ForkPool.can_use_fork()
            and len(items_list) >= min_items_for_parallel
            and num_workers >= 2
        )
        
        if use_parallel:
            try:
                ctx = mp.get_context("fork")
                pool = ctx.Pool(num_workers)
                try:
                    result = pool.map(worker_fn, items_list)
                    pool.close()
                    threading.Thread(target=pool.join, daemon=True).start()
                    return result
                except Exception:
                    pool.terminate()
                    raise
            except Exception as e:
                logger.warning(f"Fork pool failed: {e}, falling back to sequential")

        # Sequential fallback
        return [worker_fn(item) for item in items_list]

    @staticmethod
    def map_unordered(
        worker_fn: Callable[[T], R],
        items: Iterable[T],
        num_workers: int | None = None,
        min_items_for_parallel: int = 2,
    ) -> Iterable[R]:
        """Map a function over items using fork-based parallelism, unordered results.
        
        Like map but uses imap_unordered for better load balancing when
        items have varying processing times. Results may arrive out of order.
        
        Args:
            worker_fn: Function to apply to each item.
            items: Items to process.
            num_workers: Number of worker processes.
            min_items_for_parallel: Minimum items required for parallel processing.
        
        Yields:
            Results as they complete (order not guaranteed).
        """
        items_list = list(items)
        
        if not items_list:
            return
        
        if num_workers is None:
            num_workers = min(mp.cpu_count() or 1, len(items_list), 8)
        
        use_parallel = (
            ForkPool.can_use_fork()
            and len(items_list) >= min_items_for_parallel
            and num_workers >= 2
        )
        
        if use_parallel:
            try:
                ctx = mp.get_context("fork")
                with ctx.Pool(num_workers) as pool:
                    yield from pool.imap_unordered(worker_fn, items_list)
                    return
            except Exception as e:
                logger.warning(f"Fork pool failed: {e}, falling back to sequential")
        
        # Sequential fallback
        yield from (worker_fn(item) for item in items_list)

    @staticmethod
    def starmap(
        worker_fn: Callable[..., R],
        items: Iterable[tuple[Any, ...]],
        num_workers: int | None = None,
        min_items_for_parallel: int = 2,
    ) -> list[R]:
        """Map a function over argument tuples using fork-based parallelism.
        
        Like map but unpacks each item as arguments to worker_fn.
        
        Args:
            worker_fn: Function to call with unpacked arguments.
            items: Tuples of arguments to unpack.
            num_workers: Number of worker processes.
            min_items_for_parallel: Minimum items for parallel processing.
        
        Returns:
            List of results in same order as input items.
        """
        items_list = list(items)
        
        if not items_list:
            return []
        
        if num_workers is None:
            num_workers = min(mp.cpu_count() or 1, len(items_list), 8)
        
        use_parallel = (
            ForkPool.can_use_fork()
            and len(items_list) >= min_items_for_parallel
            and num_workers >= 2
        )
        
        if use_parallel:
            try:
                ctx = mp.get_context("fork")
                pool = ctx.Pool(num_workers)
                try:
                    result = pool.starmap(worker_fn, items_list)
                    pool.close()
                    threading.Thread(target=pool.join, daemon=True).start()
                    return result
                except Exception:
                    pool.terminate()
                    raise
            except Exception as e:
                logger.warning(f"Fork pool failed: {e}, falling back to sequential")

        # Sequential fallback
        return [worker_fn(*args) for args in items_list]
