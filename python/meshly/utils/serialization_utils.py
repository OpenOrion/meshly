"""Serialization utilities for arrays and assets.

This module provides utilities for:
- Extracting arrays, Packables, and ResourceRefs into serializable data with $ref checksums
- Fetching assets from various providers (dict, callable)
- Handling async asset fetchers
"""

import asyncio
import gzip
import inspect
import os
import threading
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from pydantic import BaseModel, ConfigDict, Field

from meshly.array import ArrayUtils, ArrayEncoding
from meshly.common import AssetProvider
from meshly.utils.checksum_utils import ChecksumUtils

if TYPE_CHECKING:
    from meshly.packable import Packable
    from meshly.resource import Resource


# =============================================================================
# Type Definitions
# =============================================================================

JsonValue = Union[None, bool, int, float, str, list, tuple, dict]
"""JSON-serializable types for extracted values."""

# Collections larger than this threshold are extracted in parallel.
# Below the threshold the ThreadPoolExecutor overhead isn't worth it.
_PARALLEL_THRESHOLD = 8

# Thread-local flag to prevent recursive parallelism: when already executing
# inside a pool worker, child collections run sequentially.
_in_worker = threading.local()


# =============================================================================
# ExtractedResult Model
# =============================================================================


class ExtractedResult(BaseModel):
    """Result of extracting a value for serialization (immutable)."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    
    value: JsonValue
    """The extracted/serialized value (with $ref dicts for arrays)"""
    assets: dict[str, bytes] = Field(default_factory=dict)
    """Map of checksum -> encoded bytes"""




# =============================================================================
# SerializationUtils
# =============================================================================

class SerializationUtils:
    """Utility class for serialization operations.
    
    Primary responsibilities:
    - extract_value: Recursively serialize values to JSON-compatible data with $ref checksums
    - get_asset/get_cached_asset: Fetch binary assets from various providers
    """

    # -------------------------------------------------------------------------
    # Asset Fetching
    # -------------------------------------------------------------------------

    @staticmethod
    def get_asset(assets: AssetProvider, checksum: str) -> bytes:
        """Get asset bytes from a provider (dict or callable).

        Supports both sync and async callables.

        Args:
            assets: Asset provider (dict or callable)
            checksum: Asset checksum to fetch

        Returns:
            Asset bytes

        Raises:
            KeyError: If asset not found
        """
        if callable(assets):
            result = assets(checksum)
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                    # We're inside a running event loop - create a new thread to run the coroutine
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, result)
                        result = future.result()
                except RuntimeError:
                    # No running event loop, safe to use run_until_complete
                    result = asyncio.get_event_loop().run_until_complete(result)
            return result
        if checksum not in assets:
            raise KeyError(f"Missing asset with checksum '{checksum}'")
        return assets[checksum]

    # -------------------------------------------------------------------------
    # Value Extraction (convert values to $ref checksums)
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_array_to_ref(value: object, encoding: ArrayEncoding = "array") -> ExtractedResult:
        """Extract an array and return ExtractedResult with $ref dict and asset."""
        extracted = ArrayUtils.extract(value, encoding)
        checksum = ChecksumUtils.compute_bytes_checksum(extracted.data)
        ref_metadata = extracted.info.model_copy(update={"ref": checksum})
        ref_dict = ref_metadata.model_dump(by_alias=True, exclude_none=True)
        return ExtractedResult(value=ref_dict, assets={checksum: extracted.data})

    @staticmethod
    def extract_value(value: object) -> ExtractedResult:
        """Recursively extract a value, replacing Packables and nested structures with refs."""
        from meshly.packable import Packable
        from meshly.resource import Resource

        # Arrays at top level: encode with default "array" encoding
        if ArrayUtils.is_array(value):
            return SerializationUtils._extract_array_to_ref(value)

        # sub-Packables: encode as zip or expand fields
        if isinstance(value, Packable):
            return SerializationUtils._extract_subpackable(value)

        # Resources: gzip compress file data
        if isinstance(value, Resource):
            return SerializationUtils._extract_resource(value)

        # Dicts: recursively extract values (parallel for large collections)
        if isinstance(value, dict):
            return SerializationUtils._extract_dict(value)

        # Lists/tuples/sets: recursively extract items (parallel for large collections)
        if isinstance(value, (list, tuple, set)):
            return SerializationUtils._extract_sequence(value)

        # BaseModels: extract fields
        if isinstance(value, BaseModel):
            return SerializationUtils.extract_basemodel(value)

        # Primitives: pass through unchanged
        return ExtractedResult(value=value)

    @staticmethod
    def _extract_dict(value: dict) -> ExtractedResult:
        """Extract a dict, parallelising across entries when large enough."""
        keys = list(value.keys())
        vals = list(value.values())
        results = SerializationUtils._map_parallel(SerializationUtils.extract_value, vals)
        return ExtractedResult(
            value=dict(zip(keys, [e.value for e in results])),
            assets={k: v for e in results for k, v in e.assets.items()},
        )

    @staticmethod
    def _extract_sequence(value: Union[list, tuple, set]) -> ExtractedResult:
        """Extract a list/tuple/set, parallelising across items when large enough."""
        items = list(value)
        results = SerializationUtils._map_parallel(SerializationUtils.extract_value, items)
        result_value: Any = [e.value for e in results]
        if isinstance(value, tuple):
            result_value = tuple(result_value)
        return ExtractedResult(
            value=result_value,
            assets={k: v for e in results for k, v in e.assets.items()},
        )

    @staticmethod
    def _map_parallel(fn, items: list) -> list:
        """Apply fn to each item, using a ThreadPoolExecutor when beneficial.

        Guards against recursive parallelism: if already inside a worker thread,
        runs sequentially to avoid spawning unbounded nested pools.
        """
        already_in_worker = getattr(_in_worker, "active", False)
        if already_in_worker or len(items) < _PARALLEL_THRESHOLD:
            return [fn(item) for item in items]

        max_workers = min(len(items), os.cpu_count() or 4)

        def _worker(item):
            _in_worker.active = True
            try:
                return fn(item)
            finally:
                _in_worker.active = False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_worker, items))

    # -------------------------------------------------------------------------
    # Type-Specific Extraction Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_subpackable(value: "Packable") -> ExtractedResult:
        """Extract a Packable - either self-contained or expanded."""
        from meshly.packable import PackableRefInfo

        # Self-contained: encode entire packable as zip bytes
        if value.is_contained:
            checksum = value.checksum  # Uses cached_property
            encoded = value.encode()
            ref_dict = PackableRefInfo(ref=checksum).model_dump(by_alias=True)
            return ExtractedResult(value=ref_dict, assets={checksum: encoded})

        # Expanded: use extract_basemodel to preserve type hints for List annotations
        return SerializationUtils.extract_basemodel(value)

    @staticmethod
    def _extract_resource(value: "Resource") -> ExtractedResult:
        """Extract a ResourceRef - gzip compress and store by checksum.

        Uses mtime=0 to ensure deterministic compression output for the same
        input data, which is critical for checksum-based caching.
        """
        checksum = value.checksum
        compressed = gzip.compress(value.data, compresslevel=6, mtime=0)
        # exclude_none instead of exclude_defaults to keep $type field
        ref_dict = value.model_dump(by_alias=True, exclude_none=True)
        return ExtractedResult(value=ref_dict, assets={checksum: compressed})

    @staticmethod
    def extract_basemodel(value: BaseModel, include_computed: bool = False) -> ExtractedResult:
        """Extract a BaseModel - iterate fields to preserve type info.

        Args:
            value: BaseModel instance to extract
            include_computed: If True, also extract computed fields
        """
        hints = typing.get_type_hints(type(value), include_extras=True)

        all_fields = list(type(value).model_fields)
        if include_computed:
            all_fields += list(type(value).model_computed_fields)

        # Collect non-None field values for extraction
        field_names = []
        field_values = []
        for name in all_fields:
            field_value = getattr(value, name, None)
            if field_value is not None:
                field_names.append(name)
                field_values.append(field_value)

        def _extract_field(args: tuple) -> tuple[str, ExtractedResult]:
            name, field_value = args
            if ArrayUtils.is_array(field_value):
                if ArrayUtils.is_list_annotation(hints.get(name)):
                    return name, ExtractedResult(value=np.asarray(field_value).tolist())
                encoding = ArrayUtils.get_array_encoding(hints.get(name))
                return name, SerializationUtils._extract_array_to_ref(field_value, encoding)
            return name, SerializationUtils.extract_value(field_value)

        pairs = list(zip(field_names, field_values))
        results = SerializationUtils._map_parallel(_extract_field, pairs)

        data: dict[str, JsonValue] = {}
        assets: dict[str, bytes] = {}
        for name, extracted in results:
            data[name] = extracted.value
            assets.update(extracted.assets)

        return ExtractedResult(value=data, assets=assets)
