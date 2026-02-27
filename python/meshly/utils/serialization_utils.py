"""Serialization utilities for arrays and assets.

This module provides utilities for:
- Extracting arrays, Packables, and ResourceRefs into serializable data with $ref checksums
- Fetching assets from various providers (dict, callable)
- Handling async asset fetchers
"""

import asyncio
import gzip
import inspect
import typing
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from pydantic import BaseModel, ConfigDict, Field

from meshly.array import ArrayUtils, ArrayEncoding
from meshly.common import AssetProvider
from meshly.utils.checksum_utils import ChecksumUtils
from meshly.utils.fork_pool import ForkPool

if TYPE_CHECKING:
    from meshly.packable import Packable
    from meshly.resource import Resource


@lru_cache(maxsize=None)
def _cached_type_hints(cls: type) -> dict[str, Any]:
    """Cache type hints per class to avoid repeated evaluation."""
    return typing.get_type_hints(cls, include_extras=True)


# =============================================================================
# Type Definitions
# =============================================================================

JsonValue = Union[None, bool, int, float, str, list, tuple, dict]
"""JSON-serializable types for extracted values."""


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
        # Use model_dump for maintainability - new ArrayRefInfo fields auto-included
        ref_dict = extracted.info.model_dump(exclude_defaults=True, exclude_none=True, by_alias=True)
        ref_dict["$ref"] = checksum
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

        # Dicts: recursively extract values (parallelize for Packable values)
        if isinstance(value, dict):
            keys = list(value.keys())
            items = SerializationUtils._extract_list_items(list(value.values()))
            return ExtractedResult(
                value=dict(zip(keys, [e.value for e in items])),
                assets={k: v for e in items for k, v in e.assets.items()},
            )

        # Lists/tuples/sets: recursively extract items
        if isinstance(value, (list, tuple, set)):
            items = SerializationUtils._extract_list_items(list(value))
            result_value = [e.value for e in items]
            if isinstance(value, tuple):
                result_value = tuple(result_value)
            return ExtractedResult(
                value=result_value,
                assets={k: v for e in items for k, v in e.assets.items()},
            )

        # BaseModels: extract fields
        if isinstance(value, BaseModel):
            return SerializationUtils.extract_basemodel(value)

        # Primitives: pass through unchanged
        return ExtractedResult(value=value)

    @staticmethod
    def _extract_list_items(items: list) -> list[ExtractedResult]:
        """Extract list items, parallelizing when items are Packables.
        
        Uses fork-based parallelism for lists of Packables (e.g., meshes)
        which can be independently serialized. Falls back to sequential
        for mixed types or non-Packable lists.
        """
        from meshly.packable import Packable
        
        if not items:
            return []
        
        # Check if all items are Packables (common for mesh lists)
        # Only parallelize for larger lists where overhead is worth it
        MIN_ITEMS_FOR_PARALLEL = 50
        all_packables = len(items) >= MIN_ITEMS_FOR_PARALLEL and all(
            isinstance(item, Packable) for item in items
        )
        
        if all_packables:
            # Parallel extraction for Packable lists
            return ForkPool.map(
                SerializationUtils._extract_subpackable, 
                items, 
                min_items_for_parallel=MIN_ITEMS_FOR_PARALLEL,
            )
        
        # Sequential fallback for mixed types
        return [SerializationUtils.extract_value(v) for v in items]

    # -------------------------------------------------------------------------
    # Type-Specific Extraction Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_subpackable(value: "Packable") -> ExtractedResult:
        """Extract a Packable - either self-contained or expanded."""
        from meshly.packable import PackableRefInfo
        
        # Self-contained: encode entire packable as zip bytes
        if value.is_contained:
            encoded = value.encode()  # Uses cached _encoded property
            checksum = value.checksum  # Uses cached checksum from same encoded bytes
            ref_dict = PackableRefInfo(ref=checksum).model_dump(by_alias=True)
            return ExtractedResult(value=ref_dict, assets={checksum: encoded})
        
        # Expanded: use extract_basemodel to preserve type annotations (e.g., List)
        return SerializationUtils.extract_basemodel(value)

    @staticmethod
    def _extract_resource(value: "Resource") -> ExtractedResult:
        """Extract a ResourceRef - gzip compress and store by checksum."""
        checksum = value.checksum
        compressed = gzip.compress(value.data, compresslevel=6)
        ref_dict = value.model_dump(by_alias=True, exclude_defaults=True)
        return ExtractedResult(value=ref_dict, assets={checksum: compressed})

    @staticmethod
    def extract_basemodel(value: BaseModel, include_computed: bool = False) -> ExtractedResult:
        """Extract a BaseModel - iterate fields to preserve type info.
        
        Args:
            value: BaseModel instance to extract
            include_computed: If True, also extract computed fields
        """
        hints = _cached_type_hints(type(value))
        data: dict[str, JsonValue] = {}
        assets: dict[str, bytes] = {}
        
        model_fields = type(value).model_fields
        all_fields = list(model_fields)
        if include_computed:
            all_fields += list(type(value).model_computed_fields)
        
        for name in all_fields:
            # Respect Pydantic's exclude=True on fields
            field_info = model_fields.get(name)
            if field_info is not None and field_info.exclude:
                continue
            
            field_value = getattr(value, name, None)
            if field_value is None:
                continue
            
            if ArrayUtils.is_array(field_value):
                if ArrayUtils.is_list_annotation(hints.get(name)):
                    data[name] = np.asarray(field_value).tolist()
                else:
                    encoding = ArrayUtils.get_array_encoding(hints.get(name))
                    result = SerializationUtils._extract_array_to_ref(field_value, encoding)
                    data[name] = result.value
                    assets.update(result.assets)
            else:
                extracted = SerializationUtils.extract_value(field_value)
                data[name] = extracted.value
                assets.update(extracted.assets)
        
        return ExtractedResult(value=data, assets=assets)
