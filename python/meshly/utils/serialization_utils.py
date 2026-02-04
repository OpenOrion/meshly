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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

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

        # Dicts: recursively extract values
        if isinstance(value, dict):
            items = [SerializationUtils.extract_value(v) for v in value.values()]
            return ExtractedResult(
                value=dict(zip(value.keys(), [e.value for e in items])),
                assets={k: v for e in items for k, v in e.assets.items()},
            )

        # Lists/tuples/sets: recursively extract items
        if isinstance(value, (list, tuple, set)):
            items = [SerializationUtils.extract_value(v) for v in value]
            result_value = [e.value for e in items]
            if isinstance(value, tuple):
                result_value = tuple(result_value)
            return ExtractedResult(
                value=result_value,
                assets={k: v for e in items for k, v in e.assets.items()},
            )

        # BaseModels: extract with $module metadata
        if isinstance(value, BaseModel):
            return SerializationUtils.extract_basemodel(value)

        # Primitives: pass through unchanged
        return ExtractedResult(value=value)

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
        
        # Expanded: recursively extract each field with $module metadata
        dumped = value.model_dump(mode='python')
        items = [(k, SerializationUtils.extract_value(v)) for k, v in dumped.items()]
        value_class = type(value)
        return ExtractedResult(
            value={**{k: e.value for k, e in items}, "$module": f"{value_class.__module__}.{value_class.__qualname__}"},
            assets={k: v for _, e in items for k, v in e.assets.items()},
        )

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
        hints = typing.get_type_hints(type(value), include_extras=True)
        data: dict[str, JsonValue] = {}
        assets: dict[str, bytes] = {}
        
        all_fields = list(type(value).model_fields)
        if include_computed:
            all_fields += list(type(value).model_computed_fields)
        
        for name in all_fields:
            field_value = getattr(value, name, None)
            if field_value is None:
                continue
            
            if ArrayUtils.is_array(field_value):
                encoding = ArrayUtils.get_array_encoding(hints.get(name))
                result = SerializationUtils._extract_array_to_ref(field_value, encoding)
                data[name] = result.value
                assets.update(result.assets)
            else:
                extracted = SerializationUtils.extract_value(field_value)
                data[name] = extracted.value
                assets.update(extracted.assets)
        
        value_class = type(value)
        data["$module"] = f"{value_class.__module__}.{value_class.__qualname__}"
        return ExtractedResult(value=data, assets=assets)
