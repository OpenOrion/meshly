"""Serialization utilities for arrays and assets."""

import asyncio
import inspect
from typing import Any, Union

from pydantic import BaseModel

from ..array import ArrayRefMetadata, ArrayType, ArrayUtils, ResourceRefMetadata, PackableRefMetadata
from ..constants import ExportConstants
from ..data_handler import AssetProvider, CachedAssetLoader
from .checksum_utils import ChecksumUtils


class SerializationUtils:
    """Utility class for serialization operations."""

    @staticmethod
    def get_asset(assets: AssetProvider, checksum: str) -> bytes:
        """Get asset bytes from a provider (dict, callable, or CachedAssetLoader).

        Supports both sync and async callables.

        Args:
            assets: Asset provider (dict, callable, or CachedAssetLoader)
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

    @staticmethod
    def get_cached_asset(
        assets: AssetProvider,
        checksum: str,
    ) -> bytes:
        """Get asset bytes with caching support for CachedAssetLoader.

        Args:
            assets: Asset provider
            checksum: Asset checksum

        Returns:
            Asset bytes

        Raises:
            KeyError: If asset not found
        """
        if isinstance(assets, CachedAssetLoader):
            cache_path = ExportConstants.asset_path(checksum)

            try:
                return assets.cache.read_binary(cache_path)
            except (KeyError, FileNotFoundError):
                pass

            result = assets.fetch(checksum)
            if inspect.isawaitable(result):
                result = asyncio.get_event_loop().run_until_complete(result)

            if result is None:
                try:
                    return assets.cache.read_binary(cache_path)
                except (KeyError, FileNotFoundError):
                    raise KeyError(f"Asset '{checksum}' not found in remote or cache")

            assets.cache.write_binary(cache_path, result)
            return result

        if callable(assets):
            result = assets(checksum)
            if inspect.isawaitable(result):
                result = asyncio.get_event_loop().run_until_complete(result)
            if result is None:
                raise KeyError(f"Asset fetcher returned None for checksum '{checksum}'")
            return result

        if checksum not in assets:
            raise KeyError(f"Missing asset with checksum '{checksum}'")
        return assets[checksum]

    @staticmethod
    def extract_value(
        value: Any,
        assets: dict[str, bytes],
        extensions: Union[dict[str, str], None] = None,
        encoding_type: Union[str, None] = None,
        vertex_count: Union[int, None] = None,
    ) -> Any:
        """Recursively extract a value, replacing arrays and Packables with refs.

        Args:
            value: Value to extract
            assets: Dict to populate with encoded assets
            extensions: Optional dict to populate with file extensions for ResourceRefs
            encoding_type: Optional encoding hint for arrays (array, vertex_buffer, index_sequence)
            vertex_count: For index_sequence type, the vertex count

        Returns:
            Extracted value with $ref and flat metadata for arrays
        """
        # Import here to avoid circular imports
        from ..packable import Packable
        from ..resource import ResourceRef

        if ArrayUtils.is_array(value):
            effective_type = encoding_type or "array"
            encoded_bytes, metadata = ArrayUtils.encode_with_schema(
                value, 
                encoding_type=effective_type,
                vertex_count=vertex_count,
            )
            checksum = ChecksumUtils.compute_bytes_checksum(encoded_bytes)
            assets[checksum] = encoded_bytes
            # Set ref on the metadata object
            metadata.ref = checksum
            return metadata.model_dump(by_alias=True, exclude_none=True)

        if isinstance(value, Packable):
            if value._self_contained:
                # Self-contained: encode to zip, single $ref
                encoded = value.encode()
                checksum = ChecksumUtils.compute_bytes_checksum(encoded)
                assets[checksum] = encoded
                ref_metadata = PackableRefMetadata(ref=checksum)
                return ref_metadata.model_dump(by_alias=True)
            else:
                # Expanded: recurse into fields like any BaseModel
                dumped = value.model_dump(mode='python')
                extracted = {}
                for k, v in dumped.items():
                    extracted[k] = SerializationUtils.extract_value(v, assets, extensions)
                value_class = type(value)
                extracted["$module"] = f"{value_class.__module__}.{value_class.__qualname__}"
                return extracted

        if isinstance(value, ResourceRef):
            # Read file and store as gzip-compressed bytes
            import gzip
            from pathlib import Path
            file_data = value.read_bytes()
            checksum = value.checksum
            assert checksum is not None, "ResourceRef must have a checksum after reading bytes"
            
            # Compress with gzip (better for arbitrary file data than meshoptimizer)
            compressed = gzip.compress(file_data, compresslevel=6)
            assets[checksum] = compressed
            
            # Get filename from path
            name = Path(value.path).name if value.path else None
            
            # Store ref with metadata
            ref_metadata = ResourceRefMetadata(ref=checksum, name=name)
            if value.ext and extensions is not None:
                extensions[checksum] = value.ext
            return ref_metadata.model_dump(by_alias=True, exclude_none=True)

        if isinstance(value, dict):
            return {
                k: SerializationUtils.extract_value(v, assets, extensions) for k, v in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            result = [SerializationUtils.extract_value(v, assets, extensions) for v in value]
            if isinstance(value, tuple):
                return tuple(result)
            elif isinstance(value, set):
                # Sets are converted to lists for JSON serialization
                return result
            return result

        if isinstance(value, BaseModel):
            # Iterate through actual field values to preserve type information
            # (model_dump would lose Packable types, converting them to dicts)
            extracted = {}
            for field_name in type(value).model_fields:
                field_value = getattr(value, field_name, None)
                if field_value is not None:
                    # Use extract_field_value to respect type annotations
                    extracted[field_name] = SerializationUtils.extract_field_value(
                        value, field_name, field_value, assets, extensions
                    )
            # Include $module for nested BaseModel reconstruction
            value_class = type(value)
            extracted["$module"] = f"{value_class.__module__}.{value_class.__qualname__}"
            return extracted

        return value

    @staticmethod
    def extract_field_value(
        model: "BaseModel",
        field_name: str,
        value: Any,
        assets: dict[str, bytes],
        extensions: Union[dict[str, str], None] = None,
    ) -> Any:
        """Extract a field value with encoding from the field's type annotation.
        
        Uses get_array_encoding() to extract encoding from Annotated types
        (Array, VertexBuffer, IndexSequence).
        
        Args:
            model: The model instance containing the field
            field_name: Name of the field
            value: Value to extract
            assets: Dict to populate with encoded assets
            extensions: Optional dict for ResourceRef extensions
            
        Returns:
            Extracted value with $ref and metadata
        """
        if not ArrayUtils.is_array(value):
            return SerializationUtils.extract_value(value, assets, extensions)
        
        # Get encoding from field's type annotation
        # Use get_type_hints with include_extras=True to preserve Annotated metadata
        import typing
        model_class = type(model)
        hints = typing.get_type_hints(model_class, include_extras=True)
        encoding = ArrayUtils.get_array_encoding(hints.get(field_name)) if field_name in hints else "array"
        
        # For index_sequence, try to get vertex_count from the model
        vertex_count = None
        if encoding == "index_sequence" and hasattr(model, "vertex_count"):
            vertex_count = model.vertex_count
            
        return SerializationUtils.extract_value(
            value, assets, extensions, 
            encoding_type=encoding,
            vertex_count=vertex_count,
        )
