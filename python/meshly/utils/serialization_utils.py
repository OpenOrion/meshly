"""Serialization utilities for packing/unpacking arrays and assets."""

import asyncio
import inspect
import json
from typing import Any

from pydantic import BaseModel

from ..array import ArrayMetadata, ArrayType, ArrayUtils, EncodedArray
from ..data_handler import AssetProvider, CachedAssetLoader
from .checksum_utils import ChecksumUtils


class SerializationUtils:
    """Utility class for serialization operations."""

    @staticmethod
    def pack_array(encoded: EncodedArray) -> bytes:
        """Pack an encoded array into bytes with metadata.

        Format: [4 bytes metadata length][metadata json][array data]

        Args:
            encoded: EncodedArray with metadata and data

        Returns:
            Packed bytes
        """
        metadata_json = json.dumps(encoded.metadata.model_dump()).encode("utf-8")
        return len(metadata_json).to_bytes(4, "little") + metadata_json + encoded.data

    @staticmethod
    def unpack_array(packed: bytes, array_type: ArrayType | None = None) -> Any:
        """Unpack bytes back to an array.

        Args:
            packed: Packed bytes from pack_array
            array_type: Target array type, or None to use stored type

        Returns:
            Decoded array (numpy or JAX)
        """
        metadata_len = int.from_bytes(packed[:4], "little")
        metadata_json = packed[4 : 4 + metadata_len].decode("utf-8")
        array_data = packed[4 + metadata_len :]

        metadata_dict = json.loads(metadata_json)
        metadata = ArrayMetadata(**metadata_dict)
        encoded = EncodedArray(data=array_data, metadata=metadata)

        decoded = ArrayUtils.decode_array(encoded)

        if array_type is not None:
            return ArrayUtils.convert_array(decoded, array_type)
        elif metadata.array_type != "numpy":
            return ArrayUtils.convert_array(decoded, metadata.array_type)
        return decoded

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
            cache_path = f"assets/{checksum}.bin"

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
        extensions: dict[str, str] | None = None,
    ) -> Any:
        """Recursively extract a value, replacing arrays and Packables with refs.

        Args:
            value: Value to extract
            assets: Dict to populate with encoded assets
            extensions: Optional dict to populate with file extensions for ResourceRefs

        Returns:
            Extracted value with $ref for arrays/Packables
        """
        # Import here to avoid circular imports
        from ..packable import Packable
        from ..resource import ResourceRef

        if ArrayUtils.is_array(value):
            encoded = ArrayUtils.encode_array(value)
            packed = SerializationUtils.pack_array(encoded)
            checksum = ChecksumUtils.compute_bytes_checksum(packed)
            assets[checksum] = packed
            return {"$ref": checksum}

        if isinstance(value, Packable):
            encoded = value.encode()
            checksum = ChecksumUtils.compute_bytes_checksum(encoded)
            assets[checksum] = encoded
            return {"$ref": checksum}

        if isinstance(value, ResourceRef):
            # Read file and store as asset - checksum is computed lazily by ResourceRef
            file_data = value.read_bytes()
            checksum = value.checksum  # Lazily computed from file data
            assert checksum is not None, "ResourceRef must have a checksum after reading bytes"
            assets[checksum] = file_data
            result = {"$ref": checksum}
            if value.ext:
                result["ext"] = value.ext
                if extensions is not None:
                    extensions[checksum] = value.ext
            return result

        if isinstance(value, dict):
            return {
                k: SerializationUtils.extract_value(v, assets, extensions) for k, v in value.items()
            }

        if isinstance(value, (list, tuple)):
            result = [SerializationUtils.extract_value(v, assets, extensions) for v in value]
            return result if isinstance(value, list) else tuple(result)

        if isinstance(value, BaseModel):
            extracted = {}
            for name in value.model_fields:
                field_value = getattr(value, name, None)
                if field_value is not None:
                    extracted[name] = SerializationUtils.extract_value(
                        field_value, assets, extensions
                    )
            return extracted

        return value
