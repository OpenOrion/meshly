"""Base packable class for encoded array storage.

This module provides a base class that handles automatic numpy/JAX array
detection and serialization to zip files. Classes like Mesh inherit from
this base to get automatic array encoding/decoding support.

Custom data classes can inherit from Packable to store simulation
results, time-series data, or any structured data with numpy arrays.

Packables cannot contain nested Packables. For composite structures,
use the extract() and reconstruct() methods to handle asset management.
"""

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generic,
    Optional,
    Set,
    Type,
    Any,
    TypeVar,
    Union,
)
from pydantic import BaseModel, Field, computed_field
from .array import ArrayUtils, ArrayType, Array
from .common import PathLike
from .data_handler import AssetProvider, CachedAssetLoader, DataHandler

TModel = TypeVar("TModel", bound=BaseModel)


class PackableMetadata(BaseModel):
    """Metadata for a Packable saved to zip."""
    field_data: Dict[str, Any] = Field(
        default_factory=dict, description="Non-array field values")


TPackableMetadata = TypeVar("TPackableMetadata", bound=PackableMetadata)
TPackable = TypeVar("TPackable", bound="Packable")
FieldValue = TypeVar("FieldValue")  # Value type for custom fields


@dataclass
class SerializedPackableData:
    """Result of extracting a Packable for serialization.
    
    Contains the serializable data dict with checksum references,
    plus the encoded assets (arrays as bytes).
    """
    data: Dict[str, Any]
    """Serializable dict with primitive fields and checksum refs for arrays"""
    assets: Dict[str, bytes]
    """Map of checksum -> encoded bytes for all arrays"""


class LazyModel(Generic[TModel]):
    """
    Lazy proxy for a Pydantic BaseModel that defers asset loading until field access.
    
    Fields containing $ref references are not resolved until accessed,
    allowing for truly lazy loading from external storage.
    
    Example:
        def fetch_asset(checksum: str) -> bytes:
            return cloud_storage.download(checksum)
        
        lazy = Packable.reconstruct(SimulationCase, data, fetch_asset)
        # No assets loaded yet
        
        temp = lazy.temperature  # NOW the temperature asset is fetched
        vel = lazy.velocity      # NOW the velocity asset is fetched
        
        # With a cache handler for persistence:
        cache = DataHandler.create(Path("./cache"))
        loader = CachedAssetLoader(fetch_asset, cache)
        lazy = Packable.reconstruct(SimulationCase, data, loader)
    """
    
    __slots__ = ('_model_class', '_data', '_assets', '_array_type', '_cache', '_resolved')
    
    def __init__(
        self,
        model_class: Type[TModel],
        data: Dict[str, Any],
        assets: AssetProvider,
        array_type: Optional[ArrayType] = None,
    ):
        object.__setattr__(self, '_model_class', model_class)
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, '_assets', assets)
        object.__setattr__(self, '_array_type', array_type)
        object.__setattr__(self, '_cache', {})
        object.__setattr__(self, '_resolved', None)
    
    def _get_cached_asset(self, checksum: str) -> bytes:
        """Get asset bytes, using cache if CachedAssetLoader is provided."""
        assets = object.__getattribute__(self, '_assets')
        
        # Handle CachedAssetLoader
        if isinstance(assets, CachedAssetLoader):
            cache_path = f"assets/{checksum}.bin"
            
            # Try to read from cache first
            try:
                return assets.cache.read_binary(cache_path)
            except (KeyError, FileNotFoundError):
                pass
            
            # Fetch from provider
            asset_bytes = assets.fetch(checksum)
            
            # Store in cache
            assets.cache.write_binary(cache_path, asset_bytes)
            return asset_bytes
        
        # Handle plain callable
        if callable(assets):
            return assets(checksum)
        
        # Handle dict
        if checksum not in assets:
            raise KeyError(f"Missing asset with checksum '{checksum}'")
        return assets[checksum]
    
    def __getattr__(self, name: str) -> Any:
        # Check cache first
        cache = object.__getattribute__(self, '_cache')
        if name in cache:
            return cache[name]
        
        model_class = object.__getattribute__(self, '_model_class')
        data = object.__getattribute__(self, '_data')
        array_type = object.__getattribute__(self, '_array_type')
        
        # Check if it's a model field
        if name not in model_class.model_fields:
            raise AttributeError(f"'{model_class.__name__}' has no attribute '{name}'")
        
        if name not in data:
            return None
        
        field_value = data[name]
        field_type = model_class.model_fields[name].annotation
        
        # Resolve this specific field using our caching asset getter
        resolved = Packable._resolve_value_with_type(
            field_value, field_type, self._get_cached_asset, array_type
        )
        
        # Cache the resolved value
        cache[name] = resolved
        return resolved
    
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LazyModel is read-only. Use resolve() to get a mutable model.")
    
    def resolve(self) -> TModel:
        """
        Fully resolve all fields and return the actual Pydantic model.
        
        This will fetch all remaining assets that haven't been accessed yet.
        """
        resolved = object.__getattribute__(self, '_resolved')
        if resolved is not None:
            return resolved
        
        model_class = object.__getattribute__(self, '_model_class')
        data = object.__getattribute__(self, '_data')
        array_type = object.__getattribute__(self, '_array_type')
        cache = object.__getattribute__(self, '_cache')
        
        # Resolve all fields, using cache where available
        resolved_data = {}
        for field_name, field_info in model_class.model_fields.items():
            if field_name in cache:
                resolved_data[field_name] = cache[field_name]
            elif field_name in data:
                resolved_data[field_name] = Packable._resolve_value_with_type(
                    data[field_name], field_info.annotation, self._get_cached_asset, array_type
                )
        
        result = model_class(**resolved_data)
        object.__setattr__(self, '_resolved', result)
        return result
    
    def __repr__(self) -> str:
        model_class = object.__getattribute__(self, '_model_class')
        cache = object.__getattribute__(self, '_cache')
        data = object.__getattribute__(self, '_data')
        loaded = list(cache.keys())
        pending = [k for k in data.keys() if k not in cache]
        return f"LazyModel[{model_class.__name__}](loaded={loaded}, pending={pending})"


@dataclass
class CustomFieldConfig(Generic[FieldValue, TPackableMetadata]):
    """Configuration for custom field encoding/decoding."""
    file_name: str
    """File name in zip (without .bin extension)"""
    encode: Callable[[FieldValue, Any], bytes]
    """Encoder function: (value, instance) -> bytes"""
    decode: Callable[[bytes, TPackableMetadata,
                      Optional[ArrayType]], FieldValue]
    """Decoder function: (bytes, metadata, array_type) -> value"""
    optional: bool = False
    """Whether the field is optional (won't throw if missing)"""


class Packable(BaseModel):
    """
    Base class for data containers with automatic array serialization.

    Subclasses can define numpy array attributes which will be automatically
    detected, encoded, and saved to zip files. Non-array fields are preserved
    in metadata.

    Packables cannot contain nested Packables. For composite structures,
    use extract() to get a serializable dict with asset references, and
    reconstruct() to rebuild from the dict and assets.

    Example:
        class SimulationResult(Packable):
            time: float
            temperature: np.ndarray
            velocity: np.ndarray

        result = SimulationResult(
            time=0.1,
            temperature=np.array([300.0, 301.0, 302.0]),
            velocity=np.zeros((3, 3))
        )
        result.save_to_zip("result.zip")
        
        # Load using the specific class
        loaded = SimulationResult.load_from_zip("result.zip")
        
        # Or use extract/reconstruct for custom asset management
        extracted = result.extract()
        # extracted.data contains {"time": 0.1, "temperature": {"$ref": "abc123"}, ...}
        # extracted.assets contains {"abc123": <encoded bytes>, ...}
        rebuilt = SimulationResult.reconstruct(extracted.data, extracted.assets)
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_no_direct_packable_fields()

    def _validate_no_direct_packable_fields(self) -> None:
        """Validate that this Packable has no direct Packable fields.
        
        Packables nested inside dicts or other BaseModels are allowed and will
        be handled by extract(). Only direct Packable fields are prohibited.
        """
        for field_name in type(self).model_fields:
            if field_name in self.__private_attributes__:
                continue
            value = getattr(self, field_name, None)
            if value is None:
                continue
            
            # Only reject direct Packable fields
            if isinstance(value, Packable):
                raise TypeError(
                    f"Direct Packable fields are not allowed. Field '{field_name}' "
                    f"contains a {type(value).__name__}. Packables can be nested "
                    "inside dicts or other BaseModels, and extract() will handle them."
                )

    @computed_field
    @cached_property
    def checksum(self) -> str:
        """
        Compute SHA256 checksum of the encoded content.
        Returns:
            16-character hex string (first 64 bits of SHA256)
        """
        return hashlib.sha256(self.encode()).hexdigest()[:16]

    @property
    def array_fields(self) -> Set[str]:
        """Get all array field paths, including nested arrays in dicts/BaseModels."""
        result = set()
        for field_name in type(self).model_fields:
            if field_name in self.__private_attributes__:
                continue
            value = getattr(self, field_name, None)
            if value is not None:
                result.update(ArrayUtils.extract_nested_arrays(
                    value, field_name).keys())
        return result

    def _extract_non_array_fields(self) -> Dict[str, Any]:
        """Extract non-array field values for metadata, preserving BaseModel type info."""
        model_data = {}
        direct_arrays = {f for f in self.array_fields if "." not in f}
        for name in type(self).model_fields:
            if name in self.__private_attributes__ or name in direct_arrays:
                continue
            value = getattr(self, name, None)
            if value is not None and not ArrayUtils.is_array(value):
                extracted = ArrayUtils.extract_non_arrays(value)
                if extracted is not None:
                    model_data[name] = extracted
        return model_data

    def _create_metadata(self, field_data: Dict[str, Any]) -> PackableMetadata:
        """
        Create metadata for this Packable.

        Subclasses can override this to return custom metadata types.

        Args:
            field_data: Non-array field values to include in metadata

        Returns:
            PackableMetadata (or subclass) instance
        """
        return PackableMetadata(
            field_data=field_data,
        )

    @classmethod
    def load_metadata(
        cls,
        handler: DataHandler,
        metadata_cls: Type[TPackableMetadata] = PackableMetadata
    ) -> TPackableMetadata:
        """
        Load and validate metadata using a read handler.

        Args:
            handler: ReadHandler for reading files
            metadata_cls: The metadata class to use for parsing (default: PackableMetadata)

        Returns:
            Metadata object of the specified type
        """
        metadata_text = handler.read_text("metadata.json")
        metadata_dict = json.loads(metadata_text)
        return metadata_cls(**metadata_dict)

    def save_to_zip(
        self,
        destination: Union[PathLike, BytesIO],
    ) -> None:
        """
        Save this container to a zip file.

        Args:
            destination: Path to the output zip file or BytesIO buffer
        """
        encoded = self.encode()
        if isinstance(destination, BytesIO):
            destination.write(encoded)
        else:
            Path(destination).write_bytes(encoded)

    @classmethod
    def load_from_zip(
        cls: Type[TPackable],
        source: Union[PathLike, BytesIO],
        array_type: Optional[ArrayType] = None,
    ) -> TPackable:
        """
        Load a Packable from a zip file.

        Args:
            source: Path to the input zip file or BytesIO object
            array_type: Array backend to use ("numpy" or "jax"). If None (default),
                       uses the array_type stored in each array's metadata,
                       preserving the original array types that were saved.

        Returns:
            Loaded Packable instance
            
        Raises:
            TypeError: If called on base Packable class instead of a subclass

        Example:
            mesh = Mesh.load_from_zip("mesh.zip")
        """
        if isinstance(source, BytesIO):
            source.seek(0)
            return cls.decode(source.read(), array_type)
        else:
            with open(source, "rb") as f:
                return cls.decode(f.read(), array_type)

    @classmethod
    def _get_custom_fields(cls) -> Dict[str, CustomFieldConfig]:
        """
        Get custom field configurations for this class.

        Subclasses override this to define custom encoders/decoders.

        Returns:
            Dict mapping field names to CustomFieldConfig objects
        """
        return {}

    @classmethod
    def _get_custom_field_names(cls) -> Set[str]:
        """Get set of field names that have custom encoding/decoding."""
        return set(cls._get_custom_fields().keys())

    @classmethod
    def _decode_custom_fields(
        cls,
        handler: DataHandler,
        metadata: PackableMetadata,
        data: Dict[str, Any],
        array_type: Optional[ArrayType] = None
    ) -> None:
        """Decode fields with custom decoders."""
        for field_name, config in cls._get_custom_fields().items():
            try:
                encoded_bytes = handler.read_binary(f"{config.file_name}.bin")
                data[field_name] = config.decode(
                    encoded_bytes, metadata, array_type)
            except (KeyError, FileNotFoundError):
                if not config.optional:
                    raise ValueError(
                        f"Required custom field '{field_name}' ({config.file_name}.bin) not found in zip")

    @classmethod
    def _load_standard_arrays(
        cls,
        handler: DataHandler,
        data: Dict[str, Any],
        skip_fields: Set[str],
        array_type: Optional[ArrayType] = None
    ) -> None:
        """Load standard arrays from arrays/ folder, skipping custom fields."""
        try:
            all_files = handler.list_files("arrays", recursive=True)
        except (KeyError, FileNotFoundError):
            return

        for file_path in all_files:
            file_str = str(file_path)
            if not file_str.endswith("/array.bin"):
                continue

            # Extract array name: "arrays/markerIndices/boundary/array.bin" -> "markerIndices.boundary"
            array_path = file_str[7:-10]  # Remove "arrays/" and "/array.bin"
            name = array_path.replace("/", ".")

            # Skip custom fields
            base_field = name.split(".")[0]
            if base_field in skip_fields:
                continue

            decoded = ArrayUtils.load_array(handler, name, array_type)

            if "." in name:
                # Nested array - build nested structure
                parts = name.split(".")
                current = data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = decoded
            else:
                # Flat array
                data[name] = decoded

    def _encode_standard_arrays(self, skip_fields: Set[str]) -> Dict[str, bytes]:
        """Encode standard arrays, skipping custom fields."""
        encoded_arrays = {}

        for field_name in self.array_fields:
            # Skip fields with custom encoding
            if field_name in skip_fields:
                continue

            # Handle nested array paths (e.g., "textures.diffuse")
            if "." in field_name:
                parts = field_name.split(".")
                obj = self
                for part in parts[:-1]:
                    if isinstance(obj, dict):
                        obj = obj[part]
                    else:
                        obj = getattr(obj, part)

                if isinstance(obj, dict):
                    array = obj[parts[-1]]
                else:
                    array = getattr(obj, parts[-1])

                if ArrayUtils.is_array(array):
                    encoded_arrays[field_name] = ArrayUtils.encode_array(array)
            else:
                # Handle direct array fields
                try:
                    array = getattr(self, field_name)
                    if ArrayUtils.is_array(array):
                        encoded_arrays[field_name] = ArrayUtils.encode_array(
                            array)
                except AttributeError:
                    pass

        return encoded_arrays

    def _encode_custom_fields(self, handler: DataHandler) -> None:
        """Encode fields with custom encoders."""
        for field_name, config in self._get_custom_fields().items():
            value = getattr(self, field_name)
            if value is not None:
                encoded_bytes = config.encode(value, self)
                handler.write_binary(f"{config.file_name}.bin", encoded_bytes)

    def encode(self) -> bytes:
        """
        Serialize this Packable to bytes (zip format).

        Returns:
            Bytes containing the zip-encoded data
        """
        custom_field_names = self._get_custom_field_names()

        # Encode standard arrays
        encoded_arrays = self._encode_standard_arrays(custom_field_names)

        # Create metadata
        field_data = self._extract_non_array_fields()
        metadata = self._create_metadata(field_data)

        # Write to zip
        destination = BytesIO()
        handler = DataHandler.create(destination)

        # Save standard arrays
        for name in sorted(encoded_arrays.keys()):
            ArrayUtils.save_array(handler, name, encoded_arrays[name])

        # Save custom encoded fields
        self._encode_custom_fields(handler)

        # Save metadata
        handler.write_text(
            "metadata.json",
            json.dumps(metadata.model_dump(), indent=2, sort_keys=True),
        )

        handler.finalize()
        return destination.getvalue()

    @classmethod
    def decode(
        cls: Type[TPackable],
        buf: bytes,
        array_type: Optional[ArrayType] = None,
    ) -> TPackable:
        """
        Deserialize a Packable from bytes.

        Args:
            buf: Bytes containing the zip-encoded data
            array_type: Array backend to use. If None (default), uses the
                       array_type stored in each array's metadata.

        Returns:
            Loaded Packable instance
            
        Raises:
            TypeError: If called on base Packable class instead of a subclass
        """
        if cls is Packable:
            raise TypeError(
                "Cannot decode on base Packable class. "
                "Use the specific subclass: MyClass.decode(...)"
            )
        
        handler = DataHandler.create(BytesIO(buf))
        metadata = cls.load_metadata(handler)

        # Fields to skip when loading standard arrays
        skip_fields = cls._get_custom_field_names()

        data: Dict[str, Any] = {}

        # Decode custom fields first
        cls._decode_custom_fields(handler, metadata, data, array_type)

        # Load standard arrays
        cls._load_standard_arrays(handler, data, skip_fields, array_type)

        # Merge non-array fields from metadata using schema-aware reconstruction
        if metadata.field_data:
            cls._merge_field_data_with_schema(cls, data, metadata.field_data)

        return cls(**data)

    @staticmethod
    def extract(obj: BaseModel) -> SerializedPackableData:
        """
        Extract arrays and Packables from a Pydantic BaseModel into serializable data and assets.
        
        Args:
            obj: A Pydantic BaseModel instance (including Packable subclasses)
        
        Returns an ExtractedPackable with:
        - data: A JSON-serializable dict with `{"$ref": checksum}` for arrays/Packables
        - assets: A dict mapping checksums to encoded bytes
        
        Arrays and nested Packables are stored as assets. The type information comes
        from the Pydantic schema when reconstructing, so no class/module info is stored.
        
        Example:
            mesh = Mesh(vertices=..., indices=...)
            extracted = Packable.extract(mesh)
            # extracted.data = {"vertices": {"$ref": "abc..."}, "indices": {"$ref": "def..."}}
            
            rebuilt = Mesh.reconstruct(extracted.data, extracted.assets)
        """
        if not isinstance(obj, BaseModel):
            raise TypeError(
                f"extract() requires a Pydantic BaseModel, got {type(obj).__name__}. "
                "Use Pydantic models for type-safe extraction and reconstruction."
            )
        
        assets: Dict[str, bytes] = {}
        data: Dict[str, Any] = {}
        
        for field_name in type(obj).model_fields:
            if hasattr(obj, '__private_attributes__') and field_name in obj.__private_attributes__:
                continue
            value = getattr(obj, field_name, None)
            if value is None:
                continue
            data[field_name] = Packable._extract_value(value, assets)
        
        return SerializedPackableData(data=data, assets=assets)
    
    @staticmethod
    def _extract_value(value: Any, assets: Dict[str, bytes]) -> Any:
        """Recursively extract a value, replacing arrays and nested Packables with refs."""
        # Handle arrays
        if ArrayUtils.is_array(value):
            encoded = ArrayUtils.encode_array(value)
            # Pack metadata + data together as bytes for the asset
            metadata_json = json.dumps(encoded.metadata.model_dump()).encode('utf-8')
            # Format: [4 bytes metadata length][metadata json][array data]
            packed = len(metadata_json).to_bytes(4, 'little') + metadata_json + encoded.data
            checksum = hashlib.sha256(packed).hexdigest()[:16]
            assets[checksum] = packed
            return {"$ref": checksum}
        
        # Handle Packables - extract as encoded zip bytes
        if isinstance(value, Packable):
            encoded = value.encode()
            checksum = hashlib.sha256(encoded).hexdigest()[:16]
            assets[checksum] = encoded
            return {"$ref": checksum}
        
        # Handle dicts
        if isinstance(value, dict):
            return {k: Packable._extract_value(v, assets) for k, v in value.items()}
        
        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            result = [Packable._extract_value(v, assets) for v in value]
            return result if isinstance(value, list) else tuple(result)
        
        # Handle non-Packable BaseModels - recursively extract their fields
        if isinstance(value, BaseModel):
            extracted = {}
            for name in value.model_fields:
                field_value = getattr(value, name, None)
                if field_value is not None:
                    extracted[name] = Packable._extract_value(field_value, assets)
            return extracted
        
        # Primitive value - return as-is
        return value

    @staticmethod
    def _get_asset(assets: AssetProvider, checksum: str) -> bytes:
        """Get asset bytes from either a dict or callable provider."""
        if callable(assets):
            return assets(checksum)
        if checksum not in assets:
            raise KeyError(f"Missing asset with checksum '{checksum}'")
        return assets[checksum]

    @staticmethod
    def reconstruct(
        model_class: Type[TModel],
        data: Dict[str, Any],
        assets: AssetProvider,
        array_type: Optional[ArrayType] = None,
    ) -> Union[TModel, LazyModel[TModel]]:
        """
        Reconstruct a Pydantic BaseModel from extracted data and assets.
        
        Uses the class's Pydantic schema to determine types for nested fields,
        so no runtime type information needs to be stored in the data.
        
        If assets is a dict, all assets are loaded immediately and the actual
        model is returned. If assets is a callable or CachedAssetLoader, a 
        LazyModel proxy is returned that defers asset loading until field access.
        
        Args:
            model_class: The Pydantic BaseModel class to reconstruct
            data: The data dict from extract(), with $ref references
            assets: One of:
                   - Dict mapping checksums to bytes (eager loading)
                   - Callable that takes a checksum and returns bytes (lazy loading)
                   - CachedAssetLoader with fetch callable and cache handler (lazy + disk cache)
            array_type: Array backend to use. If None, uses the type stored
                       in each array's metadata.
        
        Returns:
            - If assets is a dict: Reconstructed BaseModel instance (eager)
            - If assets is callable/CachedAssetLoader: LazyModel proxy that loads on demand
            
        Raises:
            KeyError: If a referenced asset is missing (for dict assets, raised immediately;
                     for callable assets, raised on field access)
            
        Example:
            extracted = Packable.extract(simulation_case)
            
            # Eager loading with dict - returns actual model
            rebuilt = Packable.reconstruct(SimulationCase, extracted.data, extracted.assets)
            
            # Lazy loading with callable - returns LazyModel
            def fetch_asset(checksum: str) -> bytes:
                return storage.get(checksum)
            lazy = Packable.reconstruct(SimulationCase, data, fetch_asset)
            
            # Lazy loading with disk cache
            cache = DataHandler.create(Path("./cache"))
            loader = CachedAssetLoader(fetch_asset, cache)
            lazy = Packable.reconstruct(SimulationCase, data, loader)
            
            print(lazy.time)         # Primitive field, no fetch needed
            print(lazy.temperature)  # Fetches and caches temperature asset
            model = lazy.resolve()   # Get full Pydantic model
        """
        if callable(assets) or isinstance(assets, CachedAssetLoader):
            return LazyModel(model_class, data, assets, array_type)
        
        resolved_data = Packable._resolve_refs_with_schema(
            model_class, data, assets, array_type
        )
        return model_class(**resolved_data)
    
    @staticmethod
    def _decode_packed_array(packed: bytes, array_type: Optional[ArrayType]) -> Any:
        """Decode a packed array asset (metadata + data) back to an array."""
        from .array import EncodedArray, ArrayMetadata
        
        # Unpack: [4 bytes metadata length][metadata json][array data]
        metadata_len = int.from_bytes(packed[:4], 'little')
        metadata_json = packed[4:4+metadata_len].decode('utf-8')
        array_data = packed[4+metadata_len:]
        
        metadata_dict = json.loads(metadata_json)
        metadata = ArrayMetadata(**metadata_dict)
        encoded = EncodedArray(data=array_data, metadata=metadata)
        
        decoded = ArrayUtils.decode_array(encoded)
        
        # Convert to requested array type if specified
        if array_type is not None:
            return ArrayUtils.convert_array(decoded, array_type)
        elif metadata.array_type != "numpy":
            return ArrayUtils.convert_array(decoded, metadata.array_type)
        return decoded

    @staticmethod
    def _resolve_refs_with_schema(
        model_class: Type[BaseModel],
        data: Dict[str, Any],
        assets: AssetProvider,
        array_type: Optional[ArrayType],
    ) -> Dict[str, Any]:
        """
        Resolve $ref references using Pydantic schema for type information.
        
        Uses model_class.model_fields to determine the expected type for each field,
        so no class/module information needs to be stored in the data.
        """
        result = {}
        
        for field_name, field_info in model_class.model_fields.items():
            if field_name not in data:
                continue
            
            field_value = data[field_name]
            field_type = field_info.annotation
            
            result[field_name] = Packable._resolve_value_with_type(
                field_value, field_type, assets, array_type
            )
        
        return result
    
    @staticmethod
    def _resolve_value_with_type(
        value: Any,
        expected_type: Any,
        assets: AssetProvider,
        array_type: Optional[ArrayType],
    ) -> Any:
        """Resolve a value using the expected type from Pydantic schema."""
        from typing import get_origin, get_args, Union
        
        if value is None:
            return None
        
        # Handle $ref - decode based on expected type
        if isinstance(value, dict) and "$ref" in value:
            checksum = value["$ref"]
            asset_bytes = Packable._get_asset(assets, checksum)
            
            # Determine if this is a Packable or array based on expected_type
            origin = get_origin(expected_type)
            
            # Unwrap Optional[X] -> X
            if origin is Union:
                args = get_args(expected_type)
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    expected_type = non_none[0]
                    origin = get_origin(expected_type)
            
            # Check if expected type is a Packable subclass
            if isinstance(expected_type, type) and issubclass(expected_type, Packable):
                return expected_type.decode(asset_bytes, array_type)
            
            # Otherwise assume it's an array
            return Packable._decode_packed_array(asset_bytes, array_type)
        
        # Handle nested BaseModel (non-ref dict that should be a model)
        if isinstance(value, dict):
            origin = get_origin(expected_type)
            
            # Unwrap Optional
            if origin is Union:
                args = get_args(expected_type)
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    expected_type = non_none[0]
                    origin = get_origin(expected_type)
            
            # Dict type - resolve values with value type
            if origin is dict:
                key_type, value_type = get_args(expected_type)
                return {
                    k: Packable._resolve_value_with_type(v, value_type, assets, array_type)
                    for k, v in value.items()
                }
            
            # BaseModel type - recursively resolve with schema
            if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                resolved = Packable._resolve_refs_with_schema(
                    expected_type, value, assets, array_type
                )
                return expected_type(**resolved)
            
            # Unknown dict - return as-is
            return value
        
        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            origin = get_origin(expected_type)
            
            # Unwrap Optional
            if origin is Union:
                args = get_args(expected_type)
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    expected_type = non_none[0]
                    origin = get_origin(expected_type)
            
            # Get element type
            if origin in (list, tuple):
                args = get_args(expected_type)
                elem_type = args[0] if args else Any
            else:
                elem_type = Any
            
            result = [
                Packable._resolve_value_with_type(v, elem_type, assets, array_type)
                for v in value
            ]
            return result if isinstance(value, list) else tuple(result)
        
        # Primitive - return as-is
        return value

    @staticmethod
    def _merge_field_data_with_schema(
        model_class: Type[BaseModel],
        data: Dict[str, Any],
        field_data: Dict[str, Any],
    ) -> None:
        """
        Merge metadata field_data into data, using Pydantic schema for type info.
        
        This handles the reconstruction of nested BaseModel instances without
        needing __model_class__/__model_module__ markers.
        """
        from typing import get_origin, get_args, Union
        
        for key, value in field_data.items():
            if key in ("__model_class__", "__model_module__"):
                # Skip legacy markers
                continue
            
            if key not in model_class.model_fields:
                # Unknown field - store as-is
                data[key] = value
                continue
            
            field_type = model_class.model_fields[key].annotation
            merged = Packable._merge_value_with_schema(value, field_type, data.get(key))
            data[key] = merged
    
    @staticmethod
    def _merge_value_with_schema(
        metadata_value: Any,
        expected_type: Any,
        existing_value: Any,
    ) -> Any:
        """Merge a metadata value with existing data using the schema type."""
        from typing import get_origin, get_args, Union
        
        if metadata_value is None:
            return existing_value
        
        # Unwrap Optional
        origin = get_origin(expected_type)
        if origin is Union:
            args = get_args(expected_type)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                expected_type = non_none[0]
                origin = get_origin(expected_type)
        
        # Handle dict type
        if origin is dict:
            key_type, value_type = get_args(expected_type)
            if isinstance(metadata_value, dict) and isinstance(existing_value, dict):
                # Merge dict entries
                result = dict(existing_value)
                for k, v in metadata_value.items():
                    if k in ("__model_class__", "__model_module__"):
                        continue
                    result[k] = Packable._merge_value_with_schema(
                        v, value_type, existing_value.get(k)
                    )
                return result
            elif isinstance(metadata_value, dict):
                # No existing value - reconstruct from metadata
                return {
                    k: Packable._merge_value_with_schema(v, value_type, None)
                    for k, v in metadata_value.items()
                    if k not in ("__model_class__", "__model_module__")
                }
            return metadata_value
        
        # Handle BaseModel type
        if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
            if isinstance(metadata_value, dict):
                # Filter out legacy markers
                filtered = {k: v for k, v in metadata_value.items()
                           if k not in ("__model_class__", "__model_module__")}
                
                if isinstance(existing_value, dict):
                    # Merge with existing dict data
                    merged = dict(existing_value)
                    Packable._merge_field_data_with_schema(expected_type, merged, filtered)
                    return expected_type(**merged)
                else:
                    # Reconstruct from metadata
                    data = {}
                    Packable._merge_field_data_with_schema(expected_type, data, filtered)
                    return expected_type(**data)
            return metadata_value
        
        # Handle list type  
        if origin in (list, tuple):
            if isinstance(metadata_value, (list, tuple)):
                args = get_args(expected_type)
                elem_type = args[0] if args else Any
                result = [
                    Packable._merge_value_with_schema(v, elem_type, None)
                    for v in metadata_value
                ]
                return result if origin is list else tuple(result)
            return metadata_value
        
        # Primitive - use metadata value
        return metadata_value

    def __reduce__(self):
        """
        Support for pickle serialization.

        Array types are preserved automatically via the per-array metadata.
        """
        return (
            self.__class__.decode,
            (self.encode(),),
        )

    @staticmethod
    def load_array(
        source: Union[PathLike, BytesIO],
        name: str,
        array_type: Optional[ArrayType] = None
    ) -> Array:
        """
        Load a single array from a zip file without loading the entire object.

        Useful for large files where you only need one array.

        Args:
            source: Path to the zip file or BytesIO buffer
            name: Array name (e.g., "normals" or "markerIndices.boundary")
            array_type: Array backend to use ("numpy" or "jax"). If None (default),
                       uses the array_type stored in the array's metadata.

        Returns:
            Decoded array (numpy or JAX)

        Raises:
            KeyError: If array not found in zip

        Example:
            normals = Mesh.load_array("mesh.zip", "normals")
        """
        if isinstance(source, BytesIO):
            source.seek(0)
            handler = DataHandler.create(BytesIO(source.read()))
        else:
            with open(source, "rb") as f:
                handler = DataHandler.create(BytesIO(f.read()))
        return ArrayUtils.load_array(handler, name, array_type)

    def convert_to(self: TPackable, array_type: ArrayType) -> TPackable:
        """
        Create a new Packable with all arrays converted to the specified type.

        Args:
            array_type: Target array backend ("numpy" or "jax")

        Returns:
            A new Packable with all arrays converted

        Raises:
            AssertionError: If JAX is requested but not available
        """
        data_copy = self.model_copy(deep=True)

        for field_name in data_copy.model_fields_set:
            try:
                value = getattr(data_copy, field_name)
                if value is not None:
                    converted = ArrayUtils.convert_recursive(value, array_type)
                    setattr(data_copy, field_name, converted)
            except AttributeError:
                pass

        return data_copy

