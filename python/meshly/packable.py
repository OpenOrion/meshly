"""Base packable class for encoded array storage.

This module provides a base class that handles automatic numpy/JAX array
detection and serialization to zip files. Classes like Mesh inherit from
this base to get automatic array encoding/decoding support.

Custom data classes can inherit from Packable to store simulation
results, time-series data, or any structured data with numpy arrays.

Packables cannot contain nested Packables. For composite structures,
use the extract() and reconstruct() methods to handle asset management.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Generic, TypeVar, Union, cast

from pydantic import BaseModel, Field

from .array import Array, ArrayType, ArrayUtils
from .common import PathLike
from .data_handler import AssetProvider, CachedAssetLoader, DataHandler
from .utils.checksum_utils import ChecksumUtils
from .utils.schema_utils import SchemaUtils
from .utils.serialization_utils import SerializationUtils

TModel = TypeVar("TModel", bound=BaseModel)


class PackableMetadata(BaseModel):
    """Metadata for a Packable saved to zip."""

    field_data: dict[str, Any] = Field(default_factory=dict, description="Non-array field values")


TPackableMetadata = TypeVar("TPackableMetadata", bound=PackableMetadata)
TPackable = TypeVar("TPackable", bound="Packable")
FieldValue = TypeVar("FieldValue")


@dataclass
class SerializedPackableData:
    """Result of extracting a Packable for serialization.

    Contains the serializable data dict with checksum references,
    plus the encoded assets (arrays as bytes).
    
    The data dict may contain a reserved key '$module' with the fully qualified
    class name for automatic class resolution during reconstruction.
    """

    data: dict[str, Any]
    """Serializable dict with primitive fields and checksum refs for arrays"""
    assets: dict[str, bytes]
    """Map of checksum -> encoded bytes for all arrays"""

    @property
    def checksums(self) -> list[str]:
        """Get list of all asset checksums."""
        return list(self.assets.keys())

    @staticmethod
    def extract_checksums(data: dict[str, Any]) -> list[str]:
        """Extract all $ref checksums from a serialized data dict.

        Recursively walks the data structure to find all {"$ref": checksum} entries.

        Args:
            data: Serialized data dict with $ref references

        Returns:
            List of all unique checksums found
        """
        checksums: set[str] = set()

        def _extract(obj: Any) -> None:
            if isinstance(obj, dict):
                if "$ref" in obj:
                    checksums.add(obj["$ref"])
                else:
                    for v in obj.values():
                        _extract(v)
            elif isinstance(obj, list):
                for item in obj:
                    _extract(item)

        _extract(data)
        return list(checksums)


@dataclass
class ExtractedAssets:
    """Result of extracting assets from values.

    Contains the binary assets and their file extensions (for ResourceRefs).
    """

    assets: dict[str, bytes]
    """Map of checksum -> encoded bytes"""
    extensions: dict[str, str]
    """Map of checksum -> file extension (e.g., '.stl')"""


class LazyModel(Generic[TModel]):
    """Lazy proxy for a Pydantic BaseModel that defers asset loading until field access.

    Example:
        def fetch_asset(checksum: str) -> bytes:
            return cloud_storage.download(checksum)

        lazy = Packable.reconstruct(SimulationCase, data, fetch_asset)
        # No assets loaded yet

        temp = lazy.temperature  # NOW the temperature asset is fetched
        vel = lazy.velocity      # NOW the velocity asset is fetched
    """

    __slots__ = ("_model_class", "_data", "_assets", "_array_type", "_cache", "_resolved")

    def __init__(
        self,
        model_class: type[TModel],
        data: dict[str, Any],
        assets: AssetProvider,
        array_type: ArrayType | None = None,
    ):
        object.__setattr__(self, "_model_class", model_class)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_assets", assets)
        object.__setattr__(self, "_array_type", array_type)
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_resolved", None)

    def _get_cached_asset(self, checksum: str) -> bytes:
        """Get asset bytes, using cache if CachedAssetLoader is provided."""
        return SerializationUtils.get_cached_asset(
            object.__getattribute__(self, "_assets"), checksum
        )

    def __getattr__(self, name: str) -> Any:
        cache = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]

        model_class = object.__getattribute__(self, "_model_class")
        data = object.__getattribute__(self, "_data")
        array_type = object.__getattribute__(self, "_array_type")

        if name not in model_class.model_fields:
            raise AttributeError(f"'{model_class.__name__}' has no attribute '{name}'")

        if name not in data:
            return None

        field_value = data[name]
        field_type = model_class.model_fields[name].annotation

        resolved = SchemaUtils.resolve_value_with_type(
            field_value, field_type, self._get_cached_asset, array_type
        )

        cache[name] = resolved
        return resolved

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LazyModel is read-only. Use resolve() to get a mutable model.")

    def resolve(self) -> TModel:
        """Fully resolve all fields and return the actual Pydantic model."""
        resolved = object.__getattribute__(self, "_resolved")
        if resolved is not None:
            return resolved

        model_class = object.__getattribute__(self, "_model_class")
        data = object.__getattribute__(self, "_data")
        array_type = object.__getattribute__(self, "_array_type")
        cache = object.__getattribute__(self, "_cache")

        resolved_data = {}
        for field_name, field_info in model_class.model_fields.items():
            if field_name in cache:
                resolved_data[field_name] = cache[field_name]
            elif field_name in data:
                resolved_data[field_name] = SchemaUtils.resolve_value_with_type(
                    data[field_name], field_info.annotation, self._get_cached_asset, array_type
                )

        result = model_class(**resolved_data)
        object.__setattr__(self, "_resolved", result)
        return result

    def __repr__(self) -> str:
        model_class = object.__getattribute__(self, "_model_class")
        cache = object.__getattribute__(self, "_cache")
        data = object.__getattribute__(self, "_data")
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
    decode: Callable[[bytes, TPackableMetadata, ArrayType | None], FieldValue]
    """Decoder function: (bytes, metadata, array_type) -> value"""
    optional: bool = False
    """Whether the field is optional (won't throw if missing)"""


class Packable(BaseModel):
    """Base class for data containers with automatic array serialization.

    Subclasses can define numpy array attributes which will be automatically
    detected, encoded, and saved to zip files. Non-array fields are preserved
    in metadata.

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
        loaded = SimulationResult.load_from_zip("result.zip")
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_no_direct_packable_fields()

    def _validate_no_direct_packable_fields(self) -> None:
        """Validate that this Packable has no direct Packable fields."""
        for field_name in type(self).model_fields:
            if field_name in self.__private_attributes__:
                continue
            value = getattr(self, field_name, None)
            if value is not None and isinstance(value, Packable):
                raise TypeError(
                    f"Direct Packable fields are not allowed. Field '{field_name}' "
                    f"contains a {type(value).__name__}. Packables can be nested "
                    "inside dicts or other BaseModels, and extract() will handle them."
                )

    @property
    def array_fields(self) -> set[str]:
        """Get all array field paths, including nested arrays in dicts/BaseModels."""
        result = set()
        for field_name in type(self).model_fields:
            if field_name in self.__private_attributes__:
                continue
            value = getattr(self, field_name, None)
            if value is not None:
                result.update(
                    ArrayUtils.extract_nested_arrays(
                        value, field_name, skip=lambda x: isinstance(x, Packable)
                    ).keys()
                )
        return result

    def _extract_non_array_fields(self) -> dict[str, Any]:
        """Extract non-array field values for metadata."""
        model_data = {}
        direct_arrays = {f for f in self.array_fields if "." not in f}
        for name in type(self).model_fields:
            if name in self.__private_attributes__ or name in direct_arrays:
                continue
            value = getattr(self, name, None)
            if value is not None and not ArrayUtils.is_array(value):
                extracted = ArrayUtils.extract_non_arrays(
                    value, skip=lambda x: isinstance(x, Packable)
                )
                if extracted is not None:
                    model_data[name] = extracted
        return model_data

    def _create_metadata(self, field_data: dict[str, Any]) -> PackableMetadata:
        """Create metadata for this Packable. Subclasses can override."""
        return PackableMetadata(field_data=field_data)

    @classmethod
    def load_metadata(
        cls, handler: DataHandler, metadata_cls: type[TPackableMetadata] = PackableMetadata
    ) -> TPackableMetadata:
        """Load and validate metadata using a read handler."""
        metadata_text = handler.read_text("metadata.json")
        metadata_dict = json.loads(metadata_text)
        return metadata_cls(**metadata_dict)

    def save_to_zip(self, destination: Union[PathLike, BytesIO]) -> None:
        """Save this container to a zip file."""
        encoded = self.encode()
        if isinstance(destination, BytesIO):
            destination.write(encoded)
        else:
            Path(destination).write_bytes(encoded)

    @classmethod
    def load_from_zip(
        cls: type[TPackable],
        source: Union[PathLike, BytesIO],
        array_type: ArrayType | None = None,
    ) -> TPackable:
        """Load a Packable from a zip file."""
        if isinstance(source, BytesIO):
            source.seek(0)
            return cls.decode(source.read(), array_type)
        else:
            with open(source, "rb") as f:
                return cls.decode(f.read(), array_type)

    @classmethod
    def _get_custom_fields(cls) -> dict[str, CustomFieldConfig]:
        """Get custom field configurations. Subclasses override this."""
        return {}

    @classmethod
    def _get_custom_field_names(cls) -> set[str]:
        """Get set of field names that have custom encoding/decoding."""
        return set(cls._get_custom_fields().keys())

    @classmethod
    def _decode_custom_fields(
        cls,
        handler: DataHandler,
        metadata: PackableMetadata,
        data: dict[str, Any],
        array_type: ArrayType | None = None,
    ) -> None:
        """Decode fields with custom decoders."""
        for field_name, config in cls._get_custom_fields().items():
            try:
                encoded_bytes = handler.read_binary(f"{config.file_name}.bin")
                data[field_name] = config.decode(encoded_bytes, metadata, array_type)
            except (KeyError, FileNotFoundError):
                if not config.optional:
                    raise ValueError(
                        f"Required custom field '{field_name}' ({config.file_name}.bin) not found"
                    )

    @classmethod
    def _load_standard_arrays(
        cls,
        handler: DataHandler,
        data: dict[str, Any],
        skip_fields: set[str],
        array_type: ArrayType | None = None,
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

            array_path = file_str[7:-10]  # Remove "arrays/" and "/array.bin"
            name = array_path.replace("/", ".")

            base_field = name.split(".")[0]
            if base_field in skip_fields:
                continue

            decoded = ArrayUtils.load_array(handler, name, array_type)

            if "." in name:
                parts = name.split(".")
                current = data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = decoded
            else:
                data[name] = decoded

    def _encode_standard_arrays(self, skip_fields: set[str]) -> dict[str, bytes]:
        """Encode standard arrays, skipping custom fields."""
        encoded_arrays = {}

        for field_name in self.array_fields:
            if field_name in skip_fields:
                continue

            if "." in field_name:
                parts = field_name.split(".")
                obj = self
                for part in parts[:-1]:
                    obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
                array = obj[parts[-1]] if isinstance(obj, dict) else getattr(obj, parts[-1])
                if ArrayUtils.is_array(array):
                    encoded_arrays[field_name] = ArrayUtils.encode_array(array)
            else:
                try:
                    array = getattr(self, field_name)
                    if ArrayUtils.is_array(array):
                        encoded_arrays[field_name] = ArrayUtils.encode_array(array)
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
        """Serialize this Packable to bytes (zip format)."""
        custom_field_names = self._get_custom_field_names()
        encoded_arrays = self._encode_standard_arrays(custom_field_names)
        field_data = self._extract_non_array_fields()
        metadata = self._create_metadata(field_data)

        destination = BytesIO()
        handler = DataHandler.create(destination)

        for name in sorted(encoded_arrays.keys()):
            ArrayUtils.save_array(handler, name, encoded_arrays[name])

        self._encode_custom_fields(handler)

        handler.write_text(
            "metadata.json",
            json.dumps(metadata.model_dump(), indent=2, sort_keys=True),
        )

        handler.finalize()
        return destination.getvalue()

    @classmethod
    def decode(
        cls: type[TPackable],
        buf: bytes,
        array_type: ArrayType | None = None,
    ) -> TPackable:
        """Deserialize a Packable from bytes."""
        if cls is Packable:
            raise TypeError(
                "Cannot decode on base Packable class. "
                "Use the specific subclass: MyClass.decode(...)"
            )

        handler = DataHandler.create(BytesIO(buf))
        metadata = cls.load_metadata(handler)
        skip_fields = cls._get_custom_field_names()

        data: dict[str, Any] = {}
        cls._decode_custom_fields(handler, metadata, data, array_type)
        cls._load_standard_arrays(handler, data, skip_fields, array_type)

        if metadata.field_data:
            SchemaUtils.merge_field_data_with_schema(cls, data, metadata.field_data)

        return cls(**data)

    @staticmethod
    def extract(obj: BaseModel) -> SerializedPackableData:
        """Extract arrays and Packables from a BaseModel into serializable data and assets.

        The module name is always included for potential class resolution during reconstruction,
        but it's only used if explicitly opted-in via use_stored_class=True for security.

        Args:
            obj: BaseModel instance to extract

        Returns:
            SerializedPackableData with data dict (refs for arrays), assets dict, and module name
        """
        if not isinstance(obj, BaseModel):
            raise TypeError(f"extract() requires a Pydantic BaseModel, got {type(obj).__name__}.")

        assets: dict[str, bytes] = {}
        data: dict[str, Any] = {}

        for field_name in type(obj).model_fields:
            if hasattr(obj, "__private_attributes__") and field_name in obj.__private_attributes__:
                continue
            value = getattr(obj, field_name, None)
            if value is not None:
                data[field_name] = SerializationUtils.extract_value(value, assets)

        # Include computed fields (Pydantic v2)
        for field_name in type(obj).model_computed_fields:
            value = getattr(obj, field_name, None)
            if value is not None:
                data[field_name] = SerializationUtils.extract_value(value, assets)

        # Always include module name as reserved key for potential reconstruction
        obj_class = type(obj)
        data["$module"] = f"{obj_class.__module__}.{obj_class.__qualname__}"

        return SerializedPackableData(data=data, assets=assets)

    @staticmethod
    def compute_checksum(
        obj: Union[bytes, "SerializedPackableData", "Packable", BaseModel],
    ) -> str:
        """Compute SHA256 checksum for various types of data.

        Returns:
            16-character hex string (first 64 bits of SHA256)
        """
        if isinstance(obj, bytes):
            return ChecksumUtils.compute_bytes_checksum(obj)

        if isinstance(obj, SerializedPackableData):
            return ChecksumUtils.compute_dict_checksum(obj.data, obj.assets)

        if isinstance(obj, Packable):
            return ChecksumUtils.compute_bytes_checksum(obj.encode())

        if isinstance(obj, BaseModel):
            extracted = Packable.extract(obj)
            return ChecksumUtils.compute_dict_checksum(extracted.data, extracted.assets)

        raise TypeError(
            f"compute_checksum() requires bytes, SerializedPackableData, Packable, or BaseModel, "
            f"got {type(obj).__name__}"
        )

    @classmethod
    def reconstruct(
        cls,
        data: dict[str, Any] | SerializedPackableData,
        assets: AssetProvider | None = None,
        array_type: ArrayType | None = None,
        allowed_classes: list[type[BaseModel]] | None = None,
        lazy: bool = False,
    ):
        """Reconstruct a Pydantic BaseModel from extracted data and assets.

        Args:
            data: Either a dict or SerializedPackableData containing the extracted data.
            assets: Asset provider (dict, callable, or CachedAssetLoader). If data is
                    SerializedPackableData and assets is None, uses data.assets.
            array_type: Optional array type for conversion.
            allowed_classes: List of allowed BaseModel classes. If provided and '$module' key
                            is present in data, will use the matching class if found.
            lazy: If True, returns a LazyModel that defers asset loading until field access.
                    Default is False for eager loading.

        Returns:
            Reconstructed model instance or LazyModel proxy depending on lazy parameter.
        """
        # Handle SerializedPackableData input
        if isinstance(data, SerializedPackableData):
            if assets is None:
                assets = data.assets
            data = data.data
        
        # Start with cls as the model class
        model_class = cls if cls is not Packable else None
        
        # Try to resolve from $module if cls is base Packable or doesn't match stored module
        if "$module" in data:
            module_path = data["$module"]
            cls_module_path = f"{cls.__module__}.{cls.__qualname__}"
            
            # If cls matches the stored module, use cls
            if cls is not Packable and cls_module_path == module_path:
                model_class = cls
            # Otherwise, try allowed_classes fallback
            elif allowed_classes:
                for allowed_cls in allowed_classes:
                    allowed_module_path = f"{allowed_cls.__module__}.{allowed_cls.__qualname__}"
                    if allowed_module_path == module_path:
                        model_class = allowed_cls
                        break

        if model_class is None:
            raise TypeError(
                "Cannot determine model class. Either call on a specific subclass "
                "(e.g., MyModel.reconstruct(...)) or pass allowed_classes with the "
                "class types you trust and ensure '$module' key is present in the data."
            )

        if assets is None:
            raise TypeError("assets is required when data is a dict")

        if lazy:
            return LazyModel(model_class, data, assets, array_type)

        resolved_data = SchemaUtils.resolve_refs_with_schema(model_class, data, assets, array_type)
        return model_class(**resolved_data)

    @staticmethod
    def extract_assets(*values: Any) -> ExtractedAssets:
        """Extract all assets from one or more values.

        Recursively finds all ResourceRefs, Packables, and arrays in the given values
        and returns their assets with extensions.

        This is useful for extracting assets from function arguments before remote execution,
        ensuring all referenced files are available on the remote side.

        Args:
            *values: Any values to extract assets from (BaseModels, dicts, lists, tuples, etc.)

        Returns:
            ExtractedAssets with assets dict and extensions dict

        Example:
            # Extract assets from a Pipeline and a ResourceRef
            extracted = Packable.extract_assets(pipeline, resource_ref)

            # Extract assets from function args
            extracted = Packable.extract_assets(*fn.args, *fn.kwargs.values())
        """
        assets: dict[str, bytes] = {}
        extensions: dict[str, str] = {}
        for value in values:
            SerializationUtils.extract_value(value, assets, extensions)
        return ExtractedAssets(assets=assets, extensions=extensions)

    def __reduce__(self):
        """Support for pickle serialization."""
        return (self.__class__.decode, (self.encode(),))

    @staticmethod
    def load_array(
        source: Union[PathLike, BytesIO], name: str, array_type: ArrayType | None = None
    ) -> Array:
        """Load a single array from a zip file without loading the entire object."""
        if isinstance(source, BytesIO):
            source.seek(0)
            handler = DataHandler.create(BytesIO(source.read()))
        else:
            handler = DataHandler.create(BytesIO(Path(source).read_bytes()))
        return ArrayUtils.load_array(handler, name, array_type)

    def convert_to(self: TPackable, array_type: ArrayType) -> TPackable:
        """Create a new Packable with all arrays converted to the specified type."""
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
