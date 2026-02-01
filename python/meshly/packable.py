"""Base packable class for encoded array storage.

This module provides a base class that handles automatic numpy/JAX array
detection and serialization to zip files. Classes like Mesh inherit from
this base to get automatic array encoding/decoding support.

Custom data classes can inherit from Packable to store simulation
results, time-series data, or any structured data with numpy arrays.

Packables can be nested. By default, nested Packables are "expanded" during
extraction (their fields become part of the parent's data dict with $refs for
arrays). Set `_self_contained = True` on a Packable class to make it extract
as a single zip blob reference instead.
"""

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, Union

from pydantic import BaseModel

from .array import Array, ArrayType, ArrayUtils
from .common import PathLike
from .constants import ExportConstants
from .data_handler import AssetProvider, CachedAssetLoader, DataHandler
from .utils.checksum_utils import ChecksumUtils
from .utils.schema_utils import SchemaUtils
from .utils.serialization_utils import SerializationUtils

TModel = TypeVar("TModel", bound=BaseModel)
TPackable = TypeVar("TPackable", bound="Packable")

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


class Packable(BaseModel):
    """Base class for data containers with automatic array serialization.

    Subclasses can define numpy array attributes which will be automatically
    detected, encoded, and saved to zip files. Non-array fields are preserved
    in metadata.

    Nested Packables are supported. By default, nested Packables are "expanded"
    during extraction (fields inlined with $refs for arrays). Set
    `_self_contained = True` to make the class extract as a single zip blob.

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

        # Self-contained Packable (extracts as single $ref)
        class Mesh(Packable):
            _self_contained: ClassVar[bool] = True
            vertices: np.ndarray
            faces: np.ndarray
    """

    _self_contained: ClassVar[bool] = False
    """If True, this Packable extracts as a single zip blob reference.
    If False (default), fields are expanded with $refs for individual arrays."""

    class Config:
        arbitrary_types_allowed = True

    @property
    def array_fields(self) -> set[str]:
        """Get all array field paths, including nested arrays in dicts/BaseModels."""
        result = set()
        for field_name in type(self).model_fields:
            if hasattr(self, "__private_attributes__") and field_name in self.__private_attributes__:
                continue
            value = getattr(self, field_name, None)
            if value is not None:
                result.update(
                    ArrayUtils.extract_nested_arrays(
                        value, field_name, skip=lambda x: isinstance(x, Packable)
                    ).keys()
                )
        return result

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

    def encode(self) -> bytes:
        """Serialize this Packable to bytes (zip format).

        Uses extract() internally to serialize all fields and assets.
        """
        extracted = Packable.extract(self)

        destination = BytesIO()
        handler = DataHandler.create(destination)

        # Write JSON Schema with encoding info (requires using Array/VertexBuffer/IndexSequence types)
        try:
            schema = type(self).model_json_schema()
            handler.write_text(
                ExportConstants.SCHEMA_FILE,
                json.dumps(schema, indent=2, sort_keys=True),
            )
        except Exception:
            pass  # Schema generation failed (raw np.ndarray used instead of Array types)

        # Write data as JSON
        handler.write_text(
            ExportConstants.DATA_FILE,
            json.dumps(extracted.data, indent=2, sort_keys=True),
        )

        # Write assets by checksum
        for checksum in sorted(extracted.assets.keys()):
            handler.write_binary(ExportConstants.asset_path(checksum), extracted.assets[checksum])

        handler.finalize()
        return destination.getvalue()

    @classmethod
    def decode(
        cls: type[TPackable],
        buf: bytes,
        array_type: ArrayType | None = None,
    ) -> TPackable:
        """Deserialize a Packable from bytes.

        Uses reconstruct() internally to restore all fields and assets.
        """
        if cls is Packable:
            raise TypeError(
                "Cannot decode on base Packable class. "
                "Use the specific subclass: MyClass.decode(...)"
            )

        handler = DataHandler.create(BytesIO(buf))

        # Read data
        data = json.loads(handler.read_text(ExportConstants.DATA_FILE))

        # Build assets dict from files
        assets: dict[str, bytes] = {}
        try:
            for file_path in handler.list_files(ExportConstants.ASSETS_DIR):
                file_str = str(file_path)
                if file_str.endswith(ExportConstants.ASSET_EXT):
                    checksum = ExportConstants.checksum_from_path(file_str)
                    assets[checksum] = handler.read_binary(file_str)
        except (KeyError, FileNotFoundError):
            pass  # No assets folder

        return Packable.reconstruct(cls, data, assets, array_type)

    @staticmethod
    def decode_from_schema(
        buf: bytes,
        array_type: ArrayType | None = None,
        as_model: bool = False,
    ) -> dict[str, Any] | BaseModel:
        """Deserialize a Packable using only the stored schema.json.
        
        This allows decoding without having the original class definition.
        The schema.json must be present in the zip file.
        
        Args:
            buf: Encoded bytes (zip format)
            array_type: Target array type ("numpy" or "jax")
            as_model: If True, returns a dynamically created Pydantic model instance.
                     If False (default), returns a plain dict.
            
        Returns:
            Dict with all fields resolved, or a Pydantic model instance if as_model=True
            
        Raises:
            ValueError: If schema.json is not present in the zip
            
        Example:
            # Decode to dict (default)
            data = Packable.decode_from_schema(raw_bytes)
            vertices = data["vertices"]  # numpy array
            
            # Decode to dynamic Pydantic model
            obj = Packable.decode_from_schema(raw_bytes, as_model=True)
            vertices = obj.vertices  # numpy array with attribute access
        """
        from .utils.json_schema import JsonSchema
        
        handler = DataHandler.create(BytesIO(buf))

        # Read and validate schema
        try:
            schema_dict = json.loads(handler.read_text(ExportConstants.SCHEMA_FILE))
            schema = JsonSchema.model_validate(schema_dict)
        except (KeyError, FileNotFoundError):
            raise ValueError(
                "schema.json not found in packable. "
                "This packable was created without proper Array/VertexBuffer/IndexSequence types. "
                "Use decode() with the class definition instead."
            )

        # Read data
        data = json.loads(handler.read_text(ExportConstants.DATA_FILE))

        # Build assets dict from files
        assets: dict[str, bytes] = {}
        try:
            for file_path in handler.list_files(ExportConstants.ASSETS_DIR):
                file_str = str(file_path)
                if file_str.endswith(ExportConstants.ASSET_EXT):
                    checksum = ExportConstants.checksum_from_path(file_str)
                    assets[checksum] = handler.read_binary(file_str)
        except (KeyError, FileNotFoundError):
            pass  # No assets folder

        if as_model:
            from .utils.dynamic_model import DynamicModelBuilder
            return DynamicModelBuilder.instantiate(schema, data, assets, array_type)
        
        return SchemaUtils.resolve_refs_with_json_schema(schema, data, assets, array_type)

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
                # Use extract_field_value to respect type annotations (VertexBuffer, IndexSequence)
                data[field_name] = SerializationUtils.extract_field_value(
                    obj, field_name, value, assets
                )

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

    @staticmethod
    def reconstruct(
        model_class: type[TModel],
        data: dict[str, Any] | SerializedPackableData,
        assets: AssetProvider | None = None,
        array_type: ArrayType | None = None,
        lazy: bool | None = None,
    ) -> TModel | LazyModel[TModel]:
        """Reconstruct a Pydantic BaseModel from extracted data and assets.

        Args:
            model_class: The Pydantic model class to reconstruct.
            data: Either a dict or SerializedPackableData containing the extracted data.
            assets: Asset provider (dict, callable, or CachedAssetLoader). If data is
                    SerializedPackableData and assets is None, uses data.assets.
            array_type: Optional array type for conversion.
            lazy: If True, returns a LazyModel that defers asset loading until field access.
                    If None (default), lazy loading is used when assets is a callable.

        Returns:
            Reconstructed model instance or LazyModel proxy depending on lazy parameter.
        
        Example:
            extracted = Packable.extract(my_model)
            restored = Packable.reconstruct(MyModel, extracted.data, extracted.assets)
            
            # Lazy loading with callable
            lazy = Packable.reconstruct(MyModel, data, fetch_asset_fn)
        """
        # Handle SerializedPackableData input
        if isinstance(data, SerializedPackableData):
            if assets is None:
                assets = data.assets
            data = data.data

        if assets is None:
            raise TypeError("assets is required when data is a dict")

        # Auto-detect lazy mode: callable or CachedAssetLoader → lazy by default
        use_lazy = lazy if lazy is not None else (
            (callable(assets) and not isinstance(assets, dict)) or 
            isinstance(assets, CachedAssetLoader)
        )

        if use_lazy:
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
        """Load a single array from a zip file by field name.

        Note: This requires the full object to be loaded to resolve the field name
        to its checksum. For direct array access, use decode() instead.
        """
        if isinstance(source, BytesIO):
            source.seek(0)
            handler = DataHandler.create(BytesIO(source.read()))
        else:
            handler = DataHandler.create(BytesIO(Path(source).read_bytes()))

        # Load data.json to find the checksum for this field
        data = json.loads(handler.read_text(ExportConstants.DATA_FILE))

        # Navigate to the field (supports dotted paths like "mesh.vertices")
        parts = name.split(".")
        current = data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Field '{name}' not found in packable")
            current = current[part]

        if not isinstance(current, dict) or "$ref" not in current:
            raise ValueError(f"Field '{name}' is not an array reference")

        checksum = current["$ref"]
        asset_bytes = handler.read_binary(ExportConstants.asset_path(checksum))
        
        # Build metadata from $ref object (all keys except $ref)
        from .array import ArrayRefMetadata
        metadata_dict = {k: v for k, v in current.items() if k != "$ref"}
        metadata = ArrayRefMetadata(**metadata_dict)
        
        # Default encoding is "array" for load_array (we don't have schema context here)
        return ArrayUtils.decode_with_metadata(asset_bytes, "array", metadata, array_type)

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
