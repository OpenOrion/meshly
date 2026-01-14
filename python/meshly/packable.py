"""Base packable class for encoded array storage.

This module provides a base class that handles automatic numpy/JAX array
detection and serialization to zip files. Classes like Mesh inherit from
this base to get automatic array encoding/decoding support.

Custom data classes can inherit from Packable to store simulation
results, time-series data, or any structured data with numpy arrays.
"""

import json
from dataclasses import dataclass
from io import BytesIO
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
from pydantic import BaseModel, Field
from .array import ArrayUtils, ArrayType, Array
from .common import PathLike
from .data_handler import WriteHandler, ReadHandler, ZipBuffer


class PackableMetadata(BaseModel):
    """Metadata for a Packable saved to zip."""
    class_name: str = Field(..., description="Name of the data class")
    module_name: str = Field(...,
                             description="Module containing the data class")
    field_data: Dict[str, Any] = Field(
        default_factory=dict, description="Non-array field values")


TPackableMetadata = TypeVar("TPackableMetadata", bound=PackableMetadata)
TPackable = TypeVar("TPackable", bound="Packable")
FieldValue = TypeVar("FieldValue")  # Value type for custom fields


@dataclass
class CustomFieldConfig(Generic[FieldValue, TPackableMetadata]):
    """Configuration for custom field encoding/decoding."""
    file_name: str
    """File name in zip (without .bin extension)"""
    encode: Callable[[FieldValue, Any], bytes]
    """Encoder function: (value, instance) -> bytes"""
    decode: Callable[[bytes, TPackableMetadata, Optional[ArrayType]], FieldValue]
    """Decoder function: (bytes, metadata, array_type) -> value"""
    optional: bool = False
    """Whether the field is optional (won't throw if missing)"""


class Packable(BaseModel):
    """
    Base class for data containers with automatic array serialization.

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
            class_name=self.__class__.__name__,
            module_name=self.__class__.__module__,
            field_data=field_data,
        )


    @classmethod
    def load_metadata(
        cls,
        handler: ReadHandler,
        metadata_cls: Type[TPackableMetadata] = PackableMetadata
    ) -> TPackableMetadata:
        """
        Load and validate metadata using a read handler.

        Args:
            handler: ReadHandler for reading files
            metadata_cls: The metadata class to use for parsing (default: PackableMetadata)

        Returns:
            Metadata object of the specified type

        Raises:
            ValueError: If class name doesn't match
        """
        metadata_text = handler.read_text("metadata.json")
        metadata_dict = json.loads(metadata_text)
        metadata = metadata_cls(**metadata_dict)

        if metadata.class_name != cls.__name__ or metadata.module_name != cls.__module__:
            raise ValueError(
                f"Class mismatch: expected {cls.__name__} but got {metadata.class_name} from {metadata.module_name}"
            )

        return metadata

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
            with open(destination, "wb") as f:
                f.write(encoded)

    @classmethod
    def load_from_zip(
        cls: Type[TPackable],
        source: Union[PathLike, BytesIO],
        array_type: Optional[ArrayType] = None
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
        handler: ReadHandler,
        metadata: PackableMetadata,
        data: Dict[str, Any],
        array_type: Optional[ArrayType] = None
    ) -> None:
        """Decode fields with custom decoders."""
        for field_name, config in cls._get_custom_fields().items():
            try:
                encoded_bytes = handler.read_binary(f"{config.file_name}.bin")
                data[field_name] = config.decode(encoded_bytes, metadata, array_type)
            except (KeyError, FileNotFoundError):
                if not config.optional:
                    raise ValueError(f"Required custom field '{field_name}' ({config.file_name}.bin) not found in zip")

    @classmethod
    def _load_standard_arrays(
        cls,
        handler: ReadHandler,
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
                        encoded_arrays[field_name] = ArrayUtils.encode_array(array)
                except AttributeError:
                    pass

        return encoded_arrays

    def _encode_custom_fields(self, handler: WriteHandler) -> None:
        """Encode fields with custom encoders."""
        for field_name, config in self._get_custom_fields().items():
            value = getattr(self, field_name)
            if value is not None:
                encoded_bytes = config.encode(value, self)
                handler.write_binary(f"{config.file_name}.bin", encoded_bytes)

    def encode(self) -> bytes:
        """
        Serialize this Packable to bytes.

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
        destination = ZipBuffer()
        handler = WriteHandler.create_handler(destination)

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
    def decode(cls: Type[TPackable], buf: bytes, array_type: Optional[ArrayType] = None) -> TPackable:
        """
        Deserialize a Packable from bytes.

        Args:
            buf: Bytes containing the zip-encoded data
            array_type: Array backend to use. If None (default), uses the
                       array_type stored in each array's metadata.

        Returns:
            Loaded Packable instance
        """
        handler = ReadHandler.create_handler(ZipBuffer(buf))
        metadata = cls.load_metadata(handler)
        custom_field_names = cls._get_custom_field_names()

        data: Dict[str, Any] = {}

        # Decode custom fields first
        cls._decode_custom_fields(handler, metadata, data, array_type)

        # Load standard arrays
        cls._load_standard_arrays(handler, data, custom_field_names, array_type)

        # Merge non-array fields from metadata
        if metadata.field_data:
            Packable._merge_field_data(data, metadata.field_data)

        return cls(**data)

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
            handler = ReadHandler.create_handler(ZipBuffer(source.read()))
        else:
            with open(source, "rb") as f:
                handler = ReadHandler.create_handler(ZipBuffer(f.read()))
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

    @staticmethod
    def _reconstruct_model(data: Dict[str, Any]) -> Any:
        """Reconstruct BaseModel from serialized dict with __model_class__/__model_module__."""
        if not isinstance(data, dict):
            return data

        # Recursively process nested dicts first
        processed = {k: Packable._reconstruct_model(v) if isinstance(v, dict) else v
                     for k, v in data.items() if k not in ("__model_class__", "__model_module__")}

        if "__model_class__" not in data:
            return processed

        try:
            import importlib
            module = importlib.import_module(data["__model_module__"])
            model_class = getattr(module, data["__model_class__"])
            return model_class(**processed)
        except (ImportError, AttributeError):
            return processed

    @staticmethod
    def _merge_field_data(data: Dict[str, Any], field_data: Dict[str, Any]) -> None:
        """Merge metadata fields into data, reconstructing BaseModel instances."""
        for key, value in field_data.items():
            existing = data.get(key)
            if not isinstance(value, dict):
                data[key] = value
            elif "__model_class__" in value:
                # Single BaseModel: merge arrays then reconstruct
                merged = {**value, **
                          (existing if isinstance(existing, dict) else {})}
                data[key] = Packable._reconstruct_model(merged)
            elif isinstance(existing, dict):
                # Check if dict of BaseModels
                for subkey, subval in value.items():
                    if isinstance(subval, dict) and "__model_class__" in subval:
                        merged = {**subval, **existing.get(subkey, {})}
                        existing[subkey] = Packable._reconstruct_model(merged)
                    elif isinstance(subval, dict) and isinstance(existing.get(subkey), dict):
                        Packable._merge_field_data(existing[subkey], subval)
                    else:
                        existing[subkey] = subval
            else:
                data[key] = Packable._reconstruct_model(value)
