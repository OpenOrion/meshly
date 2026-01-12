"""Base packable class for encoded array storage.

This module provides a base class that handles automatic numpy/JAX array
detection and serialization to zip files. Classes like Mesh inherit from
this base to get automatic array encoding/decoding support.

Custom data classes can inherit from Packable to store simulation
results, time-series data, or any structured data with numpy arrays.
"""

import json
import zipfile
from io import BytesIO
from typing import (
    Dict,
    Optional,
    Set,
    Type,
    Any,
    TypeVar,
    Union,
    List,
)
import numpy as np
from pydantic import BaseModel, Field

from .array import ArrayUtils, ArrayMetadata, EncodedArray
from .common import PathLike
from .utils.zip_utils import ZipUtils
from .utils.packable_utils import PackableUtils

# Optional JAX support
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = None
    HAS_JAX = False

# Array type union - supports both numpy and JAX arrays
if HAS_JAX:
    Array = Union[np.ndarray, jnp.ndarray]
else:
    Array = np.ndarray

# Recursive type for decoded array data from zip files
# Values are arrays or nested dicts containing arrays
ArrayData = Dict[str, Union[Array, Dict[str, Any]]]


class EncodedData(BaseModel):
    """Container for encoded array data from a Packable."""
    arrays: Dict[str, EncodedArray] = Field(
        default_factory=dict, description="Encoded arrays")


class PackableMetadata(BaseModel):
    """Metadata for a Packable saved to zip."""
    class_name: str = Field(..., description="Name of the data class")
    module_name: str = Field(...,
                             description="Module containing the data class")
    field_data: Dict[str, Any] = Field(
        default_factory=dict, description="Non-array field values")


T = TypeVar("T", bound="Packable")
M = TypeVar("M", bound=PackableMetadata)


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
                result.update(PackableUtils.extract_nested_arrays(
                    value, field_name).keys())
        return result

    def encode(self) -> EncodedData:
        """
        Encode this container's arrays for serialization.

        Returns:
            EncodedData with all arrays encoded
        """
        encoded_arrays = {}

        for field_name in self.array_fields:
            # Handle nested array paths (e.g., "textures.diffuse")
            if "." in field_name:
                # Extract the nested array
                parts = field_name.split(".")
                obj = self
                for part in parts[:-1]:
                    if isinstance(obj, dict):
                        obj = obj[part]
                    else:
                        obj = getattr(obj, part)

                # Get the final array
                if isinstance(obj, dict):
                    array = obj[parts[-1]]
                else:
                    array = getattr(obj, parts[-1])

                if PackableUtils.is_array(array):
                    encoded_arrays[field_name] = ArrayUtils.encode_array(array)
            else:
                # Handle direct array fields
                try:
                    array = getattr(self, field_name)
                    if PackableUtils.is_array(array):
                        encoded_arrays[field_name] = ArrayUtils.encode_array(
                            array)
                except AttributeError:
                    pass

        return EncodedData(arrays=encoded_arrays)

    def _extract_non_array_fields(self) -> Dict[str, Any]:
        """Extract non-array field values for metadata, preserving BaseModel type info."""
        model_data = {}
        direct_arrays = {f for f in self.array_fields if "." not in f}
        for name in type(self).model_fields:
            if name in self.__private_attributes__ or name in direct_arrays:
                continue
            value = getattr(self, name, None)
            if value is not None and not PackableUtils.is_array(value):
                extracted = PackableUtils.extract_non_arrays(value)
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

    def _prepare_zip_files(
        self,
        encoded_data: EncodedData,
        field_data: Dict[str, Any],
        exclude_arrays: Optional[set] = None
    ) -> List[tuple]:
        """
        Prepare list of files to write to zip.

        Args:
            encoded_data: Encoded array data
            field_data: Non-array field data for metadata
            exclude_arrays: Set of array names to exclude (handled separately)

        Returns:
            List of (filename, data) tuples
        """
        exclude_arrays = exclude_arrays or set()
        files_to_write = []

        # Add array data
        for name in sorted(encoded_data.arrays.keys()):
            if name in exclude_arrays:
                continue
            encoded_array = encoded_data.arrays[name]
            array_path = name.replace(".", "/")
            files_to_write.append(
                (f"arrays/{array_path}/array.bin", encoded_array.data))

            array_metadata = ArrayMetadata(
                shape=list(encoded_array.shape),
                dtype=str(encoded_array.dtype),
                itemsize=encoded_array.itemsize,
            )
            files_to_write.append((
                f"arrays/{array_path}/metadata.json",
                json.dumps(array_metadata.model_dump(),
                           indent=2, sort_keys=True)
            ))

        # Create metadata using overridable method
        metadata = self._create_metadata(field_data)
        files_to_write.append(("metadata.json", json.dumps(
            metadata.model_dump(), indent=2, sort_keys=True)))

        return files_to_write

        return files_to_write

    def save_to_zip(
        self,
        destination: Union[PathLike, BytesIO],
        date_time: Optional[tuple] = None
    ) -> None:
        """
        Save this container to a zip file.

        Args:
            destination: Path to the output zip file or BytesIO object
            date_time: Optional date_time tuple for deterministic zip files
        """
        encoded_data = self.encode()
        field_data = self._extract_non_array_fields()
        files_to_write = self._prepare_zip_files(encoded_data, field_data)

        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            ZipUtils.write_files(zipf, files_to_write, date_time)

    @classmethod
    def load_metadata(
        cls,
        zipf: zipfile.ZipFile,
        metadata_cls: Type[M] = PackableMetadata
    ) -> M:
        """
        Load and validate metadata from an open zip file.

        Args:
            zipf: Open ZipFile object
            metadata_cls: The metadata class to use for parsing (default: PackableMetadata)

        Returns:
            Metadata object of the specified type

        Raises:
            ValueError: If class name doesn't match
        """
        with zipf.open("metadata.json") as f:
            metadata_dict = json.loads(f.read().decode("utf-8"))
            metadata = metadata_cls(**metadata_dict)

        if metadata.class_name != cls.__name__ or metadata.module_name != cls.__module__:
            raise ValueError(
                f"Class mismatch: expected {cls.__name__} but got {metadata.class_name} from {metadata.module_name}"
            )

        return metadata

    @classmethod
    def load_from_zip(cls: Type[T], source: Union[PathLike, BytesIO], use_jax: bool = False) -> T:
        """
        Load a Packable from a zip file.

        Args:
            source: Path to the input zip file or BytesIO object
            use_jax: If True and JAX is available, decode arrays as JAX arrays

        Returns:
            Loaded Packable instance
        """
        if use_jax and not HAS_JAX:
            raise ValueError(
                "JAX is not available. Install JAX to use JAX arrays.")

        with zipfile.ZipFile(source, "r") as zipf:
            metadata = cls.load_metadata(zipf)

            # Load and decode all arrays (handles both flat and nested)
            data = ZipUtils.load_arrays(zipf, use_jax)

            # Merge non-array fields from metadata
            if metadata.field_data:
                PackableUtils.merge_field_data(data, metadata.field_data)

            return cls(**data)

    @staticmethod
    def load_array(
        source: Union[PathLike, BytesIO],
        name: str,
        use_jax: bool = False
    ) -> Array:
        """
        Load a single array from a zip file without loading the entire object.

        Useful for large files where you only need one array.

        Args:
            source: Path to the zip file or BytesIO object
            name: Array name (e.g., "normals" or "markerIndices.boundary")
            use_jax: If True, decode as JAX array

        Returns:
            Decoded array (numpy or JAX)

        Raises:
            KeyError: If array not found in zip

        Example:
            normals = Mesh.load_array("mesh.zip", "normals")
        """
        with zipfile.ZipFile(source, "r") as zipf:
            return ZipUtils.load_array(zipf, name, use_jax)

    def to_numpy(self: T) -> T:
        """
        Create a new Packable with all arrays converted to NumPy arrays.

        Returns:
            A new Packable with all arrays as NumPy arrays
        """
        if not HAS_JAX:
            return self.model_copy(deep=True)

        data_copy = self.model_copy(deep=True)

        def convert_to_numpy(obj: Any) -> Any:
            if isinstance(obj, jnp.ndarray):
                return np.array(obj)
            elif isinstance(obj, np.ndarray):
                return obj
            elif isinstance(obj, dict):
                return {key: convert_to_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert_to_numpy(item) for item in obj)
            else:
                return obj

        for field_name in data_copy.model_fields_set:
            try:
                value = getattr(data_copy, field_name)
                if value is not None:
                    converted = convert_to_numpy(value)
                    setattr(data_copy, field_name, converted)
            except AttributeError:
                pass

        return data_copy

    def to_jax(self: T) -> T:
        """
        Create a new Packable with all arrays converted to JAX arrays.

        Returns:
            A new Packable with all arrays as JAX arrays

        Raises:
            ValueError: If JAX is not available
        """
        if not HAS_JAX:
            raise ValueError(
                "JAX is not available. Install JAX to convert to JAX arrays.")

        data_copy = self.model_copy(deep=True)

        def convert_to_jax(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return jnp.array(obj)
            elif HAS_JAX and isinstance(obj, jnp.ndarray):
                return obj
            elif isinstance(obj, dict):
                return {key: convert_to_jax(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert_to_jax(item) for item in obj)
            else:
                return obj

        for field_name in data_copy.model_fields_set:
            try:
                value = getattr(data_copy, field_name)
                if value is not None:
                    converted = convert_to_jax(value)
                    setattr(data_copy, field_name, converted)
            except AttributeError:
                pass

        return data_copy
