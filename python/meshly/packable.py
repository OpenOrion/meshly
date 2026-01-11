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
    Generic,
    Optional,
    Set,
    Type,
    Any,
    TypeVar,
    Union,
    List,
    get_type_hints,
)
import numpy as np
from pydantic import BaseModel, Field

from .array import ArrayUtils, ArrayMetadata, EncodedArray
from .common import PathLike
from .utils.zip_utils import ZipUtils

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

    def copy(self: T) -> T:
        """
        Create a deep copy of this packable.

        Returns:
            A new Packable instance with copied data
        """
        copied_fields = {}

        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)

            if value is None:
                copied_fields[field_name] = None
            elif isinstance(value, np.ndarray):
                copied_fields[field_name] = value.copy()
            elif HAS_JAX and isinstance(value, jnp.ndarray):
                copied_fields[field_name] = value
            elif isinstance(value, dict):
                # Deep copy dict, including any nested arrays
                copied_dict = {}
                for key, v in value.items():
                    if isinstance(v, np.ndarray):
                        copied_dict[key] = v.copy()
                    elif HAS_JAX and isinstance(v, jnp.ndarray):
                        copied_dict[key] = v
                    else:
                        copied_dict[key] = v
                copied_fields[field_name] = copied_dict
            else:
                copied_fields[field_name] = value

        return self.__class__(**copied_fields)

    def _extract_nested_arrays(self, obj: Any, prefix: str = "") -> Dict[str, Array]:
        """
        Recursively extract numpy/JAX arrays from nested structures.

        Args:
            obj: The object to extract arrays from
            prefix: The current path prefix for nested keys

        Returns:
            Dictionary mapping dotted paths to numpy/JAX arrays
        """
        arrays = {}

        if isinstance(obj, dict):
            for key, value in obj.items():
                nested_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, np.ndarray) or (HAS_JAX and isinstance(value, jnp.ndarray)):
                    arrays[nested_key] = value
                elif isinstance(value, dict):
                    arrays.update(
                        self._extract_nested_arrays(value, nested_key))
        elif isinstance(obj, np.ndarray) or (HAS_JAX and isinstance(obj, jnp.ndarray)):
            arrays[prefix] = obj

        return arrays

    @property
    def array_fields(self) -> Set[str]:
        """
        Identify all numpy/JAX array fields in this class, including nested arrays in dictionaries.

        Returns:
            Set of field names (including dotted paths for nested arrays)
        """
        result = set()
        type_hints = get_type_hints(self.__class__)

        for field_name, field_type in type_hints.items():
            if field_name in self.__private_attributes__:
                continue
            try:
                value = getattr(self, field_name, None)
                if isinstance(value, np.ndarray) or (HAS_JAX and isinstance(value, jnp.ndarray)):
                    result.add(field_name)
                elif isinstance(value, dict):
                    # Extract nested arrays and add them with dotted notation
                    nested_arrays = self._extract_nested_arrays(
                        value, field_name)
                    result.update(nested_arrays.keys())
            except AttributeError:
                pass

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

                if isinstance(array, np.ndarray) or (HAS_JAX and isinstance(array, jnp.ndarray)):
                    encoded_arrays[field_name] = ArrayUtils.encode_array(array)
            else:
                # Handle direct array fields
                try:
                    array = getattr(self, field_name)
                    if isinstance(array, np.ndarray) or (HAS_JAX and isinstance(array, jnp.ndarray)):
                        encoded_arrays[field_name] = ArrayUtils.encode_array(
                            array)
                except AttributeError:
                    pass

        return EncodedData(arrays=encoded_arrays)

    def _extract_non_array_fields(self) -> Dict[str, Any]:
        """
        Extract non-array field values for metadata storage.

        Returns:
            Dictionary of field names to non-array values
        """
        def extract_non_arrays(obj: Any) -> Any:
            """Recursively extract non-array values from nested structures."""
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if isinstance(value, np.ndarray) or (HAS_JAX and isinstance(value, jnp.ndarray)):
                        continue
                    elif isinstance(value, dict):
                        nested_result = extract_non_arrays(value)
                        if nested_result:
                            result[key] = nested_result
                    else:
                        result[key] = value
                return result if result else None
            elif isinstance(obj, np.ndarray) or (HAS_JAX and isinstance(obj, jnp.ndarray)):
                return None
            else:
                return obj

        model_data = {}
        for field_name, field_value in self.model_dump().items():
            if field_name in self.array_fields:
                continue

            if isinstance(field_value, dict):
                non_array_content = extract_non_arrays(field_value)
                if non_array_content:
                    model_data[field_name] = non_array_content
            else:
                model_data[field_name] = field_value

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

            # Merge non-array field values from metadata
            ZipUtils.merge_field_data(data, metadata.field_data)

            return cls(**data)

    def to_numpy(self: T) -> T:
        """
        Create a new Packable with all arrays converted to NumPy arrays.

        Returns:
            A new Packable with all arrays as NumPy arrays
        """
        if not HAS_JAX:
            return self.copy()

        data_copy = self.copy()

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

        data_copy = self.copy()

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
