"""
Utilities for compressing numpy arrays.

This module provides functions for compressing numpy arrays using meshoptimizer's
encoding functions and storing/loading them as encoded data.
"""
import ctypes
from io import BytesIO
import json
from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np
from pydantic import BaseModel, Field
from meshoptimizer._loader import lib

from .data_handler import DataHandler
from .common import PathLike

# Optional JAX support
try:
    import jax.numpy as jnp
    HAS_JAX = True
    JaxArray = Union[np.ndarray, jnp.ndarray]

except ImportError:
    jnp = None
    HAS_JAX = False
    JaxArray = np.ndarray

Array = Union[np.ndarray, JaxArray]
ArrayType = Literal["numpy", "jax"]


class ArrayMetadata(BaseModel):
    """
    Pydantic model representing metadata for an encoded array.

    Used in zip files to store array metadata.
    """

    shape: List[int] = Field(..., description="Shape of the array")
    dtype: str = Field(..., description="Data type of the array as string")
    itemsize: int = Field(..., description="Size of each item in bytes")
    array_type: ArrayType = Field(
        default="numpy", description="Array backend type (numpy or jax)")


class EncodedArray(BaseModel):
    """
    A class representing an encoded numpy array with metadata.

    Attributes:
        data: Encoded data as bytes
        shape: Original array shape
        dtype: Original array data type
        itemsize: Size of each item in bytes
        array_type: Array backend type (numpy or jax)
    """
    data: bytes
    metadata: ArrayMetadata

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

    def __len__(self) -> int:
        """Return the length of the encoded data in bytes."""
        return len(self.data)


class ArrayUtils:
    """Utility class for encoding and decoding numpy arrays."""

    @staticmethod
    def is_array(obj) -> bool:
        """Check if obj is a numpy or JAX array."""
        if isinstance(obj, np.ndarray):
            return True
        if HAS_JAX and jnp is not None and isinstance(obj, jnp.ndarray):
            return True
        return False

    @staticmethod
    def detect_array_type(array: Array) -> ArrayType:
        """Detect whether an array is numpy or jax."""
        if HAS_JAX and jnp is not None and isinstance(array, jnp.ndarray):
            return "jax"
        return "numpy"

    @staticmethod
    def convert_array(array: Array, array_type: ArrayType):
        """
        Convert an array to the specified array backend type.

        Args:
            array: The array to convert (numpy or jax)
            array_type: Target array type ("numpy" or "jax")

        Returns:
            Array in the specified backend format

        Raises:
            AssertionError: If JAX is requested but not available
        """
        if array_type == "jax":
            assert HAS_JAX, "JAX is not available. Install JAX to use JAX arrays."
            if isinstance(array, jnp.ndarray):
                return array  # Already JAX
            return jnp.array(array)
        elif array_type == "numpy":  # numpy
            return np.asarray(array)
        else:
            raise ValueError(f"Unsupported array_type: {array_type}")

    @staticmethod
    def convert_recursive(obj, array_type: ArrayType):
        """
        Recursively convert arrays in nested structures to the specified type.

        Args:
            obj: Object to convert (array, dict, list, tuple, or other)
            array_type: Target array type ("numpy" or "jax")

        Returns:
            Object with all arrays converted to the specified type
        """
        if ArrayUtils.is_array(obj):
            return ArrayUtils.convert_array(obj, array_type)
        elif isinstance(obj, dict):
            return {key: ArrayUtils.convert_recursive(value, array_type) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(ArrayUtils.convert_recursive(item, array_type) for item in obj)
        else:
            return obj

    @staticmethod
    def extract_nested_arrays(obj, prefix: str = "") -> Dict[str, Array]:
        """Recursively extract arrays from nested dicts and BaseModel instances.

        Note: Packable instances are skipped - they handle their own encoding.
        """
        from pydantic import BaseModel
        from .packable import Packable
        arrays = {}
        if ArrayUtils.is_array(obj):
            arrays[prefix] = obj
        elif isinstance(obj, Packable):
            # Skip Packable instances - they encode themselves
            pass
        elif isinstance(obj, BaseModel):
            for name in type(obj).model_fields:
                value = getattr(obj, name, None)
                if value is not None:
                    key = f"{prefix}.{name}" if prefix else name
                    arrays.update(ArrayUtils.extract_nested_arrays(value, key))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                arrays.update(ArrayUtils.extract_nested_arrays(v, key))
        return arrays

    @staticmethod
    def extract_non_arrays(obj):
        """Extract non-array values, preserving BaseModel type info for reconstruction.

        Note: Packable instances are skipped - they handle their own encoding.
        """
        from pydantic import BaseModel
        from .packable import Packable
        if ArrayUtils.is_array(obj):
            return None
        if isinstance(obj, Packable):
            # Skip Packable instances - they encode themselves
            return None
        if isinstance(obj, BaseModel):
            result = {"__model_class__": obj.__class__.__name__,
                      "__model_module__": obj.__class__.__module__}
            for name in type(obj).model_fields:
                val = getattr(obj, name, None)
                if not ArrayUtils.is_array(val) and not isinstance(val, Packable):
                    extracted = ArrayUtils.extract_non_arrays(val)
                    if extracted is not None:
                        result[name] = extracted
            return result if len(result) > 2 else None
        if isinstance(obj, dict):
            result = {k: ArrayUtils.extract_non_arrays(v) for k, v in obj.items()
                      if not ArrayUtils.is_array(v) and not isinstance(v, Packable)}
            result = {k: v for k, v in result.items() if v is not None}
            return result or None
        return obj

    @staticmethod
    def encode_array(array: np.ndarray) -> EncodedArray:
        """
        Encode a numpy array using meshoptimizer's vertex buffer encoding.

        Args:
            array: numpy array to encode

        Returns:
            EncodedArray object containing the encoded data and metadata
        """

        # Convert other arrays to numpy for encoding
        array = ArrayUtils.convert_array(array, "numpy")

        # Store original shape and dtype
        original_shape = array.shape
        original_dtype = array.dtype

        # Flatten the array if it's multi-dimensional
        flattened = array.reshape(-1)

        # Convert to float32 if not already (meshoptimizer expects float32)
        if array.dtype != np.float32:
            flattened = flattened.astype(np.float32)

        # Calculate parameters for encoding
        item_count = len(flattened)
        item_size = flattened.itemsize

        # Calculate buffer size
        bound = lib.meshopt_encodeVertexBufferBound(item_count, item_size)

        # Allocate buffer
        buffer = np.zeros(bound, dtype=np.uint8)

        # Call C function
        result_size = lib.meshopt_encodeVertexBuffer(
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            bound,
            flattened.ctypes.data_as(ctypes.c_void_p),
            item_count,
            item_size
        )

        if result_size == 0:
            raise RuntimeError("Failed to encode array")

        # Return only the used portion of the buffer
        encoded_data = bytes(buffer[:result_size])

        array_type = ArrayUtils.detect_array_type(array)

        metadata = ArrayMetadata(
            shape=list(original_shape),
            dtype=str(original_dtype),
            itemsize=item_size,
            array_type=array_type
        )

        return EncodedArray(
            data=encoded_data,
            metadata=metadata
        )

    @staticmethod
    def decode_array(encoded_array: EncodedArray) -> np.ndarray:
        """
        Decode an encoded array.

        Args:
            encoded_array: EncodedArray object containing encoded data and metadata

        Returns:
            Decoded numpy array
        """
        # Calculate total number of items
        total_items = np.prod(encoded_array.metadata.shape)

        # Create buffer for encoded data
        buffer_array = np.frombuffer(encoded_array.data, dtype=np.uint8)

        # Create destination array for float32 data
        float_count = total_items
        destination = np.zeros(float_count, dtype=np.float32)

        # Call C function
        result = lib.meshopt_decodeVertexBuffer(
            destination.ctypes.data_as(ctypes.c_void_p),
            total_items,
            encoded_array.metadata.itemsize,
            buffer_array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            len(buffer_array)
        )

        if result != 0:
            raise RuntimeError(f"Failed to decode array: error code {result}")

        # Reshape the array to its original shape
        reshaped = destination.reshape(encoded_array.metadata.shape)

        # Convert back to original dtype if needed
        if encoded_array.metadata.dtype != np.float32:
            reshaped = reshaped.astype(encoded_array.metadata.dtype)

        return reshaped

    @staticmethod
    def save_array(
        handler: DataHandler,
        name: str,
        encoded_array: EncodedArray,
    ) -> None:
        """
        Save a single encoded array using a write handler.

        Args:
            handler: DataHandler for writing files
            name: Array name (e.g., "normals" or "markerIndices.boundary")
            encoded_array: EncodedArray to save
        """
        array_path = name.replace(".", "/")
        handler.write_binary(
            f"arrays/{array_path}/array.bin", encoded_array.data)
        handler.write_text(
            f"arrays/{array_path}/metadata.json",
            json.dumps(encoded_array.metadata.model_dump(),
                       indent=2, sort_keys=True),
        )

    @staticmethod
    def load_array(
        handler: DataHandler,
        name: str,
        array_type: Optional[ArrayType] = None
    ) -> Any:
        """
        Load and decode a single array using a read handler.

        Args:
            handler: DataHandler for reading files
            name: Array name (e.g., "normals" or "markerIndices.boundary")
            array_type: Target array backend type ("numpy" or "jax"). If None (default), uses
                    the array_type stored in the array's metadata.

        Returns:
            Decoded array (numpy or JAX)

        Raises:
            KeyError: If array not found
        """
        array_path = name.replace(".", "/")
        bin_path = f"arrays/{array_path}/array.bin"
        meta_path = f"arrays/{array_path}/metadata.json"

        try:
            metadata_text = handler.read_text(meta_path)
            metadata_dict = json.loads(metadata_text)
            metadata = ArrayMetadata(**metadata_dict)

            encoded_bytes = handler.read_binary(bin_path)
        except (KeyError, FileNotFoundError) as e:
            raise KeyError(f"Array '{name}' not found") from e

        encoded = EncodedArray(
            data=encoded_bytes,
            metadata=metadata
        )

        decoded = ArrayUtils.decode_array(encoded)
        return ArrayUtils.convert_array(decoded, array_type or metadata.array_type)

    @staticmethod
    def save_to_zip(
        array: Array,
        destination: Union[PathLike, BytesIO],
    ) -> None:
        """
        Save a single array to a zip file.

        Args:
            array: Array to save (numpy or JAX)
            destination: Path to the output zip file or BytesIO buffer
        """
        encoded = ArrayUtils.encode_array(array)

        zip_buffer = BytesIO()
        handler = DataHandler.create(zip_buffer)
        ArrayUtils.save_array(handler, "array", encoded)
        handler.finalize()

        if isinstance(destination, BytesIO):
            destination.write(zip_buffer.getvalue())
        else:
            with open(destination, "wb") as f:
                f.write(zip_buffer.getvalue())

    @staticmethod
    def load_from_zip(
        source: Union[PathLike, BytesIO],
        array_type: Optional[ArrayType] = None
    ) -> Array:
        """
        Load a single array from a zip file.

        Args:
            source: Path to the input zip file or BytesIO buffer
            array_type: Target array backend type ("numpy" or "jax"). If None (default),
                       uses the array_type stored in the array's metadata.

        Returns:
            Decoded array (numpy or JAX)
        """
        if isinstance(source, BytesIO):
            source.seek(0)
            handler = DataHandler.create(BytesIO(source.read()))
        else:
            with open(source, "rb") as f:
                handler = DataHandler.create(BytesIO(f.read()))

        return ArrayUtils.load_array(handler, "array", array_type)
