"""
Utilities for compressing numpy arrays.

This module provides functions for compressing numpy arrays using meshoptimizer's
encoding functions and storing/loading them as encoded data.
"""
import ctypes
import json
from io import BytesIO
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
from meshoptimizer._loader import lib
from meshoptimizer import (
    encode_vertex_buffer,
    encode_index_sequence,
    decode_vertex_buffer,
    decode_index_sequence,
)
from pydantic import BaseModel, Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from .common import PathLike
from .data_handler import DataHandler

# Optional JAX support
try:
    import jax.numpy as jnp
    HAS_JAX = True

except ImportError:
    jnp = None
    HAS_JAX = False

ArrayType = Literal["numpy", "jax"]

# Encoding types for arrays (public API)
EncodingType = Literal["array", "vertex_buffer", "index_sequence"]


class _ArrayAnnotation:
    """Pydantic annotation for numpy arrays with encoding hints.
    
    Used with Annotated to create typed array fields that generate
    proper JSON schemas with encoding information.
    """
    
    def __init__(self, encoding: EncodingType = "array"):
        self.encoding = encoding
    
    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Return core schema that accepts numpy/jax arrays."""
        return core_schema.is_instance_schema(np.ndarray)
    
    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Return JSON schema with type as the encoding."""
        return {"type": self.encoding}
    
    def __hash__(self):
        return hash(self.encoding)
    
    def __eq__(self, other):
        return isinstance(other, _ArrayAnnotation) and self.encoding == other.encoding


# Annotated array types for use in Pydantic models
Array = Annotated[np.ndarray, _ArrayAnnotation("array")]
"""Standard array encoding (meshoptimizer generic vertex buffer)."""

VertexBuffer = Annotated[np.ndarray, _ArrayAnnotation("vertex_buffer")]
"""Meshoptimizer vertex buffer encoding (optimized for 3D vertex data)."""

IndexSequence = Annotated[np.ndarray, _ArrayAnnotation("index_sequence")]
"""Meshoptimizer index sequence encoding (optimized for mesh indices)."""


class ArrayRefMetadata(BaseModel):
    """Metadata for array $ref objects in data.json.
    
    Contains all instance-specific information needed to decode an array.
    The encoding type comes from schema.json, not from this metadata.
    
    Example $ref in data.json:
        {"$ref": "abc123", "dtype": "float32", "shape": [100, 3], "vertex_count": 100, "vertex_size": 12}
    """
    
    ref: Optional[str] = Field(None, alias="$ref", description="Asset checksum reference")
    dtype: str = Field(..., description="NumPy dtype string (e.g., 'float32', 'uint32')")
    shape: list[int] = Field(..., description="Array shape")
    
    # For standard array encoding
    itemsize: Optional[int] = Field(None, description="Size of each item in bytes (for array encoding)")
    
    # For vertex_buffer encoding
    vertex_count: Optional[int] = Field(None, description="Number of vertices (for vertex_buffer)")
    vertex_size: Optional[int] = Field(None, description="Size of each vertex in bytes (for vertex_buffer)")
    
    # For index_sequence encoding
    index_count: Optional[int] = Field(None, description="Number of indices (for index_sequence)")
    
    model_config = {"populate_by_name": True}


class ResourceRefMetadata(BaseModel):
    """Metadata for resource $ref objects in data.json.
    
    Resources are gzip-compressed file data.
    
    Example $ref in data.json:
        {"$ref": "abc123", "name": "material.mtl"}
    """
    
    ref: str = Field(..., alias="$ref", description="Asset checksum reference")
    name: Optional[str] = Field(None, description="Original filename (e.g., 'material.mtl')")
    
    model_config = {"populate_by_name": True}


class PackableRefMetadata(BaseModel):
    """Metadata for self-contained packable $ref objects in data.json.
    
    Self-contained packables are encoded as zip files.
    
    Example $ref in data.json:
        {"$ref": "abc123"}
    """
    
    ref: str = Field(..., alias="$ref", description="Asset checksum reference")
    
    model_config = {"populate_by_name": True}


class ArrayMetadata(BaseModel):
    """
    Pydantic model representing metadata for an encoded array.

    Used in zip files to store array metadata.
    """

    shape: list[int] = Field(..., description="Shape of the array")
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
    def get_array_encoding(annotation: Any) -> EncodingType:
        """Extract encoding type from a type annotation.
        
        Args:
            annotation: Type annotation (may be Annotated with _ArrayAnnotation)
            
        Returns:
            The encoding type, or "array" as default
        """
        from typing import get_origin, get_args
        import types
        
        # Handle Optional/Union - unwrap to find the array type
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, types.UnionType):
            args = get_args(annotation)
            for arg in args:
                if arg is not type(None):
                    result = ArrayUtils.get_array_encoding(arg)
                    if result != "array":
                        return result
                    # Check if it's an Annotated array
                    if get_origin(arg) is Annotated:
                        inner_args = get_args(arg)
                        for inner in inner_args:
                            if isinstance(inner, _ArrayAnnotation):
                                return inner.encoding
        
        # Handle Annotated directly
        if get_origin(annotation) is Annotated:
            args = get_args(annotation)
            for arg in args:
                if isinstance(arg, _ArrayAnnotation):
                    return arg.encoding
        
        return "array"

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
    def extract_nested_arrays(
        obj,
        prefix: str = "",
        skip: Optional[callable] = None,
    ) -> dict[str, Array]:
        """Recursively extract arrays from nested dicts and BaseModel instances.

        Args:
            obj: Object to extract arrays from
            prefix: Path prefix for nested keys
            skip: Optional predicate - if skip(obj) is True, skip this object
        """
        arrays = {}
        if skip and skip(obj):
            pass
        elif ArrayUtils.is_array(obj):
            arrays[prefix] = obj
        elif isinstance(obj, BaseModel):
            for name in type(obj).model_fields:
                value = getattr(obj, name, None)
                if value is not None:
                    key = f"{prefix}.{name}" if prefix else name
                    arrays.update(ArrayUtils.extract_nested_arrays(value, key, skip))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                arrays.update(ArrayUtils.extract_nested_arrays(v, key, skip))
        return arrays

    @staticmethod
    def extract_non_arrays(obj, skip: Optional[callable] = None):
        """Extract non-array values from nested structures.

        Args:
            obj: Object to extract non-arrays from
            skip: Optional predicate - if skip(obj) is True, skip this object
        """
        if ArrayUtils.is_array(obj):
            return None
        if skip and skip(obj):
            return None
        if isinstance(obj, BaseModel):
            result = {}
            for name in type(obj).model_fields:
                val = getattr(obj, name, None)
                if not ArrayUtils.is_array(val) and not (skip and skip(val)):
                    extracted = ArrayUtils.extract_non_arrays(val, skip)
                    if extracted is not None:
                        result[name] = extracted
            return result or None
        if isinstance(obj, dict):
            result = {k: ArrayUtils.extract_non_arrays(v, skip) for k, v in obj.items()
                      if not ArrayUtils.is_array(v) and not (skip and skip(v))}
            result = {k: v for k, v in result.items() if v is not None}
            return result or None
        return obj

    @staticmethod
    def encode_array(array: np.ndarray) -> EncodedArray:
        """
        Encode a numpy array using meshoptimizer's vertex buffer encoding.

        Args:
            array: numpy array to encode (must have dtype with itemsize >= 4)

        Returns:
            EncodedArray object containing the encoded data and metadata
            
        Raises:
            ValueError: if array dtype has itemsize < 4
        """

        # Convert other arrays to numpy for encoding
        array = ArrayUtils.convert_array(array, "numpy")

        # Ensure contiguous array
        array = np.ascontiguousarray(array)

        # Store original shape and dtype
        original_shape = array.shape
        original_dtype = array.dtype

        # meshoptimizer requires vertex_size % 4 == 0
        if array.itemsize % 4 != 0:
            raise ValueError(
                f"Array dtype itemsize must be a multiple of 4, got {array.itemsize} "
                f"(dtype={original_dtype}). Use float32, int32, float64, or int64."
            )

        # Flatten the array if it's multi-dimensional
        flattened = array.reshape(-1)
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
            array_type=array_type,
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
        # Create buffer for encoded data
        buffer_array = np.frombuffer(encoded_array.data, dtype=np.uint8)

        # Get original dtype
        original_dtype = np.dtype(encoded_array.metadata.dtype)
        
        # Calculate total items based on shape
        total_items = int(np.prod(encoded_array.metadata.shape))
        
        # Create destination array with original dtype
        destination = np.zeros(total_items, dtype=original_dtype)

        result = lib.meshopt_decodeVertexBuffer(
            destination.ctypes.data_as(ctypes.c_void_p),
            total_items,
            original_dtype.itemsize,
            buffer_array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            len(buffer_array)
        )

        if result != 0:
            raise RuntimeError(f"Failed to decode array: error code {result}")

        # Reshape the array to its original shape
        reshaped = destination.reshape(encoded_array.metadata.shape)

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

    # ============================================================
    # Schema-based encoding/decoding for $ref objects
    # ============================================================

    @staticmethod
    def encode_with_schema(
        array: Array,
        encoding_type: EncodingType = "array",
        vertex_count: Optional[int] = None,
    ) -> tuple[bytes, dict]:
        """Encode an array and produce metadata for $ref.
        
        Returns raw encoded bytes (no header) and ArrayRefMetadata
        to be included directly in the $ref object in data.json.
        
        Args:
            array: Array to encode
            encoding_type: Encoding type (array, vertex_buffer, index_sequence)
            vertex_count: For index_sequence type, the number of vertices
            
        Returns:
            Tuple of (raw encoded bytes, ArrayRefMetadata)
        """
        array_np = ArrayUtils.convert_array(array, "numpy")
        
        if encoding_type == "vertex_buffer":
            if array_np.ndim == 1:
                raise ValueError("Vertex buffer requires 2D array (N x components)")
            v_count = array_np.shape[0]
            v_size = array_np.itemsize * array_np.shape[1]
            encoded_bytes = encode_vertex_buffer(array_np, v_count, v_size)
            metadata = ArrayRefMetadata(
                dtype=str(array_np.dtype),
                shape=list(array_np.shape),
                vertex_count=v_count,
                vertex_size=v_size,
            )
            
        elif encoding_type == "index_sequence":
            if array_np.ndim != 1:
                raise ValueError("Index sequence requires 1D array")
            i_count = len(array_np)
            v_count = vertex_count
            if v_count is None:
                v_count = int(array_np.max()) + 1 if len(array_np) > 0 else 0
            encoded_bytes = encode_index_sequence(array_np, i_count, v_count)
            metadata = ArrayRefMetadata(
                dtype=str(array_np.dtype),
                shape=list(array_np.shape),
                index_count=i_count,
                vertex_count=v_count,
            )
            
        else:
            # Standard meshoptimizer generic encoding - raw bytes, no header
            encoded = ArrayUtils.encode_array(array_np)
            encoded_bytes = encoded.data  # Raw bytes only
            metadata = ArrayRefMetadata(
                dtype=str(array_np.dtype),
                shape=list(array_np.shape),
                itemsize=encoded.metadata.itemsize,
            )
            
        return encoded_bytes, metadata

    @staticmethod
    def decode_with_metadata(
        data: bytes,
        encoding: EncodingType,
        metadata: "ArrayRefMetadata",
        target_array_type: Optional[ArrayType] = None,
    ) -> Array:
        """Decode raw bytes using encoding type and metadata from $ref.
        
        Args:
            data: Raw encoded bytes (no header)
            encoding: Encoding type from schema.json
            metadata: ArrayRefMetadata from data.json $ref object
            target_array_type: Target array type (defaults to "numpy")
            
        Returns:
            Decoded array
        """
        effective_type = target_array_type or "numpy"
        
        if encoding == "vertex_buffer":
            if metadata.vertex_count is None or metadata.vertex_size is None:
                raise ValueError("vertex_buffer missing vertex_count or vertex_size in metadata")
            decoded = decode_vertex_buffer(metadata.vertex_count, metadata.vertex_size, data)
            decoded = decoded.reshape(metadata.shape)
            if str(decoded.dtype) != metadata.dtype:
                decoded = decoded.astype(metadata.dtype)
                
        elif encoding == "index_sequence":
            if metadata.index_count is None:
                raise ValueError("index_sequence missing index_count in metadata")
            index_size = np.dtype(metadata.dtype).itemsize
            decoded = decode_index_sequence(metadata.index_count, index_size, data)
            if str(decoded.dtype) != metadata.dtype:
                decoded = decoded.astype(metadata.dtype)
                
        else:
            # Standard encoding - reconstruct from raw bytes
            itemsize = metadata.itemsize or 4
            arr_metadata = ArrayMetadata(
                shape=metadata.shape,
                dtype=metadata.dtype,
                itemsize=itemsize,
                array_type=effective_type,
            )
            encoded = EncodedArray(data=data, metadata=arr_metadata)
            decoded = ArrayUtils.decode_array(encoded)
            return decoded  # Already correct type
            
        return ArrayUtils.convert_array(decoded, effective_type)
