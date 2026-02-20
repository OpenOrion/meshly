"""
Array encoding utilities for meshly.

This module provides:
- Type annotations: Array, VertexBuffer, IndexSequence for Pydantic models
- Metadata model: ArrayRefModel for $ref serialization
- ArrayUtils: Encoding/decoding arrays using meshoptimizer compression

Encoding Types:
- "array": Generic meshoptimizer vertex buffer (works for any array)
- "vertex_buffer": Optimized for 3D vertex data (N x components)
- "index_sequence": Optimized for mesh indices (1D array)
"""

import ctypes
import json
import types
import zipfile
from io import BytesIO
from typing import Annotated, Any, Callable, Literal, Optional, Union, get_args, get_origin

import numpy as np
from meshoptimizer._loader import lib
from meshoptimizer import decode_vertex_buffer, decode_index_sequence
from pydantic import BaseModel, Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
from meshoptimizer import encode_vertex_buffer, encode_index_sequence

from meshly.common import PathLike, RefInfo


# =============================================================================
# JAX Support (optional dependency)
# =============================================================================

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = None
    HAS_JAX = False


# =============================================================================
# Type Definitions
# =============================================================================

ArrayType = Literal["numpy", "jax"]
"""Supported array backend types."""

ArrayEncoding = Literal["array", "vertex_buffer", "index_sequence"]
"""Array encoding types for serialization."""


# =============================================================================
# Pydantic Array Annotations (Array, VertexBuffer, IndexSequence)
# =============================================================================

class _ArrayAnnotation:
    """Pydantic annotation marker for numpy arrays with encoding hints."""
    
    def __init__(self, encoding: ArrayEncoding = "array"):
        self.encoding = encoding
    
    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Validate that input is a numpy or JAX array."""
        return core_schema.no_info_plain_validator_function(self._validate)
    
    def _validate(self, v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        if HAS_JAX and hasattr(v, '__jax_array__'):
            return v
        type_module = type(v).__module__
        if type_module.startswith('jax') or type_module.startswith('jaxlib'):
            return v
        raise ValueError(f"Expected numpy or JAX array, got {type(v)}")
    
    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": self.encoding}
    
    def __hash__(self):
        return hash(self.encoding)

    def __eq__(self, other):
        return isinstance(other, _ArrayAnnotation) and self.encoding == other.encoding


class _ListAnnotation:
    """Pydantic annotation marker for numpy arrays serialized as inline JSON lists.

    Same validation as _ArrayAnnotation (accepts numpy, JAX, list/tuple → np.ndarray),
    but serializes as .tolist() instead of binary $ref assets.
    """

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(self._validate)

    def _validate(self, v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        if isinstance(v, (list, tuple)):
            return np.array(v)
        if HAS_JAX and hasattr(v, '__jax_array__'):
            return v
        type_module = type(v).__module__
        if type_module.startswith('jax') or type_module.startswith('jaxlib'):
            return v
        raise ValueError(f"Expected numpy/JAX array or list/tuple, got {type(v)}")

    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "list"}

    def __hash__(self):
        return hash("list")

    def __eq__(self, other):
        return isinstance(other, _ListAnnotation)


# Public type aliases for use in Pydantic models
Array = Annotated[np.ndarray, _ArrayAnnotation("array")]
"""Generic array encoding (meshoptimizer vertex buffer)."""

VertexBuffer = Annotated[np.ndarray, _ArrayAnnotation("vertex_buffer")]
"""Optimized for 3D vertex data (N x components)."""

IndexSequence = Annotated[np.ndarray, _ArrayAnnotation("index_sequence")]
"""Optimized for mesh indices (1D array)."""

List = Annotated[np.ndarray, _ListAnnotation()]
"""Array serialized as inline JSON list (no binary $ref)."""


# =============================================================================
# Metadata Models (for $ref serialization)
# =============================================================================

class ArrayRefInfo(RefInfo):
    """Metadata for encoded arrays (stored in zip files and $ref in data.json).
    
    Example $ref: {"$ref": "abc123", "dtype": "float32", "shape": [100, 3]}
    """
    # $ref (for data.json serialization)
    ref: Optional[str] = Field(None, alias="$ref")
    
    # Core fields (always present)
    shape: list[int]
    dtype: str
    itemsize: int
    
    # Encoding fields
    pad_bytes: Optional[int] = None


class ExtractedArray(BaseModel):
    """Extracted array data with metadata."""
    data: bytes
    info: ArrayRefInfo
    encoding: ArrayEncoding = "array"

    class Config:
        arbitrary_types_allowed = True

    def __len__(self) -> int:
        return len(self.data)


# =============================================================================
# ArrayUtils
# =============================================================================

class ArrayUtils:
    """Utilities for encoding/decoding numpy arrays with meshoptimizer."""

    # -------------------------------------------------------------------------
    # Type Detection & Conversion
    # -------------------------------------------------------------------------

    @staticmethod
    def is_array(obj: Any) -> bool:
        """Check if obj is a numpy or JAX array."""
        if isinstance(obj, np.ndarray):
            return True
        return HAS_JAX and jnp is not None and isinstance(obj, jnp.ndarray)

    @staticmethod
    def detect_array_type(array: np.ndarray) -> ArrayType:
        """Detect whether an array is numpy or jax."""
        if HAS_JAX and jnp is not None and isinstance(array, jnp.ndarray):
            return "jax"
        return "numpy"

    @staticmethod
    def convert_array(array: np.ndarray, array_type: ArrayType) -> np.ndarray:
        """Convert an array to the specified backend type."""
        if array_type == "jax":
            assert HAS_JAX, "JAX is not available"
            return array if isinstance(array, jnp.ndarray) else jnp.array(array)
        if array_type == "numpy":
            return np.asarray(array)
        raise ValueError(f"Unsupported array_type: {array_type}")

    @staticmethod
    def convert_recursive(obj: Any, array_type: ArrayType) -> Any:
        """Recursively convert all arrays in nested structures."""
        if ArrayUtils.is_array(obj):
            return ArrayUtils.convert_array(obj, array_type)
        if isinstance(obj, dict):
            return {k: ArrayUtils.convert_recursive(v, array_type) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(ArrayUtils.convert_recursive(item, array_type) for item in obj)
        return obj

    # -------------------------------------------------------------------------
    # Type Annotation Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def get_array_encoding(annotation: Any) -> ArrayEncoding:
        """Extract encoding type from a type annotation (Array, VertexBuffer, etc)."""
        # Handle Optional/Union - unwrap to find the array type
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, types.UnionType):
            for arg in get_args(annotation):
                if arg is not type(None):
                    result = ArrayUtils.get_array_encoding(arg)
                    if result != "array":
                        return result
        
        # Handle Annotated types
        if get_origin(annotation) is Annotated:
            for arg in get_args(annotation):
                if isinstance(arg, _ArrayAnnotation):
                    return arg.encoding
        
        return "array"

    @staticmethod
    def is_list_annotation(annotation: Any) -> bool:
        """Check if a type annotation contains _ListAnnotation."""
        if annotation is None:
            return False
        # Handle Optional/Union - unwrap to find the inner type
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, types.UnionType):
            for arg in get_args(annotation):
                if arg is not type(None):
                    if ArrayUtils.is_list_annotation(arg):
                        return True
            return False
        # Handle Annotated types
        if get_origin(annotation) is Annotated:
            for arg in get_args(annotation):
                if isinstance(arg, _ListAnnotation):
                    return True
        return False

    # -------------------------------------------------------------------------
    # Nested Structure Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_nested_arrays(
        obj: Any,
        prefix: str = "",
        skip: Optional[Callable[[Any], bool]] = None,
    ) -> dict[str, np.ndarray]:
        """Recursively extract arrays from nested dicts/BaseModels.
        
        Returns dict mapping dotted paths to arrays (e.g., "mesh.vertices").
        """
        if skip and skip(obj):
            return {}
        if ArrayUtils.is_array(obj):
            return {prefix: obj} if prefix else {"": obj}
        if isinstance(obj, BaseModel):
            arrays = {}
            for name in type(obj).model_fields:
                value = getattr(obj, name, None)
                if value is not None:
                    key = f"{prefix}.{name}" if prefix else name
                    arrays.update(ArrayUtils.extract_nested_arrays(value, key, skip))
            return arrays
        if isinstance(obj, dict):
            arrays = {}
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                arrays.update(ArrayUtils.extract_nested_arrays(v, key, skip))
            return arrays
        return {}

    # -------------------------------------------------------------------------
    # Core Extract/Reconstruct (meshoptimizer)
    # -------------------------------------------------------------------------

    @staticmethod
    def extract(
        array: object,
        encoding: ArrayEncoding = "array",
    ) -> ExtractedArray:
        """Extract array to ExtractedArray with meshoptimizer compression.
        
        Args:
            array: Array to extract (numpy or JAX)
            encoding: Encoding type ("array", "vertex_buffer", "index_sequence")
            
        Returns:
            ExtractedArray with encoded data and metadata
        """
        array_np = np.ascontiguousarray(ArrayUtils.convert_array(array, "numpy"))
        original_shape = list(array_np.shape)
        original_dtype = str(array_np.dtype)
        
        # Vertex buffer: optimized for 2D vertex data (N x components)
        if encoding == "vertex_buffer":
            if array_np.ndim == 1:
                raise ValueError("Vertex buffer requires 2D array")
            vertex_count = array_np.shape[0]
            vertex_size = array_np.itemsize * array_np.shape[1]
            data = encode_vertex_buffer(array_np, vertex_count, vertex_size)
            return ExtractedArray(
                data=bytes(data),
                info=ArrayRefInfo(
                    shape=original_shape,
                    dtype=original_dtype,
                    itemsize=array_np.dtype.itemsize,
                ),
                encoding=encoding,
            )
        
        # Index sequence: optimized for 1D mesh indices
        if encoding == "index_sequence":
            if array_np.ndim != 1:
                raise ValueError("Index sequence requires 1D array")
            index_count = len(array_np)
            data = encode_index_sequence(array_np, index_count, int(array_np.max()) + 1 if index_count > 0 else 0)
            return ExtractedArray(
                data=bytes(data),
                info=ArrayRefInfo(
                    shape=original_shape,
                    dtype=original_dtype,
                    itemsize=array_np.dtype.itemsize,
                ),
                encoding=encoding,
            )
        
        # Generic array: meshoptimizer vertex buffer encoding with padding
        # meshoptimizer requires item_size % 4 == 0, so pad if needed
        pad_bytes = (4 - array_np.itemsize % 4) % 4
        flattened = array_np.reshape(-1)
        item_count = len(flattened)
        
        # Pad each element if needed (e.g., float16 -> 4 bytes)
        if pad_bytes > 0:
            padded_itemsize = array_np.itemsize + pad_bytes
            padded = np.zeros(item_count * padded_itemsize, dtype=np.uint8)
            for i in range(item_count):
                src_bytes = np.frombuffer(flattened[i:i+1].tobytes(), dtype=np.uint8)
                padded[i * padded_itemsize : i * padded_itemsize + array_np.itemsize] = src_bytes
            flattened = padded
            item_size = padded_itemsize
        else:
            item_size = flattened.itemsize

        # Encode with meshoptimizer vertex buffer encoder
        bound = lib.meshopt_encodeVertexBufferBound(item_count, item_size)
        buffer = np.zeros(bound, dtype=np.uint8)
        result_size = lib.meshopt_encodeVertexBuffer(
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            bound,
            flattened.ctypes.data_as(ctypes.c_void_p),
            item_count,
            item_size
        )
        if result_size == 0:
            raise RuntimeError("Failed to encode array")

        return ExtractedArray(
            data=bytes(buffer[:result_size]),
            info=ArrayRefInfo(
                shape=original_shape,
                dtype=original_dtype,
                itemsize=array_np.dtype.itemsize,
                pad_bytes=pad_bytes if pad_bytes > 0 else None,
            ),
            encoding=encoding,
        )



    @staticmethod
    def reconstruct(
        extracted: ExtractedArray,
        array_type: ArrayType = "numpy",
    ) -> np.ndarray:
        """Reconstruct array from ExtractedArray.
        
        Args:
            extracted: ExtractedArray with data, metadata, and encoding
            array_type: Target array type (defaults to "numpy")
            
        Returns:
            Reconstructed array (numpy or JAX)
        """
        metadata = extracted.info
        encoding = extracted.encoding
        
        # Vertex buffer: 2D array optimized encoding
        if encoding == "vertex_buffer":
            vertex_count = metadata.shape[0]
            vertex_size = metadata.itemsize * metadata.shape[1] if len(metadata.shape) > 1 else metadata.itemsize
            decoded = decode_vertex_buffer(vertex_count, vertex_size, extracted.data)
            decoded = decoded.reshape(metadata.shape)
            if str(decoded.dtype) != metadata.dtype:
                decoded = decoded.astype(metadata.dtype)
            return ArrayUtils.convert_array(decoded, array_type)
        
        # Index sequence: 1D index array encoding
        if encoding == "index_sequence":
            index_count = metadata.shape[0]
            index_size = np.dtype(metadata.dtype).itemsize
            decoded = decode_index_sequence(index_count, index_size, extracted.data)
            if str(decoded.dtype) != metadata.dtype:
                decoded = decoded.astype(metadata.dtype)
            return ArrayUtils.convert_array(decoded, array_type)
        
        # Generic array: meshoptimizer vertex buffer encoding with optional padding
        buffer_array = np.frombuffer(extracted.data, dtype=np.uint8)
        original_dtype = np.dtype(metadata.dtype)
        pad_bytes = metadata.pad_bytes or 0
        total_items = int(np.prod(metadata.shape))
        
        if pad_bytes > 0:
            padded_itemsize = original_dtype.itemsize + pad_bytes
            destination = np.zeros(total_items * padded_itemsize, dtype=np.uint8)
            
            result = lib.meshopt_decodeVertexBuffer(
                destination.ctypes.data_as(ctypes.c_void_p),
                total_items,
                padded_itemsize,
                buffer_array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                len(buffer_array)
            )
            if result != 0:
                raise RuntimeError(f"Failed to decode array: error code {result}")
            
            # Strip padding from each element
            unpadded = np.zeros(total_items, dtype=original_dtype)
            for i in range(total_items):
                src_start = i * padded_itemsize
                src_bytes = destination[src_start:src_start + original_dtype.itemsize]
                unpadded[i] = np.frombuffer(src_bytes.tobytes(), dtype=original_dtype)[0]
            decoded = unpadded.reshape(metadata.shape)
        else:
            # Direct decode without padding
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
            decoded = destination.reshape(metadata.shape)

        return ArrayUtils.convert_array(decoded, array_type)


    @staticmethod
    def decode(
        zf: zipfile.ZipFile,
        name: str,
        encoding: ArrayEncoding,
        array_type: ArrayType = "numpy"
    ) -> Any:
        """
        Decode a single array from a zip file.

        Args:
            zf: ZipFile opened for reading
            name: Array name (e.g., "normals" or "markerIndices.boundary")
            encoding: Encoding type used for the array
            array_type: Target array backend type ("numpy" or "jax")

        Returns:
            Decoded array (numpy or JAX)

        Raises:
            KeyError: If array not found
        """
        array_path = name.replace(".", "/")
        bin_path = f"arrays/{array_path}/array.bin"
        meta_path = f"arrays/{array_path}/metadata.json"

        try:
            metadata_text = zf.read(meta_path).decode("utf-8")
            metadata_dict = json.loads(metadata_text)
            metadata = ArrayRefInfo(**metadata_dict)

            encoded_bytes = zf.read(bin_path)
        except KeyError as e:
            raise KeyError(f"Array '{name}' not found") from e

        encoded_array = ExtractedArray(
            data=encoded_bytes,
            info=metadata,
            encoding=encoding,
        )

        return ArrayUtils.reconstruct(encoded_array, array_type)

    # -------------------------------------------------------------------------
    # Zip File I/O
    # -------------------------------------------------------------------------

    @staticmethod
    def save_array(
        zf: zipfile.ZipFile,
        name: str,
        encoded_array: ExtractedArray,
        date_time: tuple[int, int, int, int, int, int] = (2020, 1, 1, 0, 0, 0),
    ) -> None:
        """
        Save a single encoded array to a zip file.

        Args:
            zf: ZipFile opened for writing
            name: Array name (e.g., "normals" or "markerIndices.boundary")
            encoded_array: ExtractedArray to save
            date_time: Timestamp for deterministic output (default: 2020-01-01 00:00:00)
        """
        array_path = name.replace(".", "/")
        
        info = zipfile.ZipInfo(f"arrays/{array_path}/array.bin", date_time=date_time)
        zf.writestr(info, encoded_array.data)
        
        info = zipfile.ZipInfo(f"arrays/{array_path}/metadata.json", date_time=date_time)
        zf.writestr(info, json.dumps(encoded_array.info.model_dump(), indent=2, sort_keys=True))


    @staticmethod
    def save_to_zip(
        array: Array,
        destination: Union[PathLike, BytesIO],
        name: str = "array",
    ) -> None:
        """
        Save a single array to a zip file.

        Args:
            array: Array to save (numpy or JAX)
            destination: Path to the output zip file or BytesIO buffer
            name: Array name in the zip file (default: "array")
        """
        extracted = ArrayUtils.extract(array)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            ArrayUtils.save_array(zf, name, extracted)

        if isinstance(destination, BytesIO):
            destination.write(zip_buffer.getvalue())
        else:
            with open(destination, "wb") as f:
                f.write(zip_buffer.getvalue())

    @staticmethod
    def load_from_zip(
        source: Union[PathLike, BytesIO],
        name: str = "array",
        array_type: ArrayType = "numpy",
    ) -> Array:
        """
        Load a single array from a zip file.

        Args:
            source: Path to the input zip file or BytesIO buffer
            name: Array name in the zip file (default: "array")
            array_type: Target array backend type ("numpy" or "jax")

        Returns:
            Decoded array (numpy or JAX)
        """
        if isinstance(source, BytesIO):
            source.seek(0)
            buf = BytesIO(source.read())
        else:
            with open(source, "rb") as f:
                buf = BytesIO(f.read())

        with zipfile.ZipFile(buf, "r") as zf:
            return ArrayUtils.decode(zf, name, "array", array_type)
