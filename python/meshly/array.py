"""
Array encoding utilities for meshly.

This module provides:
- Type annotations: Array, IndexSequence for Pydantic models
- Metadata model: ArrayRefInfo for $ref serialization
- ArrayUtils: Encoding/decoding arrays using meshoptimizer compression

Encoding Types:
- "array": Generic array encoding (meshoptimizer vertex buffer)
- "index_sequence": Optimized for mesh indices (1D array)
"""

import json
import types
import zipfile
from io import BytesIO
from typing import Annotated, Any, Callable, Literal, Optional, Union, get_args, get_origin

import numpy as np
from pydantic import BaseModel, Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from meshly.common import PathLike, RefInfo


# =============================================================================
# Type Definitions
# =============================================================================

ArrayEncoding = Literal["array", "index_sequence"]
"""Array encoding strategies."""

ArrayType = Literal["numpy", "jax"]
"""Supported array backend types."""


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
# Pydantic Array Annotations
# =============================================================================

class _ArrayAnnotation:
    """Pydantic annotation marker for numpy arrays with encoding hints."""
    
    def __init__(self, encoding: ArrayEncoding = "array"):
        self.encoding = encoding
    
    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
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
    """Pydantic annotation for arrays serialized as inline JSON lists."""

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


# Public type aliases for Pydantic models
Array = Annotated[np.ndarray, _ArrayAnnotation("array")]
"""Generic array encoding (meshoptimizer vertex buffer)."""

IndexSequence = Annotated[np.ndarray, _ArrayAnnotation("index_sequence")]
"""Optimized for mesh indices (1D array)."""

List = Annotated[np.ndarray, _ListAnnotation()]
"""Array serialized as inline JSON list (no binary $ref)."""


# =============================================================================
# Metadata Models
# =============================================================================

class ArrayRefInfo(RefInfo):
    """Metadata for encoded arrays (stored in $ref).
    
    Example: {"$ref": "abc123", "dtype": "float32", "shape": [100, 3]}
    """
    ref: Optional[str] = Field(None, alias="$ref")
    shape: list[int]
    dtype: str
    itemsize: int


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
        """Extract encoding type from a type annotation (Array, IndexSequence)."""
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, types.UnionType):
            for arg in get_args(annotation):
                if arg is not type(None):
                    result = ArrayUtils.get_array_encoding(arg)
                    if result != "array":
                        return result
        
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
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, types.UnionType):
            for arg in get_args(annotation):
                if arg is not type(None):
                    if ArrayUtils.is_list_annotation(arg):
                        return True
            return False
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
        """Recursively extract arrays from nested dicts/BaseModels."""
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
    # Core Extract/Reconstruct
    # -------------------------------------------------------------------------

    @staticmethod
    def extract(array: object, encoding: ArrayEncoding = "array") -> ExtractedArray:
        """Extract array to ExtractedArray with meshoptimizer compression."""
        from meshoptimizer import encode_vertex_buffer, encode_index_sequence
        
        array_np = np.ascontiguousarray(ArrayUtils.convert_array(array, "numpy"))
        original_itemsize = array_np.dtype.itemsize
        
        if encoding == "index_sequence":
            if array_np.ndim != 1:
                raise ValueError("Index sequence requires 1D array")
            index_count = len(array_np)
            vertex_count = int(array_np.max()) + 1 if index_count > 0 else 0
            data = bytes(encode_index_sequence(array_np, index_count, vertex_count))
        else:
            flat = array_np.ravel()
            # meshoptimizer requires vertex_size to be multiple of 4
            # Pad elements if needed
            if original_itemsize % 4 != 0:
                padded_size = ((original_itemsize + 3) // 4) * 4
                padding = padded_size - original_itemsize
                flat_bytes = flat.tobytes()
                # Pad each element
                padded_chunks = [flat_bytes[i:i+original_itemsize] + b'\x00' * padding 
                                 for i in range(0, len(flat_bytes), original_itemsize)]
                padded_bytes = b''.join(padded_chunks)
                padded_array = np.frombuffer(padded_bytes, dtype=np.uint8).reshape(-1, padded_size)
                data = bytes(encode_vertex_buffer(padded_array, len(flat), padded_size))
            else:
                data = bytes(encode_vertex_buffer(flat, len(flat), original_itemsize))
        
        return ExtractedArray(
            data=data,
            info=ArrayRefInfo(
                shape=list(array_np.shape),
                dtype=str(array_np.dtype),
                itemsize=original_itemsize,
            ),
            encoding=encoding,
        )

    @staticmethod
    def reconstruct(
        extracted: ExtractedArray,
        array_type: ArrayType = "numpy",
    ) -> np.ndarray:
        """Reconstruct array from ExtractedArray."""
        from meshoptimizer import decode_vertex_buffer, decode_index_sequence
        
        metadata = extracted.info
        dtype = np.dtype(metadata.dtype)
        shape = tuple(metadata.shape)
        original_itemsize = metadata.itemsize
        
        if extracted.encoding == "index_sequence":
            decoded = decode_index_sequence(shape[0], dtype.itemsize, extracted.data)
            if decoded.dtype != dtype:
                decoded = decoded.astype(dtype)
        else:
            total_items = int(np.prod(shape))
            # Handle padded data (itemsize not multiple of 4)
            if original_itemsize % 4 != 0:
                padded_size = ((original_itemsize + 3) // 4) * 4
                # Decode as raw bytes (no dtype) - returns float32 by default, reinterpret as uint8
                decoded_raw = decode_vertex_buffer(total_items, padded_size, extracted.data)
                # Convert to bytes view
                decoded_bytes = decoded_raw.view(np.uint8).reshape(total_items, padded_size)
                # Strip padding from each element
                unpadded_bytes = decoded_bytes[:, :original_itemsize].tobytes()
                decoded = np.frombuffer(unpadded_bytes, dtype=dtype).reshape(shape)
            else:
                decoded = decode_vertex_buffer(total_items, dtype.itemsize, extracted.data, dtype=dtype)
                decoded = decoded.reshape(shape)
        
        return ArrayUtils.convert_array(decoded, array_type)

    @staticmethod
    def decode(
        zf: zipfile.ZipFile,
        name: str,
        encoding: ArrayEncoding,
        array_type: ArrayType = "numpy"
    ) -> Any:
        """Decode a single array from a zip file."""
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

        extracted = ExtractedArray(data=encoded_bytes, info=metadata, encoding=encoding)
        return ArrayUtils.reconstruct(extracted, array_type)

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
        """Save a single encoded array to a zip file."""
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
        """Save a single array to a zip file."""
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
        """Load a single array from a zip file."""
        if isinstance(source, BytesIO):
            source.seek(0)
            buf = BytesIO(source.read())
        else:
            with open(source, "rb") as f:
                buf = BytesIO(f.read())

        with zipfile.ZipFile(buf, "r") as zf:
            return ArrayUtils.decode(zf, name, "array", array_type)
