"""
Utilities for compressing numpy arrays.

This module provides functions for compressing numpy arrays using meshoptimizer's
encoding functions and storing/loading them as encoded data.
"""
import ctypes
import json
import zipfile
from typing import List, Tuple, Union, Optional, Type, TypeVar, Any, Generic
from io import BytesIO
import numpy as np
from pydantic import BaseModel, Field
from meshoptimizer._loader import lib
from .common import PathLike



class EncodedArrayModel(BaseModel):
    """
    Pydantic model representing an encoded numpy array with metadata.

    This is a Pydantic version of the EncodedArray class in arrayutils.py.
    """

    data: bytes = Field(..., description="Encoded data as bytes")
    shape: Tuple[int, ...] = Field(..., description="Original array shape")
    dtype: str = Field(..., description="Original array data type as string")
    itemsize: int = Field(..., description="Size of each item in bytes")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

class ArrayMetadata(BaseModel):
    """
    Pydantic model representing metadata for an encoded array.

    Used in the save_to_zip method to store array metadata.
    """

    shape: List[int] = Field(..., description="Shape of the array")
    dtype: str = Field(..., description="Data type of the array as string")
    itemsize: int = Field(..., description="Size of each item in bytes")





# Type variable for custom metadata
T = TypeVar("T", bound=BaseModel)


class ArrayResult(BaseModel, Generic[T]):
    """
    Result class containing both the decoded array and optional custom metadata.
    
    Generic over the custom metadata type for better type safety.
    """
    array: np.ndarray = Field(..., description="The decoded numpy array")
    custom_metadata: Optional[T] = Field(None, description="Custom metadata if present")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class EncodedArray(BaseModel):
    """
    A class representing an encoded numpy array with metadata.
    
    Attributes:
        data: Encoded data as bytes
        shape: Original array shape
        dtype: Original array data type
        itemsize: Size of each item in bytes
    """
    data: bytes
    shape: Tuple[int, ...]
    dtype: np.dtype
    itemsize: int
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def __len__(self) -> int:
        """Return the length of the encoded data in bytes."""
        return len(self.data)


class ArrayUtils:
    """Utility class for encoding and decoding numpy arrays."""
    
    @staticmethod
    def encode_array(array: np.ndarray) -> EncodedArray:
        """
        Encode a numpy array using meshoptimizer's vertex buffer encoding.
        
        Args:
            array: numpy array to encode
            
        Returns:
            EncodedArray object containing the encoded data and metadata
        """
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
        
        return EncodedArray(
            data=encoded_data,
            shape=original_shape,
            dtype=original_dtype,
            itemsize=item_size
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
        total_items = np.prod(encoded_array.shape)
        
        # Create buffer for encoded data
        buffer_array = np.frombuffer(encoded_array.data, dtype=np.uint8)
        
        # Create destination array for float32 data
        float_count = total_items
        destination = np.zeros(float_count, dtype=np.float32)
        
        # Call C function
        result = lib.meshopt_decodeVertexBuffer(
            destination.ctypes.data_as(ctypes.c_void_p),
            total_items,
            encoded_array.itemsize,
            buffer_array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            len(buffer_array)
        )
        
        if result != 0:
            raise RuntimeError(f"Failed to decode array: error code {result}")
        
        # Reshape the array to its original shape
        reshaped = destination.reshape(encoded_array.shape)
        
        # Convert back to original dtype if needed
        if encoded_array.dtype != np.float32:
            reshaped = reshaped.astype(encoded_array.dtype)
        
        return reshaped

    @staticmethod
    def save_to_zip(
        array: np.ndarray,
        destination: Union["PathLike", BytesIO],
        custom_metadata: Optional[BaseModel] = None,
        date_time: Optional[tuple] = None
    ) -> None:
        """
        Save an array to a zip file with optional custom metadata.
        
        Args:
            array: numpy array to save
            destination: Path to the output zip file or BytesIO object
            custom_metadata: Optional custom metadata (Pydantic model subclassing BaseModel)
            date_time: Optional date_time tuple for deterministic zip files
        """
        # Encode the array
        encoded_array = ArrayUtils.encode_array(array)
        
        # Create array metadata
        array_metadata = ArrayMetadata(
            shape=list(encoded_array.shape),
            dtype=str(encoded_array.dtype),
            itemsize=encoded_array.itemsize,
        )
        
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            # Collect files to write
            files_to_write = [
                ("array.bin", encoded_array.data),
                ("metadata.json", json.dumps(array_metadata.model_dump(), indent=2, sort_keys=True))
            ]
            
            # Add custom metadata if provided
            if custom_metadata is not None:
                custom_metadata_dict = {
                    "class_name": custom_metadata.__class__.__name__,
                    "module_name": custom_metadata.__class__.__module__,
                    "data": custom_metadata.model_dump()
                }
                files_to_write.append((
                    "custom_metadata.json",
                    json.dumps(custom_metadata_dict, indent=2, sort_keys=True)
                ))
            
            # Write files in sorted order for deterministic output
            for filename, data in sorted(files_to_write):
                if date_time is not None:
                    info = zipfile.ZipInfo(filename=filename, date_time=date_time)
                else:
                    info = zipfile.ZipInfo(filename=filename)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16  # Fixed file permissions
                if isinstance(data, str):
                    data = data.encode('utf-8')
                zipf.writestr(info, data)

    @staticmethod
    def load_from_zip(
        source: Union["PathLike", BytesIO],
        custom_metadata_class: Optional[Type[T]] = None
    ) -> ArrayResult[T]:
        """
        Load an array from a zip file, optionally with custom metadata.
        
        Args:
            source: Path to the input zip file or BytesIO object
            custom_metadata_class: Optional custom metadata class to load
            
        Returns:
            ArrayResult containing the decoded array and custom metadata (None if no metadata class provided)
        """
        with zipfile.ZipFile(source, "r") as zipf:
            # Load basic metadata
            with zipf.open("metadata.json") as f:
                metadata_dict = json.loads(f.read().decode("utf-8"))
                array_metadata = ArrayMetadata(**metadata_dict)
            
            # Load binary data
            with zipf.open("array.bin") as f:
                encoded_data = f.read()
            
            # Create EncodedArray object
            encoded_array = EncodedArray(
                data=encoded_data,
                shape=tuple(array_metadata.shape),
                dtype=np.dtype(array_metadata.dtype),
                itemsize=array_metadata.itemsize
            )
            
            # Decode the array
            decoded_array = ArrayUtils.decode_array(encoded_array)
            
            # Load custom metadata if present and class is provided
            custom_metadata = None
            if custom_metadata_class is not None and "custom_metadata.json" in zipf.namelist():
                with zipf.open("custom_metadata.json") as f:
                    custom_metadata_dict = json.loads(f.read().decode("utf-8"))
                
                # Use the provided class
                custom_metadata = custom_metadata_class(**custom_metadata_dict["data"])
            
            return ArrayResult(array=decoded_array, custom_metadata=custom_metadata)

