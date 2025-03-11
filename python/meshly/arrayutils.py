"""
Utilities for compressing numpy arrays.

This module provides functions for compressing numpy arrays using meshoptimizer's
encoding functions and storing/loading them as encoded data.
"""
import ctypes
import json
import struct
from typing import Tuple
import numpy as np
from meshoptimizer._loader import lib

class EncodedArray:
    """
    A class representing an encoded numpy array with metadata.
    
    Attributes:
        data: Encoded data as bytes
        shape: Original array shape
        dtype: Original array data type
        itemsize: Size of each item in bytes
    """
    def __init__(self, data: bytes, shape: Tuple[int, ...], dtype: np.dtype, itemsize: int):
        self.data = data
        self.shape = shape
        self.dtype = dtype
        self.itemsize = itemsize
    
    def __len__(self) -> int:
        """Return the length of the encoded data in bytes."""
        return len(self.data)

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
