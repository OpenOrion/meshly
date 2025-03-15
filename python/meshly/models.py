"""
Pydantic models for mesh data structures.

This module provides Pydantic classes for the data structures used in the mesh module,
including encoded mesh data, array metadata, and serialization formats.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, RootModel



class EncodedMesh(BaseModel):
    """
    Pydantic model representing an encoded mesh with its vertices and indices.

    This is a Pydantic version of the EncodedMesh class in mesh.py.
    """

    vertices: bytes = Field(..., description="Encoded vertex buffer")
    indices: Optional[bytes] = Field(
        None, description="Encoded index buffer (optional)"
    )
    vertex_count: int = Field(..., description="Number of vertices")
    vertex_size: int = Field(..., description="Size of each vertex in bytes")
    index_count: Optional[int] = Field(None, description="Number of indices (optional)")
    index_size: int = Field(..., description="Size of each index in bytes")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


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


class MeshMetadata(BaseModel):
    """
    Pydantic model representing metadata for an encoded mesh.

    Used in the save_to_zip method to store mesh metadata.
    """

    vertex_count: int = Field(..., description="Number of vertices")
    vertex_size: int = Field(..., description="Size of each vertex in bytes")
    index_count: Optional[int] = Field(None, description="Number of indices (optional)")
    index_size: int = Field(..., description="Size of each index in bytes")


class ModelData(RootModel):
    """
    Pydantic model representing non-array model data.

    Used in the save_to_zip method to store model data that isn't numpy arrays.
    """

    root: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of model fields that aren't numpy arrays",
    )


class MeshFileMetadata(BaseModel):
    """
    Pydantic model representing general metadata for a mesh file.

    Used in the save_to_zip method to store general metadata.
    """

    class_name: str = Field(..., description="Name of the mesh class")
    module_name: str = Field(
        ..., description="Name of the module containing the mesh class"
    )
    field_data: Optional[Dict[str, Any]] = Field(
        None, description="Dictionary of model fields that aren't numpy arrays"
    )
