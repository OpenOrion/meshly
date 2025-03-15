"""
High-level mesh abstraction for easier use of meshoptimizer.

This module provides:
1. Mesh class as a Pydantic base class for representing 3D meshes
2. MeshUtils class for mesh optimization and encoding/decoding operations
3. Functions for encoding and decoding meshes
"""

import json
from pathlib import Path
import zipfile
from io import BytesIO
from typing import ClassVar, Dict, Optional, Set, Type, TypeVar, Union, get_type_hints

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator

# Use meshoptimizer directly
from meshoptimizer import (
    # Encoder functions
    encode_vertex_buffer,
    encode_index_buffer,
    decode_vertex_buffer,
    decode_index_buffer,
    optimize_vertex_cache,
    optimize_overdraw,
    optimize_vertex_fetch,
    simplify,
)

from .array import EncodedArray, ArrayUtils
from .models import EncodedMesh, ArrayMetadata, MeshMetadata, MeshFileMetadata

PathLike = Union[str, Path]


# Type variable for the Mesh class
T = TypeVar("T", bound="Mesh")


class Mesh(BaseModel):
    """
    A Pydantic base class representing a 3D mesh.

    Users can inherit from this class to define custom mesh types with additional
    numpy array attributes that will be automatically encoded/decoded.
    """

    # Required fields
    vertices: np.ndarray = Field(..., description="Vertex data as a numpy array")
    indices: Optional[np.ndarray] = Field(
        None, description="Index data as a numpy array"
    )

    # Private attributes to store computed values
    _vertex_count: int = PrivateAttr(0)
    _index_count: int = PrivateAttr(0)
    _array_fields: ClassVar[Set[str]] = set()

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize a mesh with vertices and optional indices."""
        super().__init__(**data)
        self._update_counts()
        self._initialize_array_fields()

    def _update_counts(self):
        """Update vertex and index counts."""
        self._vertex_count = len(self.vertices)
        self._index_count = len(self.indices) if self.indices is not None else 0

    def _initialize_array_fields(self):
        """Identify all numpy array fields in this class."""
        cls = self.__class__
        type_hints = get_type_hints(cls)

        # Find all fields that are numpy arrays
        for field_name, field_type in type_hints.items():
            if field_name in self.__private_attributes__:
                continue

            try:
                value = getattr(self, field_name, None)
                if isinstance(value, np.ndarray):
                    cls._array_fields.add(field_name)
            except AttributeError:
                # Skip attributes that don't exist
                pass

    @property
    def vertex_count(self) -> int:
        """Get the number of vertices."""
        return self._vertex_count

    @property
    def index_count(self) -> int:
        """Get the number of indices."""
        return self._index_count

    @model_validator(mode="after")
    def validate_arrays(self) -> "Mesh":
        """Validate and convert arrays to the correct types."""
        # Ensure vertices is a float32 array
        if self.vertices is not None:
            self.vertices = np.asarray(self.vertices, dtype=np.float32)

        # Ensure indices is a uint32 array if present
        if self.indices is not None:
            self.indices = np.asarray(self.indices, dtype=np.uint32)

        # Update counts
        self._update_counts()
        return self


class MeshUtils:
    """
    Utility class for mesh optimization and encoding/decoding operations.
    """

    @staticmethod
    def optimize_vertex_cache(mesh: Mesh) -> Mesh:
        """
        Optimize the mesh for vertex cache efficiency.

        Args:
            mesh: The mesh to optimize

        Returns:
            The optimized mesh
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        optimized_indices = np.zeros_like(mesh.indices)
        optimize_vertex_cache(
            optimized_indices, mesh.indices, mesh._index_count, mesh._vertex_count
        )

        mesh.indices = optimized_indices
        return mesh

    @staticmethod
    def optimize_overdraw(mesh: Mesh, threshold: float = 1.05) -> Mesh:
        """
        Optimize the mesh for overdraw.

        Args:
            mesh: The mesh to optimize
            threshold: threshold for optimization (default: 1.05)

        Returns:
            The optimized mesh
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        optimized_indices = np.zeros_like(mesh.indices)
        optimize_overdraw(
            optimized_indices,
            mesh.indices,
            mesh.vertices,
            mesh._index_count,
            mesh._vertex_count,
            mesh.vertices.itemsize * mesh.vertices.shape[1],
            threshold,
        )

        mesh.indices = optimized_indices
        return mesh

    @staticmethod
    def optimize_vertex_fetch(mesh: Mesh) -> Mesh:
        """
        Optimize the mesh for vertex fetch efficiency.

        Args:
            mesh: The mesh to optimize

        Returns:
            The optimized mesh
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        optimized_vertices = np.zeros_like(mesh.vertices)
        unique_vertex_count = optimize_vertex_fetch(
            optimized_vertices,
            mesh.indices,
            mesh.vertices,
            mesh._index_count,
            mesh._vertex_count,
            mesh.vertices.itemsize * mesh.vertices.shape[1],
        )

        mesh.vertices = optimized_vertices[:unique_vertex_count]
        mesh._vertex_count = unique_vertex_count
        return mesh

    @staticmethod
    def simplify(
        mesh: Mesh, target_ratio: float = 0.25, target_error: float = 0.01, options: int = 0
    ) -> Mesh:
        """
        Simplify the mesh.

        Args:
            mesh: The mesh to simplify
            target_ratio: ratio of triangles to keep (default: 0.25)
            target_error: target error (default: 0.01)
            options: simplification options (default: 0)

        Returns:
            The simplified mesh
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to simplify")

        target_index_count = int(mesh._index_count * target_ratio)
        simplified_indices = np.zeros(mesh._index_count, dtype=np.uint32)

        result_error = np.array([0.0], dtype=np.float32)
        new_index_count = simplify(
            simplified_indices,
            mesh.indices,
            mesh.vertices,
            mesh._index_count,
            mesh._vertex_count,
            mesh.vertices.itemsize * mesh.vertices.shape[1],
            target_index_count,
            target_error,
            options,
            result_error,
        )

        mesh.indices = simplified_indices[:new_index_count]
        mesh._index_count = new_index_count
        return mesh

    @staticmethod
    def encode(mesh: Mesh) -> Dict[str, Union[EncodedMesh, Dict[str, EncodedArray]]]:
        """
        Encode the mesh and all numpy array fields for efficient transmission.

        Args:
            mesh: The mesh to encode

        Returns:
            Dictionary containing:
            - 'mesh': EncodedMesh object with encoded vertices and indices
            - 'arrays': Dictionary mapping field names to EncodedArray objects
        """
        # Encode vertex buffer
        encoded_vertices = encode_vertex_buffer(
            mesh.vertices,
            mesh._vertex_count,
            mesh.vertices.itemsize * mesh.vertices.shape[1],
        )

        # Encode index buffer if present
        encoded_indices = None
        if mesh.indices is not None:
            encoded_indices = encode_index_buffer(
                mesh.indices, mesh._index_count, mesh.indices.itemsize
            )

        # Create encoded mesh
        encoded_mesh = EncodedMesh(
            vertices=encoded_vertices,
            indices=encoded_indices,
            vertex_count=mesh._vertex_count,
            vertex_size=mesh.vertices.itemsize * mesh.vertices.shape[1],
            index_count=mesh._index_count if mesh.indices is not None else None,
            index_size=mesh.indices.itemsize if mesh.indices is not None else 4,
        )

        # Encode additional array fields
        encoded_arrays = {}
        for field_name in mesh.__class__._array_fields:
            if field_name in ("vertices", "indices"):
                continue  # Skip the main vertices and indices

            try:
                array = getattr(mesh, field_name)
                if isinstance(array, np.ndarray):
                    encoded_arrays[field_name] = ArrayUtils.encode_array(array)
            except AttributeError:
                # Skip attributes that don't exist
                pass

        return {"mesh": encoded_mesh, "arrays": encoded_arrays}

    @staticmethod
    def save_to_zip(mesh: Mesh, source: Union[PathLike, BytesIO]) -> None:
        """
        Save the mesh to a zip file.

        Args:
            mesh: The mesh to save
            source: Path to the output zip file
        """
        encoded_data = MeshUtils.encode(mesh)
        encoded_mesh = encoded_data["mesh"]
        encoded_arrays = encoded_data["arrays"]

        # Create metadata
        metadata = MeshFileMetadata(
            class_name=mesh.__class__.__name__, module_name=mesh.__class__.__module__
        )

        # Add model fields that aren't numpy arrays
        model_data = {}
        for field_name, field_value in mesh.model_dump().items():
            if field_name not in mesh.__class__._array_fields:
                model_data[field_name] = field_value

        if model_data:
            metadata.field_data = model_data

        with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Save mesh data
            zipf.writestr("mesh/vertices.bin", encoded_mesh.vertices)
            if encoded_mesh.indices is not None:
                zipf.writestr("mesh/indices.bin", encoded_mesh.indices)

            # Save mesh metadata
            mesh_metadata = MeshMetadata(
                vertex_count=encoded_mesh.vertex_count,
                vertex_size=encoded_mesh.vertex_size,
                index_count=encoded_mesh.index_count,
                index_size=encoded_mesh.index_size,
            )
            zipf.writestr(
                "mesh/metadata.json", json.dumps(mesh_metadata.model_dump(), indent=2)
            )

            # Save array data
            for name, encoded_array in encoded_arrays.items():
                zipf.writestr(f"arrays/{name}.bin", encoded_array.data)

                # Save array metadata
                array_metadata = ArrayMetadata(
                    shape=list(encoded_array.shape),
                    dtype=str(encoded_array.dtype),
                    itemsize=encoded_array.itemsize,
                )
                zipf.writestr(
                    f"arrays/{name}_metadata.json",
                    json.dumps(array_metadata.model_dump(), indent=2),
                )

            # Save general metadata
            zipf.writestr("metadata.json", json.dumps(metadata.model_dump(), indent=2))

    @staticmethod
    def load_from_zip(cls: Type[T], destination: Union[PathLike, BytesIO]) -> T:
        """
        Load a mesh from a zip file.

        Args:
            cls: The mesh class to instantiate
            destination: Path to the input zip file

        Returns:
            Mesh object loaded from the zip file
        """
        with zipfile.ZipFile(destination, "r") as zipf:
            # Load general metadata
            with zipf.open("metadata.json") as f:
                metadata_dict = json.loads(f.read().decode("utf-8"))
                metadata = MeshFileMetadata(**metadata_dict)

            # Check if the class matches
            class_name = metadata.class_name
            module_name = metadata.module_name

            # If the class doesn't match, try to import it
            if class_name != cls.__name__ or module_name != cls.__module__:
                try:
                    import importlib

                    module = importlib.import_module(module_name)
                    target_cls = getattr(module, class_name)
                except (ImportError, AttributeError):
                    # Fall back to the provided class
                    target_cls = cls
            else:
                target_cls = cls

            # Load mesh metadata
            with zipf.open("mesh/metadata.json") as f:
                mesh_metadata_dict = json.loads(f.read().decode("utf-8"))
                mesh_metadata = MeshMetadata(**mesh_metadata_dict)

            # Load mesh data
            with zipf.open("mesh/vertices.bin") as f:
                encoded_vertices = f.read()

            encoded_indices = None
            if "mesh/indices.bin" in zipf.namelist():
                with zipf.open("mesh/indices.bin") as f:
                    encoded_indices = f.read()

            # Create encoded mesh model
            encoded_mesh = EncodedMesh(
                vertices=encoded_vertices,
                indices=encoded_indices,
                vertex_count=mesh_metadata.vertex_count,
                vertex_size=mesh_metadata.vertex_size,
                index_count=mesh_metadata.index_count,
                index_size=mesh_metadata.index_size,
            )

            # Decode mesh data
            vertices = decode_vertex_buffer(
                encoded_mesh.vertex_count,
                encoded_mesh.vertex_size,
                encoded_mesh.vertices,
            )

            indices = None
            if (
                encoded_mesh.indices is not None
                and encoded_mesh.index_count is not None
            ):
                indices = decode_index_buffer(
                    encoded_mesh.index_count,
                    encoded_mesh.index_size,
                    encoded_mesh.indices,
                )

            # Load additional array data
            arrays = {}
            for name in [
                n
                for n in zipf.namelist()
                if n.startswith("arrays/") and n.endswith(".bin")
            ]:
                array_name = name.split("/")[1].split(".")[0]

                # Load array metadata
                with zipf.open(f"arrays/{array_name}_metadata.json") as f:
                    array_metadata_dict = json.loads(f.read().decode("utf-8"))
                    array_metadata = ArrayMetadata(**array_metadata_dict)

                # Load array data
                with zipf.open(name) as f:
                    encoded_data = f.read()

                # Create encoded array
                encoded_array = EncodedArray(
                    data=encoded_data,
                    shape=tuple(array_metadata.shape),
                    dtype=np.dtype(array_metadata.dtype),
                    itemsize=array_metadata.itemsize,
                )

                # Decode array
                arrays[array_name] = ArrayUtils.decode_array(encoded_array)

            # Create mesh object with all data
            mesh_data = {"vertices": vertices, "indices": indices, **arrays}

            # Add non-array model data if present
            if metadata.field_data is not None:
                mesh_data.update(metadata.field_data)

            return target_cls(**mesh_data)

    @staticmethod
    def decode(cls: Type[T], encoded_mesh: EncodedMesh) -> T:
        """
        Decode an encoded mesh.

        Args:
            cls: The mesh class to instantiate
            encoded_mesh: EncodedMesh object to decode

        Returns:
            Decoded Mesh object
        """
        # Decode vertex buffer
        vertices = decode_vertex_buffer(
            encoded_mesh.vertex_count, encoded_mesh.vertex_size, encoded_mesh.vertices
        )

        # Decode index buffer if present
        indices = None
        if encoded_mesh.indices is not None and encoded_mesh.index_count is not None:
            indices = decode_index_buffer(
                encoded_mesh.index_count, encoded_mesh.index_size, encoded_mesh.indices
            )

        return cls(vertices=vertices, indices=indices)
