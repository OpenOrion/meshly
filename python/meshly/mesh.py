"""
High-level mesh abstraction for easier use of meshoptimizer.

This module provides:
1. Mesh class as a Pydantic base class for representing 3D meshes
2. MeshUtils class for mesh optimization and encoding/decoding operations
3. Functions for encoding and decoding meshes

MeshUtils provides the following operations:
- triangulate: Convert meshes with mixed polygon types to pure triangle meshes
- optimize_vertex_cache: Optimize mesh for vertex cache efficiency
- optimize_overdraw: Optimize mesh to reduce overdraw
- optimize_vertex_fetch: Optimize mesh for vertex fetch efficiency
- simplify: Reduce mesh complexity while preserving shape
- encode/decode: Compress meshes for efficient storage and transmission
- save_to_zip/load_from_zip: Persist meshes to compressed archive format
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
    Literal,
    Sequence,
    Tuple,
    get_type_hints,
)
import numpy as np
from pydantic import BaseModel, Field, model_validator

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

# Use meshoptimizer directly
from meshoptimizer import (
    # Encoder functions
    encode_vertex_buffer,
    encode_index_sequence,
    decode_vertex_buffer,
    decode_index_sequence,
    optimize_vertex_cache,
    optimize_overdraw,
    optimize_vertex_fetch,
    simplify,
)

from .array import ArrayMetadata, EncodedArray, ArrayUtils
from .common import PathLike
from .cell_types import CellTypeUtils, VTKCellType
from .element_utils import ElementUtils, TriangulationUtils

# Type variable for the Mesh class
T = TypeVar("T", bound="Mesh")

# Removed ARRAY_NAME_SEPERATOR - now using nested directories


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
    index_count: Optional[int] = Field(
        None, description="Number of indices (optional)")
    index_size: int = Field(..., description="Size of each index in bytes")
    index_sizes: Optional[bytes] = Field(
        None, description="Encoded polygon sizes (optional)"
    )
    arrays: Dict[str, EncodedArray] = Field(
        default_factory=dict, description="Dictionary of additional encoded arrays"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class MeshSize(BaseModel):
    """
    Pydantic model representing size metadata for an encoded mesh.

    Used in the save_to_zip method to store mesh size information.
    """

    vertex_count: int = Field(..., description="Number of vertices")
    vertex_size: int = Field(..., description="Size of each vertex in bytes")
    index_count: Optional[int] = Field(
        None, description="Number of indices (optional)")
    index_size: int = Field(..., description="Size of each index in bytes")


class MeshMetadata(BaseModel):
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
    mesh_size: MeshSize = Field(
        description="Size metadata for the encoded mesh")


class Mesh(BaseModel):
    """
    A Pydantic base class representing a 3D mesh.

    Users can inherit from this class to define custom mesh types with additional
    numpy array attributes that will be automatically encoded/decoded.
    """

    # Required fields
    vertices: Array = Field(...,
                            description="Vertex data as a numpy or JAX array")
    indices: Optional[Union[Array, List[Any]]] = Field(
        None, description="Index data as a flattened 1D numpy/JAX array or list of polygons"
    )
    index_sizes: Optional[Union[Array, List[int]]] = None
    """
    Size of each polygon (number of vertices per polygon).
    If not provided, will be automatically inferred from indices structure:
    - For 2D numpy/JAX arrays: uniform polygon size from array shape
    - For list of lists: individual polygon sizes
    If explicitly provided, will be validated against inferred structure.
    """

    cell_types: Optional[Union[Array, List[int]]] = None
    """
    Cell type identifier for each polygon, corresponding to index_sizes.
    Common VTK cell types include:
    - 1: Vertex, 3: Line, 5: Triangle, 9: Quad, 10: Tetra, 12: Hexahedron, 13: Wedge, 14: Pyramid
    If not provided, will be automatically inferred from polygon sizes:
    - Size 1: Vertex (1), Size 2: Line (3), Size 3: Triangle (5), Size 4: Quad (9)
    If explicitly provided, must have same length as index_sizes.
    """

    # Mesh dimension - auto-computed from cell_types if not provided
    dim: Optional[int] = Field(
        default=None, description="Mesh dimension (2D or 3D). Auto-computed from cell types if not provided.")

    # Marker structure - accepts both sequence of sequences and flattened arrays, converts to flattened internally
    markers: Dict[str, Union[Sequence[Union[Sequence[int], Array]], Array]] = Field(
        default_factory=dict, description="marker node indices - accepts sequence of sequences or flattened arrays")
    # sizes of each marker element (standardized approach like index_sizes)
    marker_sizes: dict[str, Array] = Field(
        default_factory=dict, description="sizes of each marker element")
    # VTK cell types for each marker element, map to GMSH types with VTK_TO_GMSH_ELEMENT_TYPE
    marker_cell_types: dict[str, Array] = Field(
        default_factory=dict, description="VTK cell types for each marker element")

    def copy(self: T) -> T:
        """
        Create a deep copy of the mesh.

        Returns:
            A new mesh instance with copied data
        """
        # Create a dictionary to hold the copied fields
        copied_fields = {}

        # Copy all fields, with special handling for numpy/JAX arrays
        for field_name in self.model_fields_set:
            value = getattr(self, field_name)
            if isinstance(value, np.ndarray):
                copied_fields[field_name] = value.copy()
            elif HAS_JAX and isinstance(value, jnp.ndarray):
                copied_fields[field_name] = value
            else:
                copied_fields[field_name] = value

        # Create a new instance of the same class
        return self.__class__(**copied_fields)

    @property
    def vertex_count(self) -> int:
        """Get the number of vertices."""
        return len(self.vertices)

    @property
    def index_count(self) -> int:
        """Get the number of indices."""
        return len(self.indices) if self.indices is not None else 0

    @property
    def polygon_count(self) -> int:
        """Get the number of polygons."""
        return len(self.index_sizes) if self.index_sizes is not None else 0

    @property
    def is_uniform_polygons(self) -> bool:
        """Check if all polygons have the same number of vertices."""
        if self.index_sizes is None:
            return True  # No polygon info means uniform (legacy)
        return ElementUtils.is_uniform_elements(self.index_sizes)

    def get_polygon_indices(self) -> Union[Array, list]:
        """
        Get indices in their original polygon structure.

        Returns:
            For uniform polygons: 2D numpy/JAX array where each row is a polygon
            For mixed polygons: List of lists where each sublist is a polygon
        """
        if self.indices is None:
            return None

        if self.index_sizes is None:
            # Legacy format - assume triangles
            if len(self.indices) % 3 == 0:
                return self.indices.reshape(-1, 3)
            else:
                raise ValueError(
                    "Cannot determine polygon structure without index_sizes")

        # Use ElementUtils for consistent reconstruction
        return ElementUtils.get_element_structure(self.indices, self.index_sizes)

    def get_reconstructed_markers(self) -> Dict[str, List[List[int]]]:
        """Reconstruct marker elements from flattened structure back to list of lists"""
        reconstructed = {}

        for marker_name, flattened_indices in self.markers.items():
            sizes = self.marker_sizes[marker_name]
            marker_cell_types = self.marker_cell_types.get(marker_name, None)

            # Use ElementUtils to reconstruct elements
            try:
                elements = ElementUtils.convert_flattened_to_list(
                    flattened_indices, sizes, marker_cell_types
                )
                reconstructed[marker_name] = elements
            except ValueError as e:
                raise ValueError(
                    f"Error reconstructing marker '{marker_name}': {e}")

        return reconstructed

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
        """Identify all numpy/JAX array fields in this class, including nested arrays in dictionaries."""
        result = set()
        type_hints = get_type_hints(self.__class__)

        # Find all fields that are numpy arrays or contain numpy arrays
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
                # Skip attributes that don't exist
                pass

        return result

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_arrays(self) -> "Mesh":
        """
        Validate and convert arrays to the correct types.

        This method handles various input formats for indices and automatically infers
        index_sizes when not explicitly provided:

        - 2D numpy arrays: Assumes uniform polygons, infers size from array shape
        - List of lists: Supports mixed polygon sizes, infers from individual polygon lengths
        - Flat arrays: Requires explicit index_sizes for polygon structure

        When index_sizes is explicitly provided, it validates that the structure matches
        the inferred polygon sizes and that the sum equals the total number of indices.

        Cell types are automatically inferred from polygon sizes if not provided:
        - Size 1: Vertex (1), Size 2: Line (3), Size 3: Triangle (5), Size 4: Quad (9)

        Raises:
            ValueError: If explicit index_sizes doesn't match inferred structure or
                       if sum of index_sizes doesn't match total indices count, or
                       if cell_types length doesn't match index_sizes length.
        """
        # Ensure vertices is a float32 array, preserving array type (numpy/JAX)
        if self.vertices is not None:
            if HAS_JAX and isinstance(self.vertices, jnp.ndarray):
                # Keep as JAX array
                self.vertices = self.vertices.astype(jnp.float32)
            else:
                # Convert to numpy array
                self.vertices = np.asarray(self.vertices, dtype=np.float32)

        # Handle indices - convert to flattened 1D array and extract size info using ElementUtils
        if self.indices is not None:
            # Convert JAX arrays to numpy first if needed
            indices_to_process = self.indices
            index_sizes_to_process = self.index_sizes
            cell_types_to_process = self.cell_types

            if HAS_JAX and isinstance(indices_to_process, jnp.ndarray):
                indices_to_process = np.asarray(indices_to_process)
            if HAS_JAX and isinstance(index_sizes_to_process, jnp.ndarray):
                index_sizes_to_process = np.asarray(index_sizes_to_process)
            if HAS_JAX and isinstance(cell_types_to_process, jnp.ndarray):
                cell_types_to_process = np.asarray(cell_types_to_process)

            try:
                self.indices, self.index_sizes, self.cell_types = ElementUtils.convert_array_input(
                    indices_to_process, index_sizes_to_process, cell_types_to_process
                )
            except ValueError as e:
                raise ValueError(f"Error processing indices: {e}")

        # Auto-compute dimension from cell types if not explicitly provided
        if self.dim is None:
            if self.cell_types is not None and len(self.cell_types) > 0:
                self.dim = CellTypeUtils.get_mesh_dimension(self.cell_types)
            else:
                # Default to 3D if no cell types available
                self.dim = 3

        # Handle marker conversion - convert sequence format to flattened arrays
        if self.markers:
            converted_markers = {}
            for marker_name, marker_data in self.markers.items():
                try:
                    # Handle JAX arrays
                    marker_data_to_process = marker_data
                    if HAS_JAX and isinstance(marker_data_to_process, jnp.ndarray):
                        marker_data_to_process = np.asarray(
                            marker_data_to_process)

                    if isinstance(marker_data_to_process, np.ndarray):
                        # Already a numpy array, keep as is but validate it has corresponding sizes/types
                        converted_markers[marker_name] = np.asarray(
                            marker_data_to_process, dtype=np.uint32)

                        # If marker_cell_types is defined but marker_sizes is missing, calculate it automatically
                        if marker_name in self.marker_cell_types and marker_name not in self.marker_sizes:
                            self.marker_sizes[marker_name] = CellTypeUtils.infer_sizes_from_vtk_cell_types(
                                self.marker_cell_types[marker_name])

                        # Validate that we have both sizes and types
                        if marker_name not in self.marker_sizes or marker_name not in self.marker_cell_types:
                            raise ValueError(
                                f"Marker '{marker_name}' provided as array but missing marker_sizes or marker_cell_types")
                    else:
                        # Convert sequence of sequences to flattened structure using ElementUtils
                        # This handles lists, tuples, or any sequence type
                        marker_list = [list(element)
                                       for element in marker_data_to_process]
                        flattened_indices, sizes, cell_types = ElementUtils.convert_list_to_flattened(
                            marker_list)
                        converted_markers[marker_name] = flattened_indices
                        self.marker_sizes[marker_name] = sizes
                        self.marker_cell_types[marker_name] = cell_types

                except ValueError as e:
                    raise ValueError(
                        f"Error converting markers for '{marker_name}': {e}")

            # Update markers to be the flattened arrays
            self.markers = converted_markers

        return self

    @staticmethod
    def combine(
        meshes: List["Mesh"],
        marker_names: Optional[List[str]] = None,
        preserve_markers: bool = True,
    ) -> "Mesh":
        """
        Combine multiple meshes into a single mesh.

        Args:
            meshes: List of Mesh objects to combine
            marker_names: Optional list of marker names to assign to each mesh.
                         If provided, each mesh is assigned exclusively to its corresponding marker name,
                         completely replacing any existing markers from that mesh.
                         Must have same length as meshes list.
            preserve_markers: Whether to preserve existing markers from source meshes (default: True).
                            Only applies when marker_names is None. If marker_names is provided, this is ignored.
                            If True, existing markers are kept with their original names.
                            If multiple meshes have the same marker name, their elements are combined.

        Returns:
            A new combined Mesh object

        Raises:
            ValueError: If meshes list is empty or if marker_names length doesn't match meshes length

        Example:
            >>> mesh1 = Mesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), indices=np.array([0, 1, 2]))
            >>> mesh2 = Mesh(vertices=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]]), indices=np.array([0, 1, 2]))
            >>> combined = Mesh.combine([mesh1, mesh2], marker_names=["part1", "part2"])
        """
        if not meshes:
            raise ValueError("Cannot combine empty list of meshes")

        if marker_names is not None and len(marker_names) != len(meshes):
            raise ValueError(
                f"marker_names length ({len(marker_names)}) must match meshes length ({len(meshes)})"
            )

        # Pre-compute vertex offsets for all meshes
        vertex_offsets = MeshUtils.compute_vertex_offsets(meshes)

        # Collect vertices and indices from all meshes
        all_vertices = [mesh.vertices for mesh in meshes]
        all_indices = [mesh.indices + vertex_offsets[i]
                       for i, mesh in enumerate(meshes) if mesh.indices is not None]
        all_index_sizes = [
            mesh.index_sizes for mesh in meshes if mesh.index_sizes is not None]
        all_cell_types = [
            mesh.cell_types for mesh in meshes if mesh.cell_types is not None]

        # Build markers using utility methods
        if marker_names is not None:
            combined_markers, combined_marker_sizes, combined_marker_cell_types = \
                MeshUtils.combine_markers_with_names(
                    meshes, marker_names, vertex_offsets)
        elif preserve_markers:
            combined_markers, combined_marker_sizes, combined_marker_cell_types = \
                MeshUtils.preserve_existing_markers(meshes, vertex_offsets)
        else:
            combined_markers = {}
            combined_marker_sizes = {}
            combined_marker_cell_types = {}

        # Concatenate all arrays
        combined_vertices = np.concatenate(all_vertices, axis=0)
        combined_indices = np.concatenate(
            all_indices, axis=0) if all_indices else None
        combined_index_sizes = np.concatenate(
            all_index_sizes, axis=0) if all_index_sizes else None
        combined_cell_types_array = np.concatenate(
            all_cell_types, axis=0) if all_cell_types else None

        # Get dimension from first mesh
        dim = meshes[0].dim

        # Create combined mesh
        return Mesh(
            vertices=combined_vertices,
            indices=combined_indices,
            index_sizes=combined_index_sizes,
            cell_types=combined_cell_types_array,
            dim=dim,
            markers=combined_markers,
            marker_sizes=combined_marker_sizes,
            marker_cell_types=combined_marker_cell_types,
        )

    def extract_by_marker(self, marker_name: str) -> "Mesh":
        """
        Extract a submesh containing only the elements referenced by a specific marker.

        This method creates a new mesh containing only the vertices and elements (if any)
        that are referenced by the specified marker.

        Args:
            marker_name: Name of the marker to extract

        Returns:
            A new Mesh object containing only the vertices/elements from the marker

        Raises:
            ValueError: If marker_name doesn't exist in the mesh

        Example:
            >>> mesh = Mesh(vertices=vertices, indices=indices, markers={"boundary": [0, 1, 2]})
            >>> boundary_mesh = mesh.extract_by_marker("boundary")
        """
        if marker_name not in self.markers:
            raise ValueError(
                f"Marker '{marker_name}' not found. Available markers: {list(self.markers.keys())}"
            )

        # Get marker data
        marker_indices = self.markers[marker_name]
        marker_sizes = self.marker_sizes.get(marker_name)
        marker_cell_types = self.marker_cell_types.get(marker_name)

        if marker_sizes is None or marker_cell_types is None:
            raise ValueError(
                f"Marker '{marker_name}' is missing size or cell type information"
            )

        # Reconstruct marker elements
        marker_elements = ElementUtils.get_element_structure(
            marker_indices, marker_sizes
        )

        # Find all unique vertex indices referenced by the marker
        unique_vertices = np.unique(marker_indices)

        # Extract vertices
        extracted_vertices = self.vertices[unique_vertices]

        # Create vectorized mapping using searchsorted for O(n log n) instead of O(n^2)
        # searchsorted finds where each marker_index would be inserted in the sorted unique_vertices array
        remapped_indices = np.searchsorted(
            unique_vertices, marker_indices).astype(np.uint32)

        # Create new mesh with extracted data
        return Mesh(
            vertices=extracted_vertices,
            indices=remapped_indices,
            index_sizes=marker_sizes.copy(),
            cell_types=marker_cell_types.copy(),
            dim=self.dim,
        )


class MeshUtils:
    """
    Utility class for mesh optimization and encoding/decoding operations.
    """

    @staticmethod
    def compute_vertex_offsets(meshes: List[Mesh]) -> np.ndarray:
        """
        Compute vertex offsets for combining multiple meshes.

        Args:
            meshes: List of meshes to compute offsets for

        Returns:
            Array of vertex offsets for each mesh
        """
        vertex_counts = np.array([mesh.vertex_count for mesh in meshes])
        return np.concatenate([[0], np.cumsum(vertex_counts)[:-1]])

    @staticmethod
    def add_marker_to_dict(
        marker_dict: Dict[str, np.ndarray],
        marker_sizes: Dict[str, np.ndarray],
        marker_cell_types: Dict[str, np.ndarray],
        marker_name: str,
        indices: np.ndarray,
        sizes: np.ndarray,
        cell_types: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add or append marker data to marker dictionaries.

        If the marker name already exists, the data is concatenated.
        Otherwise, a new marker is created.

        Args:
            marker_dict: Dictionary of marker indices (modified in place)
            marker_sizes: Dictionary of marker sizes (modified in place)
            marker_cell_types: Dictionary of marker cell types (modified in place)
            marker_name: Name of the marker
            indices: Indices for this marker
            sizes: Sizes for this marker
            cell_types: Optional cell types for this marker
        """
        if marker_name in marker_dict:
            marker_dict[marker_name] = np.concatenate(
                [marker_dict[marker_name], indices]
            )
            marker_sizes[marker_name] = np.concatenate(
                [marker_sizes[marker_name], sizes]
            )
            if cell_types is not None:
                marker_cell_types[marker_name] = np.concatenate(
                    [marker_cell_types[marker_name], cell_types]
                )
        else:
            marker_dict[marker_name] = indices
            marker_sizes[marker_name] = sizes
            if cell_types is not None:
                marker_cell_types[marker_name] = cell_types

    @staticmethod
    def create_cell_marker_from_mesh(
        mesh: Mesh,
        vertex_offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Create marker data from a mesh's cell structure.

        Args:
            mesh: Source mesh
            vertex_offset: Offset to apply to vertex indices

        Returns:
            Tuple of (indices, sizes, cell_types)
        """
        if mesh.indices is not None and mesh.index_sizes is not None:
            # Create marker from mesh's cell structure
            indices = mesh.indices.copy() + vertex_offset
            sizes = mesh.index_sizes.copy()
            cell_types = mesh.cell_types.copy() if mesh.cell_types is not None else None
            return indices, sizes, cell_types
        else:
            # No cells, create vertex marker instead
            vertex_count = mesh.vertex_count
            indices = np.arange(vertex_offset, vertex_offset +
                                vertex_count, dtype=np.uint32)
            sizes = np.ones(vertex_count, dtype=np.uint32)
            cell_types = np.ones(vertex_count, dtype=np.uint32)
            return indices, sizes, cell_types

    @staticmethod
    def combine_markers_with_names(
        meshes: List[Mesh],
        marker_names: List[str],
        vertex_offsets: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Create markers from meshes using provided marker names.

        Each mesh is assigned to its corresponding marker name.

        Args:
            meshes: List of meshes
            marker_names: List of marker names (must match length of meshes)
            vertex_offsets: Precomputed vertex offsets for each mesh

        Returns:
            Tuple of (markers, marker_sizes, marker_cell_types) dictionaries
        """
        combined_markers = {}
        combined_marker_sizes = {}
        combined_marker_cell_types = {}

        for mesh_idx, marker_name in enumerate(marker_names):
            mesh = meshes[mesh_idx]
            indices, sizes, cell_types = MeshUtils.create_cell_marker_from_mesh(
                mesh, vertex_offsets[mesh_idx]
            )
            MeshUtils.add_marker_to_dict(
                combined_markers,
                combined_marker_sizes,
                combined_marker_cell_types,
                marker_name,
                indices,
                sizes,
                cell_types,
            )

        return combined_markers, combined_marker_sizes, combined_marker_cell_types

    @staticmethod
    def preserve_existing_markers(
        meshes: List[Mesh],
        vertex_offsets: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Preserve existing markers from source meshes.

        If multiple meshes have the same marker name, their elements are combined.

        Args:
            meshes: List of meshes
            vertex_offsets: Precomputed vertex offsets for each mesh

        Returns:
            Tuple of (markers, marker_sizes, marker_cell_types) dictionaries
        """
        combined_markers = {}
        combined_marker_sizes = {}
        combined_marker_cell_types = {}

        for mesh_idx, mesh in enumerate(meshes):
            if not mesh.markers:
                continue

            for marker_name, marker_indices in mesh.markers.items():
                offset_marker_indices = marker_indices + \
                    vertex_offsets[mesh_idx]

                # Get sizes and types for this marker
                sizes = mesh.marker_sizes.get(marker_name)
                cell_types = mesh.marker_cell_types.get(marker_name)

                if sizes is None:
                    # Skip markers without size info
                    continue

                MeshUtils.add_marker_to_dict(
                    combined_markers,
                    combined_marker_sizes,
                    combined_marker_cell_types,
                    marker_name,
                    offset_marker_indices,
                    sizes,
                    cell_types,
                )

        return combined_markers, combined_marker_sizes, combined_marker_cell_types

    @staticmethod
    def triangulate(mesh: Mesh) -> Mesh:
        """
        Convert a mesh to a pure triangle surface mesh.

        For polygon meshes (2D surface cells like triangles, quads, polygons):
            Uses fan triangulation: for each polygon with n vertices (n >= 3),
            creates (n-2) triangles by connecting the first vertex to all
            non-adjacent vertex pairs.

        For volume meshes (3D cells like hexahedra, tetrahedra, wedges, pyramids):
            Extracts the surface faces of each cell and triangulates them.
            This creates a surface mesh representation of the volume.

        Examples:
            - Triangle (3 vertices): 1 triangle
            - Quad (4 vertices): 2 triangles
            - Pentagon (5 vertices): 3 triangles
            - Tetrahedron (4 vertices, 4 faces): 4 triangles
            - Hexahedron (8 vertices, 6 quad faces): 12 triangles
            - Wedge (6 vertices, 5 faces): 8 triangles
            - Pyramid (5 vertices, 5 faces): 6 triangles

        Args:
            mesh: The mesh to triangulate (can have mixed cell types)

        Returns:
            A new mesh with all cells converted to triangles

        Raises:
            ValueError: If mesh has no indices or index_sizes, or contains
                        unsupported cell types

        Note:
            - Vertices are preserved unchanged
            - Markers are preserved unchanged
            - Only the cell structure (indices, index_sizes, cell_types) is modified
        """
        if mesh.indices is None or mesh.index_sizes is None:
            raise ValueError(
                "Mesh must have indices and index_sizes to triangulate")

        # Check if already all triangles
        if np.all(mesh.index_sizes == 3) and np.all(mesh.cell_types == VTKCellType.VTK_TRIANGLE):
            return mesh.copy()

        # Compute cell offsets once
        cell_offsets = np.concatenate(
            [[0], np.cumsum(mesh.index_sizes[:-1])]).astype(np.uint32)

        cell_types = mesh.cell_types
        index_sizes = mesh.index_sizes
        indices = mesh.indices
        vertices = mesh.vertices

        # Pre-check planarity for volume cells to reclassify them as polygons
        volume_types = set(
            TriangulationUtils._get_volume_cell_patterns().keys())
        effective_types = cell_types.copy()
        for i, (cell_type, size, offset) in enumerate(zip(cell_types, index_sizes, cell_offsets)):
            if cell_type in volume_types:
                cell_indices = indices[offset:offset + size]
                if TriangulationUtils.is_planar_cell(vertices, cell_indices):
                    effective_types[i] = VTKCellType.VTK_POLYGON

        result_chunks = []

        # Process triangles (already done, just copy)
        tri_mask = effective_types == VTKCellType.VTK_TRIANGLE
        if np.any(tri_mask):
            tri_offsets = cell_offsets[tri_mask]
            tri_sizes = index_sizes[tri_mask]
            for offset, size in zip(tri_offsets, tri_sizes):
                result_chunks.append(indices[offset:offset + size].copy())

        # Process all polygon types (quads, general polygons, reclassified planar cells)
        polygon_types = {VTKCellType.VTK_QUAD, VTKCellType.VTK_POLYGON}
        polygon_mask = np.isin(effective_types, list(polygon_types))
        if np.any(polygon_mask):
            poly_offsets = cell_offsets[polygon_mask]
            poly_sizes = index_sizes[polygon_mask]

            if np.any(poly_sizes < 3):
                invalid_idx = np.where(poly_sizes < 3)[0][0]
                raise ValueError(
                    f"Polygon with {poly_sizes[invalid_idx]} vertices cannot be triangulated (minimum 3 required)")

            # Group by size for efficient batch processing
            for size in np.unique(poly_sizes):
                size_mask = poly_sizes == size
                size_offsets = poly_offsets[size_mask]
                if len(size_offsets) > 0:
                    tris = TriangulationUtils.triangulate_polygons(
                        indices, size_offsets, size)
                    if len(tris) > 0:
                        result_chunks.append(tris)

        # Process volume cells using pattern-based triangulation
        volume_patterns = TriangulationUtils._get_volume_cell_patterns()
        for cell_type, (cell_size, tri_pattern) in volume_patterns.items():
            mask = effective_types == cell_type
            if np.any(mask):
                offsets = cell_offsets[mask]
                if len(offsets) > 0:
                    tris = TriangulationUtils.triangulate_uniform_cells(
                        indices, offsets, cell_size, tri_pattern)
                    if len(tris) > 0:
                        result_chunks.append(tris)

        # Check for unsupported types
        skip_types = {VTKCellType.VTK_VERTEX, VTKCellType.VTK_LINE}
        supported_types = {
            VTKCellType.VTK_TRIANGLE} | polygon_types | volume_types
        all_handled = supported_types | skip_types

        for i, ct in enumerate(effective_types):
            if ct not in all_handled:
                raise ValueError(f"Unsupported cell type {ct} at cell {i}")

        if not result_chunks:
            raise ValueError("No triangulatable cells found in mesh")

        triangulated_indices_flat = np.concatenate(result_chunks)
        num_triangles = len(triangulated_indices_flat) // 3

        return Mesh(
            vertices=mesh.vertices.copy(),
            indices=triangulated_indices_flat,
            index_sizes=np.full(num_triangles, 3, dtype=np.uint32),
            cell_types=np.full(
                num_triangles, VTKCellType.VTK_TRIANGLE, dtype=np.uint32),
            dim=mesh.dim,
            markers={name: data.copy() for name, data in mesh.markers.items()},
            marker_sizes={name: data.copy()
                          for name, data in mesh.marker_sizes.items()},
            marker_cell_types={name: data.copy()
                               for name, data in mesh.marker_cell_types.items()},
        )

    @staticmethod
    def optimize_vertex_cache(mesh: Mesh) -> Mesh:
        """
        Optimize the mesh for vertex cache efficiency.

        Args:
            mesh: The mesh to optimize

        Returns:
            A new optimized mesh, original mesh is unchanged
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        # Create a copy of the mesh
        result_mesh = mesh.copy()

        optimized_indices = np.zeros_like(result_mesh.indices)
        optimize_vertex_cache(
            optimized_indices, result_mesh.indices, result_mesh.index_count, result_mesh.vertex_count
        )

        result_mesh.indices = optimized_indices
        return result_mesh

    @staticmethod
    def optimize_overdraw(mesh: Mesh, threshold: float = 1.05) -> Mesh:
        """
        Optimize the mesh for overdraw.

        Args:
            mesh: The mesh to optimize
            threshold: threshold for optimization (default: 1.05)

        Returns:
            A new optimized mesh, original mesh is unchanged
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        # Create a copy of the mesh
        result_mesh = mesh.copy()

        optimized_indices = np.zeros_like(result_mesh.indices)
        optimize_overdraw(
            optimized_indices,
            result_mesh.indices,
            result_mesh.vertices,
            result_mesh.index_count,
            result_mesh.vertex_count,
            result_mesh.vertices.itemsize * result_mesh.vertices.shape[1],
            threshold,
        )

        result_mesh.indices = optimized_indices
        return result_mesh

    @staticmethod
    def optimize_vertex_fetch(mesh: Mesh) -> Mesh:
        """
        Optimize the mesh for vertex fetch efficiency.

        Args:
            mesh: The mesh to optimize

        Returns:
            A new optimized mesh, original mesh is unchanged
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        # Create a copy of the mesh
        result_mesh = mesh.copy()

        optimized_vertices = np.zeros_like(result_mesh.vertices)
        unique_vertex_count = optimize_vertex_fetch(
            optimized_vertices,
            result_mesh.indices,
            result_mesh.vertices,
            result_mesh.index_count,
            result_mesh.vertex_count,
            result_mesh.vertices.itemsize * result_mesh.vertices.shape[1],
        )

        result_mesh.vertices = optimized_vertices[:unique_vertex_count]
        # No need to update vertex_count as it's calculated on-the-fly
        return result_mesh

    @staticmethod
    def simplify(
        mesh: Mesh,
        target_ratio: float = 0.25,
        target_error: float = 0.01,
        options: int = 0,
    ) -> Mesh:
        """
        Simplify the mesh.

        Args:
            mesh: The mesh to simplify
            target_ratio: ratio of triangles to keep (default: 0.25)
            target_error: target error (default: 0.01)
            options: simplification options (default: 0)

        Returns:
            A new simplified mesh, original mesh is unchanged
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to simplify")

        # Create a copy of the mesh
        result_mesh = mesh.copy()

        target_index_count = int(result_mesh.index_count * target_ratio)
        simplified_indices = np.zeros(result_mesh.index_count, dtype=np.uint32)

        result_error = np.array([0.0], dtype=np.float32)
        new_index_count = simplify(
            simplified_indices,
            result_mesh.indices,
            result_mesh.vertices,
            result_mesh.index_count,
            result_mesh.vertex_count,
            result_mesh.vertices.itemsize * result_mesh.vertices.shape[1],
            target_index_count,
            target_error,
            options,
            result_error,
        )

        result_mesh.indices = simplified_indices[:new_index_count]
        # No need to update index_count as it's calculated on-the-fly
        return result_mesh

    @staticmethod
    def encode(mesh: Mesh):
        """
        Encode the mesh and all numpy array fields for efficient transmission.

        Args:
            mesh: The mesh to encode

        Returns:
            EncodedMesh object with encoded vertices, indices, and arrays
        """
        # Encode vertex buffer
        encoded_vertices = encode_vertex_buffer(
            mesh.vertices,
            mesh.vertex_count,
            mesh.vertices.itemsize * mesh.vertices.shape[1],
        )

        # Encode index buffer if present
        encoded_indices = None
        if mesh.indices is not None:
            encoded_indices = encode_index_sequence(
                mesh.indices, mesh.index_count, mesh.vertex_count
            )

        # Note: index_sizes will be encoded as a regular array in the arrays dict
        encoded_index_sizes = None

        # Encode additional array fields, including nested arrays from dictionaries
        encoded_arrays = {}
        for field_name in mesh.array_fields:
            if field_name in ("vertices", "indices"):
                continue  # Skip the main vertices and indices

            # Handle nested array paths (e.g., "textures.diffuse")
            if "." in field_name:
                # Extract the nested array
                parts = field_name.split(".")
                obj = mesh
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
                    array = getattr(mesh, field_name)
                    if isinstance(array, np.ndarray) or (HAS_JAX and isinstance(array, jnp.ndarray)):
                        encoded_arrays[field_name] = ArrayUtils.encode_array(
                            array)
                except AttributeError:
                    # Skip attributes that don't exist
                    pass

        # Handle index_sizes as a special case - store in arrays if present
        if mesh.index_sizes is not None:
            encoded_arrays["index_sizes"] = ArrayUtils.encode_array(
                mesh.index_sizes)

        # Handle cell_types as a special case - store in arrays if present
        if mesh.cell_types is not None:
            encoded_arrays["cell_types"] = ArrayUtils.encode_array(
                mesh.cell_types)

        # Create encoded mesh
        return EncodedMesh(
            vertices=encoded_vertices,
            indices=encoded_indices,
            vertex_count=mesh.vertex_count,
            vertex_size=mesh.vertices.itemsize * mesh.vertices.shape[1],
            index_count=mesh.index_count if mesh.indices is not None else None,
            index_size=mesh.indices.itemsize if mesh.indices is not None else 4,
            index_sizes=encoded_index_sizes,
            arrays=encoded_arrays,
        )

    @staticmethod
    def save_to_zip(mesh: Mesh, source: Union[PathLike, BytesIO], date_time: Optional[tuple] = None) -> None:
        """
        Save the mesh to a zip file.

        Args:
            mesh: The mesh to save
            source: Path to the output zip file
        """
        encoded_mesh = MeshUtils.encode(mesh)

        # Add model fields that aren't numpy arrays, preserving non-array values in dicts
        model_data = {}

        def extract_non_arrays(obj: Any, prefix: str = "") -> Any:
            """Recursively extract non-array values from nested structures."""
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    nested_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, np.ndarray) or (HAS_JAX and isinstance(value, jnp.ndarray)):
                        # Skip arrays - they're stored separately
                        continue
                    elif isinstance(value, dict):
                        # Recursively process nested dicts
                        nested_result = extract_non_arrays(value, nested_key)
                        if nested_result:  # Only include non-empty dicts
                            result[key] = nested_result
                    else:
                        # Include non-array values
                        result[key] = value
                return result if result else None
            elif isinstance(obj, np.ndarray) or (HAS_JAX and isinstance(obj, jnp.ndarray)):
                # Skip arrays
                return None
            else:
                # Include non-array values
                return obj

        for field_name, field_value in mesh.model_dump().items():
            # Skip direct array fields (they're stored separately)
            if field_name in mesh.array_fields:
                continue

            if isinstance(field_value, dict):
                # Extract non-array values from dictionaries
                non_array_content = extract_non_arrays(field_value, field_name)
                if non_array_content:
                    model_data[field_name] = non_array_content
            else:
                # Include scalar fields directly
                model_data[field_name] = field_value

        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            # Create mesh size metadata
            mesh_size = MeshSize(
                vertex_count=encoded_mesh.vertex_count,
                vertex_size=encoded_mesh.vertex_size,
                index_count=encoded_mesh.index_count,
                index_size=encoded_mesh.index_size,
            )

            metadata = MeshMetadata(
                class_name=mesh.__class__.__name__,
                module_name=mesh.__class__.__module__,
                mesh_size=mesh_size,
                field_data=model_data,
            )

            # Collect all files to write in sorted order for deterministic output
            files_to_write = []

            # Add mesh data
            files_to_write.append(("mesh/vertices.bin", encoded_mesh.vertices))
            if encoded_mesh.indices is not None:
                files_to_write.append(
                    ("mesh/indices.bin", encoded_mesh.indices))

            # Add array data (sorted by name for deterministic order)
            for name in sorted(encoded_mesh.arrays.keys()):
                encoded_array = encoded_mesh.arrays[name]
                # Convert dots to nested directory structure, each array gets its own directory
                array_path = name.replace(".", "/")
                files_to_write.append(
                    (f"arrays/{array_path}/array.bin", encoded_array.data))

                # Save array metadata
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

            # Add general metadata
            files_to_write.append(("metadata.json", json.dumps(
                metadata.model_dump(), indent=2, sort_keys=True)))

            # Sort files by path for deterministic order and write them
            for filename, data in sorted(files_to_write):
                if date_time is not None:
                    info = zipfile.ZipInfo(
                        filename=filename, date_time=date_time)
                else:
                    info = zipfile.ZipInfo(filename=filename)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16  # Fixed file permissions
                if isinstance(data, str):
                    data = data.encode('utf-8')
                zipf.writestr(info, data)

    @staticmethod
    def load_from_zip(cls: Type[T], destination: Union[PathLike, BytesIO], use_jax: bool = False) -> T:
        """
        Load a mesh from a zip file.

        Args:
            cls: The mesh class to instantiate
            destination: Path to the input zip file
            use_jax: If True and JAX is available, decode arrays as JAX arrays instead of numpy arrays

        Returns:
            Mesh object loaded from the zip file
        """
        with zipfile.ZipFile(destination, "r") as zipf:
            # Load general metadata
            with zipf.open("metadata.json") as f:
                metadata_dict = json.loads(f.read().decode("utf-8"))
                metadata = MeshMetadata(**metadata_dict)

            # Check if the class matches
            class_name = metadata.class_name
            module_name = metadata.module_name

            # If the class doesn't match, try to import it
            if class_name != cls.__name__ or module_name != cls.__module__:
                raise ValueError(
                    f"Class mismatch: expected {cls.__name__} but got {class_name} from {module_name}"
                )
            else:
                target_cls = cls

            # Get mesh size metadata from the file metadata
            mesh_size = metadata.mesh_size

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
                vertex_count=mesh_size.vertex_count,
                vertex_size=mesh_size.vertex_size,
                index_count=mesh_size.index_count,
                index_size=mesh_size.index_size,
                index_sizes=None,  # Not used anymore
                arrays={},  # Will be populated below
            )

            # Load additional array data
            for array_file_name in [
                file_name
                for file_name in zipf.namelist()
                if file_name.startswith("arrays/") and file_name.endswith("/array.bin")
            ]:
                # Extract the array path by removing "arrays/" prefix and "/array.bin" suffix
                # Remove "arrays/" and "/array.bin"
                array_path = array_file_name[7:-10]

                # Load array metadata
                with zipf.open(f"arrays/{array_path}/metadata.json") as f:
                    array_metadata_dict = json.loads(f.read().decode("utf-8"))
                    array_metadata = ArrayMetadata(**array_metadata_dict)

                # Load array data
                with zipf.open(array_file_name) as f:
                    encoded_data = f.read()

                # Create encoded array
                encoded_array = EncodedArray(
                    data=encoded_data,
                    shape=tuple(array_metadata.shape),
                    dtype=np.dtype(array_metadata.dtype),
                    itemsize=array_metadata.itemsize,
                )

                # Convert directory path back to dotted name
                original_name = array_path.replace("/", ".")

                # Add to encoded mesh arrays with original name
                encoded_mesh.arrays[original_name] = encoded_array

            # Decode the mesh using MeshUtils.decode
            decoded_mesh = MeshUtils.decode(
                target_cls, encoded_mesh, use_jax=use_jax)

            # Add any additional field data from the metadata, merging with existing dict fields
            if metadata.field_data:
                for field_name, field_value in metadata.field_data.items():
                    existing_value = getattr(decoded_mesh, field_name, None)
                    if existing_value is not None and isinstance(existing_value, dict) and isinstance(field_value, dict):
                        # Merge non-array values into existing dictionary structure
                        def merge_dicts(target: dict, source: dict) -> None:
                            """Recursively merge source dict into target dict."""
                            for key, value in source.items():
                                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                                    merge_dicts(target[key], value)
                                else:
                                    target[key] = value

                        merge_dicts(existing_value, field_value)
                    else:
                        # Set scalar fields directly
                        setattr(decoded_mesh, field_name, field_value)

            return decoded_mesh

    @staticmethod
    def decode(cls: Type[T], encoded_mesh: EncodedMesh, use_jax: bool = False) -> T:
        """
        Decode an encoded mesh.

        Args:
            cls: The mesh class to instantiate
            encoded_mesh: EncodedMesh object to decode
            use_jax: If True and JAX is available, decode arrays as JAX arrays instead of numpy arrays

        Returns:
            Decoded Mesh object
        """
        if use_jax and not HAS_JAX:
            raise ValueError(
                "JAX is not available. Install JAX to use JAX arrays.")
        # Decode vertex buffer
        vertices = decode_vertex_buffer(
            encoded_mesh.vertex_count, encoded_mesh.vertex_size, encoded_mesh.vertices
        )
        if use_jax:
            vertices = jnp.array(vertices)

        # Decode index buffer if present
        indices = None
        if encoded_mesh.indices is not None and encoded_mesh.index_count is not None:
            indices = decode_index_sequence(
                encoded_mesh.index_count, encoded_mesh.index_size, encoded_mesh.indices
            )
            if use_jax:
                indices = jnp.array(indices)

        # Create the base mesh object
        mesh_args = {
            "vertices": vertices,
            "indices": indices,
        }

        # Decode index_sizes and cell_types if present (stored in arrays dict)
        index_sizes = None
        cell_types = None

        # Decode additional arrays if present
        dict_fields = {}  # Will store reconstructed dictionary fields

        if encoded_mesh.arrays:
            for name, encoded_array in encoded_mesh.arrays.items():
                # Decode the array
                decoded_array = ArrayUtils.decode_array(encoded_array)

                # Convert to JAX array if requested
                if use_jax and HAS_JAX:
                    decoded_array = jnp.array(decoded_array)

                # Check if this is a nested array (contains dots)
                if name == "index_sizes":
                    # Special handling for index_sizes
                    index_sizes = decoded_array
                elif name == "cell_types":
                    # Special handling for cell_types
                    cell_types = decoded_array
                elif "." in name:
                    # This is a nested array from a dictionary field
                    parts = name.split(".")
                    field_name = parts[0]

                    # Initialize the dictionary field if not exists
                    if field_name not in dict_fields:
                        dict_fields[field_name] = {}

                    # Navigate/create the nested structure
                    current_dict = dict_fields[field_name]
                    for part in parts[1:-1]:
                        if part not in current_dict:
                            current_dict[part] = {}
                        current_dict = current_dict[part]

                    # Set the final array
                    current_dict[parts[-1]] = decoded_array
                else:
                    # This is a direct array field
                    mesh_args[name] = decoded_array

        # Add index_sizes and cell_types to mesh args
        mesh_args["index_sizes"] = index_sizes
        mesh_args["cell_types"] = cell_types

        # Add reconstructed dictionary fields to mesh arguments
        mesh_args.update(dict_fields)

        # Create and return the mesh object
        return cls(**mesh_args)

    @staticmethod
    def to_numpy(mesh: Mesh) -> Mesh:
        """
        Create a new mesh with all arrays converted to NumPy arrays.

        This method creates a new mesh instance where all arrays (including
        nested arrays in dictionaries and custom fields) are converted to
        NumPy arrays. The original mesh is unchanged.

        Args:
            mesh: The mesh to convert

        Returns:
            A new mesh with all arrays as NumPy arrays

        Example:
            >>> jax_mesh = Mesh(vertices=jnp.array([[0,0,0]]), indices=jnp.array([0]))
            >>> numpy_mesh = MeshUtils.to_numpy(jax_mesh)
            >>> isinstance(numpy_mesh.vertices, np.ndarray)  # True
        """
        if not HAS_JAX:
            # If JAX isn't available, just return a copy
            return mesh.copy()

        mesh_copy = mesh.copy()

        def convert_to_numpy(obj: Any) -> Any:
            """Recursively convert arrays to NumPy."""
            if isinstance(obj, jnp.ndarray):
                return np.array(obj)
            elif isinstance(obj, np.ndarray):
                return obj  # Already NumPy
            elif isinstance(obj, dict):
                return {key: convert_to_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert_to_numpy(item) for item in obj)
            else:
                return obj

        # Convert all fields
        for field_name in mesh_copy.model_fields_set:
            try:
                value = getattr(mesh_copy, field_name)
                if value is not None:
                    converted_value = convert_to_numpy(value)
                    setattr(mesh_copy, field_name, converted_value)
            except AttributeError:
                pass

        return mesh_copy

    @staticmethod
    def to_jax(mesh: Mesh) -> Mesh:
        """
        Create a new mesh with all arrays converted to JAX arrays.

        This method creates a new mesh instance where all arrays (including
        nested arrays in dictionaries and custom fields) are converted to
        JAX arrays. The original mesh is unchanged.

        Args:
            mesh: The mesh to convert

        Returns:
            A new mesh with all arrays as JAX arrays

        Raises:
            ValueError: If JAX is not available

        Example:
            >>> numpy_mesh = Mesh(vertices=np.array([[0,0,0]]), indices=np.array([0]))
            >>> jax_mesh = MeshUtils.to_jax(numpy_mesh)
            >>> hasattr(jax_mesh.vertices, 'device')  # True
        """
        if not HAS_JAX:
            raise ValueError(
                "JAX is not available. Install JAX to convert to JAX arrays.")

        mesh_copy = mesh.copy()

        def convert_to_jax(obj: Any) -> Any:
            """Recursively convert arrays to JAX."""
            if isinstance(obj, np.ndarray):
                return jnp.array(obj)
            elif HAS_JAX and isinstance(obj, jnp.ndarray):
                return obj  # Already JAX
            elif isinstance(obj, dict):
                return {key: convert_to_jax(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert_to_jax(item) for item in obj)
            else:
                return obj

        # Convert all fields
        for field_name in mesh_copy.model_fields_set:
            try:
                value = getattr(mesh_copy, field_name)
                if value is not None:
                    converted_value = convert_to_jax(value)
                    setattr(mesh_copy, field_name, converted_value)
            except AttributeError:
                pass

        return mesh_copy
