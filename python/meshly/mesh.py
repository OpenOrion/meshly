"""
High-level mesh abstraction for easier use of meshoptimizer.

This module provides:
1. Mesh class inheriting from Packable for 3D mesh representation
2. Specialized mesh encoding using meshoptimizer for vertices/indices
3. Mesh operations via MeshUtils utility class

The Mesh class inherits from Packable and adds:
- Specialized meshoptimizer encoding for vertices and indices
- Mesh operations: triangulate, optimize, simplify
- Marker support for boundary conditions and regions
"""

import json
import zipfile
from io import BytesIO
from typing import (
    Dict,
    Optional,
    Type,
    Any,
    TypeVar,
    Union,
    List,
    Sequence,
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
    encode_vertex_buffer,
    encode_index_sequence,
    decode_vertex_buffer,
    decode_index_sequence,
    optimize_vertex_cache as meshopt_optimize_vertex_cache,
    optimize_overdraw as meshopt_optimize_overdraw,
    optimize_vertex_fetch as meshopt_optimize_vertex_fetch,
    simplify as meshopt_simplify,
)

from .packable import Packable, PackableMetadata
from .array import ArrayMetadata, EncodedArray
from .common import PathLike
from .cell_types import CellTypeUtils, VTKCellType
from .utils import ElementUtils, TriangulationUtils, MeshUtils, ZipUtils

# Type variable for the Mesh class
T = TypeVar("T", bound="Mesh")


class EncodedMesh(Packable):
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


class MeshSizeInfo(BaseModel):
    """Mesh size information for meshoptimizer encoding/decoding."""
    vertex_count: int = Field(..., description="Number of vertices")
    vertex_size: int = Field(..., description="Size of each vertex in bytes")
    index_count: Optional[int] = Field(None, description="Number of indices")
    index_size: int = Field(default=4, description="Size of each index in bytes")


class MeshMetadata(PackableMetadata):
    """Metadata for a Mesh saved to zip, extending PackableMetadata with mesh-specific info."""
    mesh_size: MeshSizeInfo = Field(..., description="Mesh size information for decoding")


class Mesh(Packable):
    """
    A Pydantic base class representing a 3D mesh.

    Users can inherit from this class to define custom mesh types with additional
    numpy array attributes that will be automatically encoded/decoded.

    Inherits from Packable for automatic array serialization. Mesh adds
    specialized handling for vertices and indices using meshoptimizer encoding.
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

    # Mesh operations
    def triangulate(self) -> "Mesh":
        """
        Convert mesh to pure triangle surface mesh.

        For polygon meshes (2D surface cells like triangles, quads, polygons):
            Uses fan triangulation: for each polygon with n vertices (n >= 3),
            creates (n-2) triangles by connecting the first vertex to all
            non-adjacent vertex pairs.

        For volume meshes (3D cells like hexahedra, tetrahedra, wedges, pyramids):
            Extracts the surface faces of each cell and triangulates them.

        Returns:
            A new mesh with all cells converted to triangles
        """
        if self.indices is None or self.index_sizes is None:
            raise ValueError(
                "Mesh must have indices and index_sizes to triangulate")

        # Check if already all triangles
        if np.all(self.index_sizes == 3) and np.all(self.cell_types == VTKCellType.VTK_TRIANGLE):
            return self.copy()

        # Compute cell offsets once
        cell_offsets = np.concatenate(
            [[0], np.cumsum(self.index_sizes[:-1])]).astype(np.uint32)

        # Pre-check planarity for volume cells to reclassify them as polygons
        volume_types = set(
            TriangulationUtils._get_volume_cell_patterns().keys())
        effective_types = self.cell_types.copy()
        for i, (cell_type, size, offset) in enumerate(zip(self.cell_types, self.index_sizes, cell_offsets)):
            if cell_type in volume_types:
                cell_indices = self.indices[offset:offset + size]
                if TriangulationUtils.is_planar_cell(self.vertices, cell_indices):
                    effective_types[i] = VTKCellType.VTK_POLYGON

        result_chunks = []

        # Process triangles (already done, just copy)
        tri_mask = effective_types == VTKCellType.VTK_TRIANGLE
        if np.any(tri_mask):
            tri_offsets = cell_offsets[tri_mask]
            tri_sizes = self.index_sizes[tri_mask]
            for offset, size in zip(tri_offsets, tri_sizes):
                result_chunks.append(self.indices[offset:offset + size].copy())

        # Process all polygon types
        polygon_types = {VTKCellType.VTK_QUAD, VTKCellType.VTK_POLYGON}
        polygon_mask = np.isin(effective_types, list(polygon_types))
        if np.any(polygon_mask):
            poly_offsets = cell_offsets[polygon_mask]
            poly_sizes = self.index_sizes[polygon_mask]

            if np.any(poly_sizes < 3):
                invalid_idx = np.where(poly_sizes < 3)[0][0]
                raise ValueError(
                    f"Polygon with {poly_sizes[invalid_idx]} vertices cannot be triangulated")

            for size in np.unique(poly_sizes):
                size_mask = poly_sizes == size
                size_offsets = poly_offsets[size_mask]
                if len(size_offsets) > 0:
                    tris = TriangulationUtils.triangulate_polygons(
                        self.indices, size_offsets, size)
                    if len(tris) > 0:
                        result_chunks.append(tris)

        # Process volume cells
        volume_patterns = TriangulationUtils._get_volume_cell_patterns()
        for cell_type, (cell_size, tri_pattern) in volume_patterns.items():
            mask = effective_types == cell_type
            if np.any(mask):
                offsets = cell_offsets[mask]
                if len(offsets) > 0:
                    tris = TriangulationUtils.triangulate_uniform_cells(
                        self.indices, offsets, cell_size, tri_pattern)
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
            vertices=self.vertices.copy(),
            indices=triangulated_indices_flat,
            index_sizes=np.full(num_triangles, 3, dtype=np.uint32),
            cell_types=np.full(
                num_triangles, VTKCellType.VTK_TRIANGLE, dtype=np.uint32),
            dim=self.dim,
            markers={name: data.copy() for name, data in self.markers.items()},
            marker_sizes={name: data.copy()
                          for name, data in self.marker_sizes.items()},
            marker_cell_types={name: data.copy()
                               for name, data in self.marker_cell_types.items()},
        )

    def optimize_vertex_cache(self) -> "Mesh":
        """Optimize mesh for vertex cache efficiency."""
        if self.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        result_mesh = self.copy()
        optimized_indices = np.zeros_like(result_mesh.indices)
        meshopt_optimize_vertex_cache(
            optimized_indices, result_mesh.indices, result_mesh.index_count, result_mesh.vertex_count
        )
        result_mesh.indices = optimized_indices
        return result_mesh

    def optimize_overdraw(self, threshold: float = 1.05) -> "Mesh":
        """Optimize mesh to reduce overdraw."""
        if self.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        result_mesh = self.copy()
        optimized_indices = np.zeros_like(result_mesh.indices)
        meshopt_optimize_overdraw(
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

    def optimize_vertex_fetch(self) -> "Mesh":
        """Optimize mesh for vertex fetch efficiency."""
        if self.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        result_mesh = self.copy()
        optimized_vertices = np.zeros_like(result_mesh.vertices)
        unique_vertex_count = meshopt_optimize_vertex_fetch(
            optimized_vertices,
            result_mesh.indices,
            result_mesh.vertices,
            result_mesh.index_count,
            result_mesh.vertex_count,
            result_mesh.vertices.itemsize * result_mesh.vertices.shape[1],
        )
        result_mesh.vertices = optimized_vertices[:unique_vertex_count]
        return result_mesh

    def simplify(self, target_ratio: float = 0.25, target_error: float = 0.01, options: int = 0) -> "Mesh":
        """Simplify mesh to reduce complexity.

        Note: Simplification only works on triangle meshes. The simplified result
        will have updated index_sizes and cell_types for the new triangle count.
        """
        if self.indices is None:
            raise ValueError("Mesh has no indices to simplify")

        result_mesh = self.copy()
        target_index_count = int(result_mesh.index_count * target_ratio)
        simplified_indices = np.zeros(result_mesh.index_count, dtype=np.uint32)

        result_error = np.array([0.0], dtype=np.float32)
        new_index_count = meshopt_simplify(
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

        # Update index_sizes and cell_types for new triangle count
        num_triangles = new_index_count // 3
        result_mesh.index_sizes = np.full(num_triangles, 3, dtype=np.uint32)
        result_mesh.cell_types = np.full(
            num_triangles, VTKCellType.VTK_TRIANGLE, dtype=np.uint8)

        return result_mesh

    def _create_metadata(self, field_data: Dict[str, Any]) -> MeshMetadata:
        """Create MeshMetadata with mesh size info for meshoptimizer decoding."""
        mesh_size = MeshSizeInfo(
            vertex_count=self.vertex_count,
            vertex_size=self.vertices.itemsize * self.vertices.shape[1],
            index_count=self.index_count if self.indices is not None else None,
            index_size=self.indices.itemsize if self.indices is not None else 4,
        )
        return MeshMetadata(
            class_name=self.__class__.__name__,
            module_name=self.__class__.__module__,
            field_data=field_data,
            mesh_size=mesh_size,
        )

    # Override save_to_zip for mesh-specific encoding (vertices/indices use meshoptimizer)
    def save_to_zip(
        self,
        destination: Union[PathLike, BytesIO],
        date_time: Optional[tuple] = None
    ) -> None:
        """Save mesh to a zip file with meshoptimizer compression for vertices/indices."""
        # Encode vertices/indices using meshoptimizer
        encoded_vertices = encode_vertex_buffer(
            self.vertices,
            self.vertex_count,
            self.vertices.itemsize * self.vertices.shape[1],
        )

        encoded_indices = None
        if self.indices is not None:
            encoded_indices = encode_index_sequence(
                self.indices, self.index_count, self.vertex_count
            )

        # Use Packable helpers for the rest
        encoded_data = self.encode()
        field_data = self._extract_non_array_fields()

        # Prepare files using parent helper, excluding vertices/indices (handled specially)
        files_to_write = self._prepare_zip_files(
            encoded_data, field_data,
            exclude_arrays={"vertices", "indices"}
        )

        # Add mesh-specific files
        files_to_write.append(("mesh/vertices.bin", encoded_vertices))
        if encoded_indices is not None:
            files_to_write.append(("mesh/indices.bin", encoded_indices))

        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            ZipUtils.write_files(zipf, files_to_write, date_time)

    @classmethod
    def load_from_zip(cls: Type[T], source: Union[PathLike, BytesIO], use_jax: bool = False) -> T:
        """Load mesh from a zip file."""
        if use_jax and not HAS_JAX:
            raise ValueError(
                "JAX is not available. Install JAX to use JAX arrays.")

        with zipfile.ZipFile(source, "r") as zipf:
            metadata = cls.load_metadata(zipf, MeshMetadata)

            # Get mesh size info from typed metadata
            mesh_size = metadata.mesh_size

            # Decode vertices using meshoptimizer
            with zipf.open("mesh/vertices.bin") as f:
                encoded_vertices = f.read()
            vertices = decode_vertex_buffer(
                mesh_size.vertex_count, mesh_size.vertex_size, encoded_vertices
            )
            if use_jax:
                vertices = jnp.array(vertices)

            # Decode indices using meshoptimizer
            indices = None
            if "mesh/indices.bin" in zipf.namelist() and mesh_size.index_count:
                with zipf.open("mesh/indices.bin") as f:
                    encoded_indices = f.read()
                indices = decode_index_sequence(
                    mesh_size.index_count, mesh_size.index_size, encoded_indices
                )
                if use_jax:
                    indices = jnp.array(indices)

            # Load and decode other arrays
            data = ZipUtils.load_arrays(zipf, use_jax)

            # Build mesh args
            mesh_data = {"vertices": vertices, "indices": indices}
            mesh_data.update(data)

            # Merge non-array field values from metadata
            ZipUtils.merge_field_data(mesh_data, metadata.field_data)

            return cls(**mesh_data)
