"""
High-level mesh abstraction for easier use of meshoptimizer.

This module provides:
1. Mesh class inheriting from Packable for 3D mesh representation
2. Specialized mesh encoding using meshoptimizer for vertices/indices
3. Mesh operations via MeshUtils utility class

The Mesh class inherits from Packable and adds:
- Specialized meshoptimizer encoding for vertices and indices via Annotated types
- Mesh operations: triangulate, optimize, simplify
- Marker support for boundary conditions and regions
"""

from meshly.array import Array, ArrayUtils, IndexSequence, VertexBuffer
from meshly.cell_types import CellTypeUtils, VTKCellType
from meshly.packable import Packable
from meshly.utils import ElementUtils, MeshUtils, TriangulationUtils
from meshoptimizer import (
    optimize_vertex_cache as meshopt_optimize_vertex_cache,
    optimize_overdraw as meshopt_optimize_overdraw,
    optimize_vertex_fetch as meshopt_optimize_vertex_fetch,
    simplify as meshopt_simplify,
)
from pathlib import Path
from typing import (
    ClassVar,
    Dict,
    Optional,
    Any,
    TypeVar,
    List,
    Sequence,
    Union,
)
import numpy as np
from pydantic import Field, model_validator
from pydantic.json_schema import JsonSchemaValue, GetJsonSchemaHandler
from pydantic_core import core_schema as pydantic_core_schema


# Type variable for the Mesh class
TMesh = TypeVar("T", bound="Mesh")


class Mesh(Packable):
    """
    A Pydantic base class representing a 3D mesh.

    Users can inherit from this class to define custom mesh types with additional
    numpy array attributes that will be automatically encoded/decoded.

    Inherits from Packable for automatic array serialization. Mesh uses
    specialized meshoptimizer encoding for vertices and indices via Annotated
    types (VertexBuffer, IndexSequence).
    """

    is_contained: ClassVar[bool] = True
    """Mesh extracts as a single zip blob when nested in other Packables."""

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema_obj: pydantic_core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Inject x-base hint into JSON schema indicating this is a Mesh."""
        json_schema = handler(core_schema_obj)
        json_schema = handler.resolve_ref_schema(json_schema)
        json_schema['x-base'] = 'Mesh'
        return json_schema

    # ============================================================
    # Field definitions with encoding specified via Annotated types
    # ============================================================

    vertices: VertexBuffer = Field(
        ...,
        description="Vertex data as a numpy or JAX array",
    )
    indices: Optional[Union[IndexSequence, List[Any]]] = Field(
        None,
        description="Index data as a flattened 1D numpy/JAX array",
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

        Handles various input formats for indices and automatically infers
        index_sizes and cell_types when not explicitly provided.
        """
        # Helper to convert arrays to numpy (needed for meshoptimizer)
        def to_numpy(arr):
            return ArrayUtils.convert_array(arr, "numpy") if ArrayUtils.is_array(arr) else arr

        # Ensure vertices is float32, preserving JAX type if present
        if self.vertices is not None:
            vertex_type = ArrayUtils.detect_array_type(self.vertices)
            self.vertices = ArrayUtils.convert_array(
                np.asarray(self.vertices, dtype=np.float32), vertex_type)

        # Process indices through ElementUtils
        if self.indices is not None:
            self.indices, self.index_sizes, self.cell_types = ElementUtils.convert_array_input(
                to_numpy(self.indices), to_numpy(
                    self.index_sizes), to_numpy(self.cell_types)
            )

        # Auto-compute dimension from cell types
        if self.dim is None:
            self.dim = CellTypeUtils.get_mesh_dimension(self.cell_types) \
                if self.cell_types is not None and len(self.cell_types) > 0 else 3

        # Convert markers to flattened arrays
        if self.markers:
            converted_markers = {}
            for name, data in self.markers.items():
                data = to_numpy(data)
                if isinstance(data, np.ndarray):
                    converted_markers[name] = data.astype(np.uint32)
                    # Auto-calculate sizes from cell_types if missing
                    if name in self.marker_cell_types and name not in self.marker_sizes:
                        self.marker_sizes[name] = CellTypeUtils.infer_sizes_from_vtk_cell_types(
                            self.marker_cell_types[name])
                    if name not in self.marker_sizes or name not in self.marker_cell_types:
                        raise ValueError(
                            f"Marker '{name}' missing marker_sizes or marker_cell_types")
                else:
                    # Convert list of lists to flattened structure
                    indices, sizes, types = ElementUtils.convert_list_to_flattened(
                        [list(el) for el in data])
                    converted_markers[name] = indices
                    self.marker_sizes[name] = sizes
                    self.marker_cell_types[name] = types
            self.markers = converted_markers

        return self

    @classmethod
    def combine(
        cls: type[TMesh],
        meshes: List[TMesh],
        marker_names: Optional[List[str]] = None,
        preserve_markers: bool = True,
    ) -> TMesh:
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

        # Collect extra fields from subclass (take from first mesh)
        extra_fields = {
            name: getattr(meshes[0], name)
            for name in cls.model_fields.keys() - Mesh.model_fields.keys()
        }

        # Create combined mesh
        return cls(
            vertices=combined_vertices,
            indices=combined_indices,
            index_sizes=combined_index_sizes,
            cell_types=combined_cell_types_array,
            dim=dim,
            markers=combined_markers,
            marker_sizes=combined_marker_sizes,
            marker_cell_types=combined_marker_cell_types,
            **extra_fields,
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
            return self.model_copy(deep=True)

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

        result_mesh = self.model_copy(deep=True)
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

        result_mesh = self.model_copy(deep=True)
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

        result_mesh = self.model_copy(deep=True)
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

        result_mesh = self.model_copy(deep=True)
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

    def to_pyvista(self):
        """Convert mesh to a PyVista UnstructuredGrid.
        
        Requires pyvista to be installed (available in dev dependencies).
        
        Returns:
            pv.UnstructuredGrid: PyVista mesh object
            
        Raises:
            ImportError: If pyvista is not installed
            
        Example:
            >>> mesh = Mesh(vertices=vertices, indices=indices)
            >>> pv_mesh = mesh.to_pyvista()
            >>> pv_mesh.plot()
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError(
                "pyvista is required for VTK export. "
                "Install with: pip install meshly[dev] or pip install pyvista"
            )
        
        if self.indices is None or self.index_sizes is None or self.cell_types is None:
            raise ValueError("Mesh must have indices, index_sizes, and cell_types for VTK export")
        
        # Build VTK cell array: [size0, idx0, idx1, ..., size1, idx0, idx1, ...]
        cells = []
        offset = 0
        for size in self.index_sizes:
            cells.append(size)
            cells.extend(self.indices[offset:offset + size])
            offset += size
        
        return pv.UnstructuredGrid(
            np.array(cells, dtype=np.int64),
            np.array(self.cell_types, dtype=np.uint8),
            np.asarray(self.vertices, dtype=np.float64),
        )

    def save_vtk(self, path: Union[str, Path]) -> None:
        """Save mesh to a VTK file.
        
        Requires pyvista to be installed (available in dev dependencies).
        Supports .vtk, .vtu, .ply, .stl and other formats supported by PyVista.
        
        Args:
            path: Output file path. Format determined by extension.
            
        Example:
            >>> mesh.save_vtk("output.vtu")
            >>> mesh.save_vtk("output.stl")  # For triangle meshes
        """
        pv_mesh = self.to_pyvista()
        pv_mesh.save(str(path))
