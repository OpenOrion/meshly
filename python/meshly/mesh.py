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

Design principles:
- Fields use canonical types only (numpy arrays)
- Factory classmethods handle flexible input formats
- No complex validation/conversion in __init__
"""

from meshly.array import Array, ArrayUtils, IndexSequence
from meshly.cell_types import CellTypeUtils, VTKCellType
from meshly.packable import Packable
from meshly.utils import ElementUtils, MeshUtils, TriangulationUtils, ElementData

import meshoptimizer
from pathlib import Path
from typing import (
    ClassVar,
    Dict,
    Optional,
    TypeVar,
    List,
    Sequence,
    Union,
    overload,
)
import numpy as np
from pydantic import Field
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema as pydantic_core_schema
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import GetJsonSchemaHandler


TMesh = TypeVar("TMesh", bound="Mesh")


class Mesh(Packable):
    """
    A Pydantic base class representing a 3D mesh.

    Users can inherit from this class to define custom mesh types with additional
    numpy array attributes that will be automatically encoded/decoded.

    Inherits from Packable for automatic array serialization. Mesh uses
    specialized meshoptimizer encoding for vertices and indices via Annotated
    types (Array, IndexSequence).
    
    For flexible input formats, use factory methods:
    - Mesh.from_polygons() - create from list of polygon vertex indices
    - Mesh.from_triangles() - create from Nx3 triangle array
    """

    is_contained: ClassVar[bool] = True
    """Mesh extracts as a single zip blob when nested in other Packables."""

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema_obj: pydantic_core_schema.CoreSchema, handler: "GetJsonSchemaHandler"
    ) -> JsonSchemaValue:
        """Inject x-base hint into JSON schema indicating this is a Mesh."""
        json_schema = handler(core_schema_obj)
        json_schema = handler.resolve_ref_schema(json_schema)
        json_schema['x-base'] = 'Mesh'
        return json_schema

    # ============================================================
    # Field definitions - canonical types only (no Optional)
    # ============================================================

    vertices: Array = Field(..., description="Vertex positions as Nx3 float32 array")
    indices: IndexSequence = Field(..., description="Flattened 1D index array (uint32)")
    index_sizes: Array = Field(..., description="Number of vertices per cell (uint8)")
    cell_types: Array = Field(..., description="VTK cell type per cell (uint8)")
    dim: int = Field(default=3, description="Mesh dimension (2D or 3D)")
    
    # Markers - all canonical numpy arrays
    markers: Dict[str, Array] = Field(default_factory=dict, description="Marker indices (flattened)")
    marker_sizes: Dict[str, Array] = Field(default_factory=dict, description="Marker cell sizes")
    marker_cell_types: Dict[str, Array] = Field(default_factory=dict, description="Marker VTK cell types")

    # ============================================================
    # Factory methods for flexible input
    # ============================================================

    @classmethod
    def from_polygons(
        cls: type[TMesh],
        vertices: np.ndarray,
        polygons: Sequence[Sequence[int]],
        *,
        dim: Optional[int] = None,
        markers: Optional[Dict[str, Sequence[Sequence[int]]]] = None,
        **kwargs,
    ) -> TMesh:
        """Create mesh from list of polygon vertex indices.
        
        Args:
            vertices: Nx3 vertex positions
            polygons: List of polygons, each polygon is a list of vertex indices
            dim: Mesh dimension (auto-detected if None)
            markers: Optional dict of marker name -> list of element indices
            **kwargs: Additional fields for subclasses
            
        Returns:
            New Mesh instance
            
        Example:
            >>> mesh = Mesh.from_polygons(
            ...     vertices=np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]]),
            ...     polygons=[[0, 1, 2], [0, 2, 3]]
            ... )
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        result = ElementData.from_polygons([list(p) for p in polygons])
        
        # Process markers
        marker_dict: Dict[str, Array] = {}
        marker_sizes_dict: Dict[str, Array] = {}
        marker_cell_types_dict: Dict[str, Array] = {}
        
        if markers:
            for name, elements in markers.items():
                marker_result = ElementData.from_polygons([list(e) for e in elements])
                marker_dict[name] = marker_result.indices
                marker_sizes_dict[name] = marker_result.sizes
                marker_cell_types_dict[name] = marker_result.cell_types
        
        # Auto-compute dimension
        if dim is None:
            dim = CellTypeUtils.get_mesh_dimension(result.cell_types) if len(result.cell_types) > 0 else 3
        
        return cls(
            vertices=vertices,
            indices=result.indices,
            index_sizes=result.sizes,
            cell_types=result.cell_types,
            dim=dim,
            markers=marker_dict,
            marker_sizes=marker_sizes_dict,
            marker_cell_types=marker_cell_types_dict,
            **kwargs,
        )

    @classmethod
    def from_triangles(
        cls: type[TMesh],
        vertices: np.ndarray,
        triangles: np.ndarray,
        *,
        dim: int = 3,
        markers: Optional[Dict[str, Sequence[Sequence[int]]]] = None,
        **kwargs,
    ) -> TMesh:
        """Create mesh from Nx3 triangle index array.
        
        Args:
            vertices: Mx3 vertex positions
            triangles: Nx3 triangle indices
            dim: Mesh dimension (default 3)
            markers: Optional dict of marker name -> list of element indices
            **kwargs: Additional fields for subclasses
            
        Returns:
            New Mesh instance
            
        Example:
            >>> mesh = Mesh.from_triangles(
            ...     vertices=np.array([[0,0,0], [1,0,0], [0,1,0]]),
            ...     triangles=np.array([[0, 1, 2]])
            ... )
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.uint32).reshape(-1, 3)
        
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(f"triangles must be Nx3 array, got shape {triangles.shape}")
        
        num_triangles = len(triangles)
        
        # Process markers
        marker_dict: Dict[str, Array] = {}
        marker_sizes_dict: Dict[str, Array] = {}
        marker_cell_types_dict: Dict[str, Array] = {}
        
        if markers:
            for name, elements in markers.items():
                marker_result = ElementData.from_polygons([list(e) for e in elements])
                marker_dict[name] = marker_result.indices
                marker_sizes_dict[name] = marker_result.sizes
                marker_cell_types_dict[name] = marker_result.cell_types
        
        return cls(
            vertices=vertices,
            indices=triangles.flatten(),
            index_sizes=np.full(num_triangles, 3, dtype=np.uint8),
            cell_types=np.full(num_triangles, VTKCellType.VTK_TRIANGLE, dtype=np.uint8),
            dim=dim,
            markers=marker_dict,
            marker_sizes=marker_sizes_dict,
            marker_cell_types=marker_cell_types_dict,
            **kwargs,
        )

    @classmethod
    def create(
        cls: type[TMesh],
        vertices: np.ndarray,
        indices: Union[np.ndarray, Sequence[Sequence[int]], None] = None,
        *,
        index_sizes: Optional[np.ndarray] = None,
        cell_types: Optional[np.ndarray] = None,
        dim: Optional[int] = None,
        markers: Optional[Dict[str, Union[np.ndarray, Sequence[Sequence[int]]]]] = None,
        marker_sizes: Optional[Dict[str, np.ndarray]] = None,
        marker_cell_types: Optional[Dict[str, np.ndarray]] = None,
        **kwargs,
    ) -> TMesh:
        """Create mesh from various input formats (convenience wrapper).
        
        Handles:
        - 2D index arrays (uniform polygons)
        - List of lists (mixed polygons)
        - Flat arrays with explicit sizes
        - Markers in list or array format
        
        For best performance, use the direct constructor with pre-processed arrays.
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        
        # Process indices
        if indices is None:
            final_indices = np.array([], dtype=np.uint32)
            final_sizes = np.array([], dtype=np.uint8)
            final_types = np.array([], dtype=np.uint8)
        else:
            result = ElementUtils.convert_array_input(indices, index_sizes, cell_types)
            if result is not None:
                final_indices = result.indices
                final_sizes = result.sizes
                final_types = result.cell_types
            else:
                final_indices = np.array([], dtype=np.uint32)
                final_sizes = np.array([], dtype=np.uint8)
                final_types = np.array([], dtype=np.uint8)
        
        # Auto-compute dimension
        if dim is None and len(final_types) > 0:
            dim = CellTypeUtils.get_mesh_dimension(final_types)
        dim = dim or 3
        
        # Process markers
        final_markers: Dict[str, Array] = {}
        final_marker_sizes: Dict[str, Array] = {}
        final_marker_cell_types: Dict[str, Array] = {}
        
        if markers:
            for name, data in markers.items():
                if isinstance(data, np.ndarray):
                    final_markers[name] = data.astype(np.uint32)
                    if marker_sizes and name in marker_sizes:
                        final_marker_sizes[name] = marker_sizes[name]
                    if marker_cell_types and name in marker_cell_types:
                        final_marker_cell_types[name] = marker_cell_types[name]
                    # Validate required fields present
                    if name not in final_marker_sizes or name not in final_marker_cell_types:
                        raise ValueError(f"Marker '{name}' given as array requires marker_sizes and marker_cell_types")
                else:
                    # List of lists - convert
                    marker_result = ElementData.from_polygons([list(el) for el in data])
                    final_markers[name] = marker_result.indices
                    final_marker_sizes[name] = marker_result.sizes
                    final_marker_cell_types[name] = marker_result.cell_types
        
        return cls(
            vertices=vertices,
            indices=final_indices,
            index_sizes=final_sizes,
            cell_types=final_types,
            dim=dim,
            markers=final_markers,
            marker_sizes=final_marker_sizes,
            marker_cell_types=final_marker_cell_types,
            **kwargs,
        )

    # ============================================================
    # Properties
    # ============================================================

    @property
    def vertex_count(self) -> int:
        """Get the number of vertices."""
        return len(self.vertices)

    @property
    def index_count(self) -> int:
        """Get the number of indices."""
        return len(self.indices)

    @property
    def polygon_count(self) -> int:
        """Get the number of polygons/cells."""
        return len(self.index_sizes)

    @property
    def is_uniform_polygons(self) -> bool:
        """Check if all polygons have the same number of vertices."""
        if len(self.index_sizes) == 0:
            return True
        return ElementUtils.is_uniform_elements(self.index_sizes)

    def get_polygon_indices(self) -> Union[np.ndarray, List[List[int]]]:
        """Get indices in their original polygon structure.

        Returns:
            For uniform polygons: 2D numpy array where each row is a polygon
            For mixed polygons: List of lists where each sublist is a polygon
        """
        if len(self.indices) == 0:
            return np.array([], dtype=np.uint32).reshape(0, 3)

        if len(self.index_sizes) == 0:
            # Legacy format - assume triangles
            if len(self.indices) % 3 == 0:
                return self.indices.reshape(-1, 3)
            raise ValueError("Cannot determine polygon structure without index_sizes")

        return ElementUtils.get_element_structure(self.indices, self.index_sizes)

    def get_reconstructed_markers(self) -> Dict[str, List[List[int]]]:
        """Reconstruct marker elements from flattened structure back to list of lists."""
        reconstructed = {}
        for name, flattened_indices in self.markers.items():
            sizes = self.marker_sizes[name]
            cell_types = self.marker_cell_types.get(name)
            elements = ElementUtils.convert_flattened_to_list(flattened_indices, sizes, cell_types)
            reconstructed[name] = elements
        return reconstructed

    # ============================================================
    # Class methods
    # ============================================================

    @classmethod
    def combine(
        cls: type[TMesh],
        meshes: Sequence[TMesh],
        marker_names: Optional[List[str]] = None,
        preserve_markers: bool = True,
    ) -> TMesh:
        """Combine multiple meshes into a single mesh.

        Args:
            meshes: List of Mesh objects to combine
            marker_names: Optional marker names to assign to each mesh
            preserve_markers: Whether to preserve existing markers (default: True)

        Returns:
            A new combined Mesh object
        """
        if not meshes:
            raise ValueError("Cannot combine empty list of meshes")

        if marker_names is not None and len(marker_names) != len(meshes):
            raise ValueError(
                f"marker_names length ({len(marker_names)}) must match meshes length ({len(meshes)})"
            )

        vertex_offsets = MeshUtils.compute_vertex_offsets(list(meshes))

        all_vertices = [mesh.vertices for mesh in meshes]
        all_indices = [
            mesh.indices + vertex_offsets[i]
            for i, mesh in enumerate(meshes) if len(mesh.indices) > 0
        ]
        all_index_sizes = [mesh.index_sizes for mesh in meshes if len(mesh.index_sizes) > 0]
        all_cell_types = [mesh.cell_types for mesh in meshes if len(mesh.cell_types) > 0]

        if marker_names is not None:
            combined_markers, combined_marker_sizes, combined_marker_cell_types = \
                MeshUtils.combine_markers_with_names(list(meshes), marker_names, vertex_offsets)
        elif preserve_markers:
            combined_markers, combined_marker_sizes, combined_marker_cell_types = \
                MeshUtils.preserve_existing_markers(list(meshes), vertex_offsets)
        else:
            combined_markers = {}
            combined_marker_sizes = {}
            combined_marker_cell_types = {}

        combined_vertices = np.concatenate(all_vertices, axis=0)
        combined_indices = np.concatenate(all_indices, axis=0) if all_indices else np.array([], dtype=np.uint32)
        combined_index_sizes = np.concatenate(all_index_sizes, axis=0) if all_index_sizes else np.array([], dtype=np.uint8)
        combined_cell_types = np.concatenate(all_cell_types, axis=0) if all_cell_types else np.array([], dtype=np.uint8)

        extra_fields = {
            name: getattr(meshes[0], name)
            for name in cls.model_fields.keys() - Mesh.model_fields.keys()
        }

        return cls(
            vertices=combined_vertices,
            indices=combined_indices,
            index_sizes=combined_index_sizes,
            cell_types=combined_cell_types,
            dim=meshes[0].dim,
            markers=combined_markers,
            marker_sizes=combined_marker_sizes,
            marker_cell_types=combined_marker_cell_types,
            **extra_fields,
        )

    def extract_by_marker(self, marker_name: str) -> "Mesh":
        """Extract a submesh containing only elements referenced by a marker.

        Args:
            marker_name: Name of the marker to extract

        Returns:
            A new Mesh object containing only the marker's vertices/elements
        """
        if marker_name not in self.markers:
            raise ValueError(
                f"Marker '{marker_name}' not found. Available: {list(self.markers.keys())}"
            )

        marker_indices = self.markers[marker_name]
        marker_sizes = self.marker_sizes.get(marker_name)
        marker_cell_types = self.marker_cell_types.get(marker_name)

        if marker_sizes is None or marker_cell_types is None:
            raise ValueError(f"Marker '{marker_name}' missing size or cell type info")

        unique_vertices = np.unique(marker_indices)
        extracted_vertices = self.vertices[unique_vertices]
        remapped_indices = np.searchsorted(unique_vertices, marker_indices).astype(np.uint32)

        return Mesh(
            vertices=extracted_vertices,
            indices=remapped_indices,
            index_sizes=marker_sizes.copy(),
            cell_types=marker_cell_types.copy(),
            dim=self.dim,
        )

    # ============================================================
    # Mesh operations
    # ============================================================

    def triangulate(self) -> "Mesh":
        """Convert mesh to pure triangle surface mesh.

        Returns:
            A new mesh with all cells converted to triangles
        """
        if len(self.indices) == 0 or len(self.index_sizes) == 0:
            raise ValueError("Mesh must have indices and index_sizes to triangulate")

        if np.all(self.index_sizes == 3) and np.all(self.cell_types == VTKCellType.VTK_TRIANGLE):
            return self.model_copy(deep=True)

        cell_offsets = np.concatenate([[0], np.cumsum(self.index_sizes[:-1])]).astype(np.uint32)

        volume_types = set(TriangulationUtils._get_volume_cell_patterns().keys())
        effective_types = self.cell_types.copy()
        
        for i, (cell_type, size, offset) in enumerate(zip(self.cell_types, self.index_sizes, cell_offsets)):
            if cell_type in volume_types:
                cell_indices = self.indices[offset:offset + size]
                if TriangulationUtils.is_planar_cell(self.vertices, cell_indices):
                    effective_types[i] = VTKCellType.VTK_POLYGON

        result_chunks = []

        # Process triangles
        tri_mask = effective_types == VTKCellType.VTK_TRIANGLE
        if np.any(tri_mask):
            tri_offsets = cell_offsets[tri_mask]
            tri_sizes = self.index_sizes[tri_mask]
            for offset, size in zip(tri_offsets, tri_sizes):
                result_chunks.append(self.indices[offset:offset + size].copy())

        # Process polygons
        polygon_types = {VTKCellType.VTK_QUAD, VTKCellType.VTK_POLYGON}
        polygon_mask = np.isin(effective_types, list(polygon_types))
        if np.any(polygon_mask):
            poly_offsets = cell_offsets[polygon_mask]
            poly_sizes = self.index_sizes[polygon_mask]

            if np.any(poly_sizes < 3):
                invalid_idx = np.where(poly_sizes < 3)[0][0]
                raise ValueError(f"Polygon with {poly_sizes[invalid_idx]} vertices cannot be triangulated")

            for size in np.unique(poly_sizes):
                size_mask = poly_sizes == size
                size_offsets = poly_offsets[size_mask]
                if len(size_offsets) > 0:
                    tris = TriangulationUtils.triangulate_polygons(self.indices, size_offsets, size)
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
                        self.indices, offsets, cell_size, tri_pattern
                    )
                    if len(tris) > 0:
                        result_chunks.append(tris)

        # Validate all types handled
        skip_types = {VTKCellType.VTK_VERTEX, VTKCellType.VTK_LINE}
        supported_types = {VTKCellType.VTK_TRIANGLE} | polygon_types | volume_types
        all_handled = supported_types | skip_types

        for i, ct in enumerate(effective_types):
            if ct not in all_handled:
                raise ValueError(f"Unsupported cell type {ct} at cell {i}")

        if not result_chunks:
            raise ValueError("No triangulatable cells found in mesh")

        triangulated_indices = np.concatenate(result_chunks)
        num_triangles = len(triangulated_indices) // 3

        return Mesh(
            vertices=self.vertices.copy(),
            indices=triangulated_indices,
            index_sizes=np.full(num_triangles, 3, dtype=np.uint8),
            cell_types=np.full(num_triangles, VTKCellType.VTK_TRIANGLE, dtype=np.uint8),
            dim=self.dim,
            markers={name: data.copy() for name, data in self.markers.items()},
            marker_sizes={name: data.copy() for name, data in self.marker_sizes.items()},
            marker_cell_types={name: data.copy() for name, data in self.marker_cell_types.items()},
        )

    def optimize_vertex_cache(self) -> "Mesh":
        """Optimize mesh for vertex cache efficiency."""
        if len(self.indices) == 0:
            raise ValueError("Mesh has no indices to optimize")

        result_mesh = self.model_copy(deep=True)
        optimized_indices = np.zeros_like(result_mesh.indices)
        meshoptimizer.optimize_vertex_cache(
            optimized_indices, result_mesh.indices, result_mesh.index_count, result_mesh.vertex_count
        )
        result_mesh.indices = optimized_indices
        return result_mesh

    def optimize_overdraw(self, threshold: float = 1.05) -> "Mesh":
        """Optimize mesh to reduce overdraw."""
        if len(self.indices) == 0:
            raise ValueError("Mesh has no indices to optimize")

        result_mesh = self.model_copy(deep=True)
        optimized_indices = np.zeros_like(result_mesh.indices)
        meshoptimizer.optimize_overdraw(
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
        if len(self.indices) == 0:
            raise ValueError("Mesh has no indices to optimize")

        result_mesh = self.model_copy(deep=True)
        optimized_vertices = np.zeros_like(result_mesh.vertices)
        unique_vertex_count = meshoptimizer.optimize_vertex_fetch(
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

        Note: Simplification only works on triangle meshes.
        """
        if len(self.indices) == 0:
            raise ValueError("Mesh has no indices to simplify")

        result_mesh = self.model_copy(deep=True)
        target_index_count = int(result_mesh.index_count * target_ratio)
        simplified_indices = np.zeros(result_mesh.index_count, dtype=np.uint32)

        result_error = np.array([0.0], dtype=np.float32)
        new_index_count = meshoptimizer.simplify(
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

        num_triangles = new_index_count // 3
        result_mesh.index_sizes = np.full(num_triangles, 3, dtype=np.uint8)
        result_mesh.cell_types = np.full(num_triangles, VTKCellType.VTK_TRIANGLE, dtype=np.uint8)

        return result_mesh

    # ============================================================
    # Export methods
    # ============================================================

    def to_pyvista(self):
        """Convert mesh to a PyVista UnstructuredGrid.
        
        Requires pyvista to be installed.
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError(
                "pyvista is required for VTK export. "
                "Install with: pip install meshly[dev] or pip install pyvista"
            )
        
        if len(self.indices) == 0 or len(self.index_sizes) == 0 or len(self.cell_types) == 0:
            raise ValueError("Mesh must have indices, index_sizes, and cell_types for VTK export")
        
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
        
        Supports .vtk, .vtu, .ply, .stl and other formats supported by PyVista.
        """
        pv_mesh = self.to_pyvista()
        pv_mesh.save(str(path))
