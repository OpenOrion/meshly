"""
Mesh utility functions for combining meshes and managing markers.

This module contains utility functions that help with mesh combination
and marker management. The actual mesh operations (triangulate, optimize,
simplify) are instance methods on the Mesh class.
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..mesh import Mesh


class MeshUtils:
    """Utility class for mesh marker and combination operations."""

    @staticmethod
    def compute_vertex_offsets(meshes: List["Mesh"]) -> np.ndarray:
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
        mesh: "Mesh",
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
            indices = mesh.indices.copy() + vertex_offset
            sizes = mesh.index_sizes.copy()
            cell_types = mesh.cell_types.copy() if mesh.cell_types is not None else None
            return indices, sizes, cell_types
        else:
            vertex_count = mesh.vertex_count
            indices = np.arange(vertex_offset, vertex_offset +
                                vertex_count, dtype=np.uint32)
            sizes = np.ones(vertex_count, dtype=np.uint32)
            cell_types = np.ones(vertex_count, dtype=np.uint32)
            return indices, sizes, cell_types

    @staticmethod
    def combine_markers_with_names(
        meshes: List["Mesh"],
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
        meshes: List["Mesh"],
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
                offset_marker_indices = marker_indices + vertex_offsets[mesh_idx]

                sizes = mesh.marker_sizes.get(marker_name)
                cell_types = mesh.marker_cell_types.get(marker_name)

                if sizes is None:
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
