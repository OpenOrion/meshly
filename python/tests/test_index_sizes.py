"""
Tests for the index_sizes and cell_types functionality in the Mesh class.

This file contains tests to verify that the index_sizes and cell_types fields work correctly
for different polygon types and formats, including automatic inference,
validation, and preservation during serialization.
"""
import os
import tempfile
import numpy as np
import pytest
from io import BytesIO
from pydantic import ValidationError

from meshly import Mesh


class TestIndexSizes:
    """Test index_sizes functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        # Create vertices for a simple mesh
        self.vertices = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.5, 0.5, 1.0],  # 4
            [2.0, 0.0, 0.0],  # 5
            [2.0, 1.0, 0.0]   # 6
        ], dtype=np.float32)

    def test_triangular_indices_no_index_sizes(self):
        """Test triangular mesh without explicit index_sizes."""
        # Traditional triangular mesh format
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        mesh = Mesh(vertices=self.vertices, indices=indices)

        assert mesh.index_count == 6
        assert mesh.polygon_count == 2  # Now auto-infers triangles
        # Auto-inferred uniform triangles
        assert mesh.is_uniform_polygons
        # Auto-inferred triangle sizes
        np.testing.assert_array_equal(mesh.index_sizes, [3, 3])

    def test_quad_indices_2d_array(self):
        """Test quad mesh using 2D numpy array with automatic index_sizes inference."""
        # 2D array format for uniform quads
        indices = np.array([
            [0, 1, 2, 3],  # First quad
            [1, 5, 6, 2]   # Second quad
        ], dtype=np.uint32)

        mesh = Mesh(vertices=self.vertices, indices=indices)

        assert mesh.index_count == 8  # Flattened to 8 indices
        assert mesh.polygon_count == 2  # 2 polygons
        assert mesh.is_uniform_polygons  # All quads
        np.testing.assert_array_equal(mesh.index_sizes, [4, 4])

        # Check that we can get back the original structure
        polygon_indices = mesh.get_polygon_indices()
        np.testing.assert_array_equal(polygon_indices, indices)

    def test_mixed_polygons_list_of_lists(self):
        """Test mixed polygon mesh using list of lists with automatic index_sizes inference."""
        # Mixed polygon format: triangle, quad, pentagon
        indices = [
            [0, 1, 2],        # Triangle
            [1, 5, 6, 2],     # Quad
            [0, 1, 4, 3, 2]   # Pentagon
        ]

        mesh = Mesh(vertices=self.vertices, indices=indices)

        assert mesh.index_count == 12  # 3 + 4 + 5 = 12
        assert mesh.polygon_count == 3  # 3 polygons
        assert not mesh.is_uniform_polygons  # Mixed sizes
        np.testing.assert_array_equal(mesh.index_sizes, [3, 4, 5])

        # Check that we can get back the original structure
        polygon_indices = mesh.get_polygon_indices()
        assert len(polygon_indices) == 3
        assert polygon_indices[0] == [0, 1, 2]
        assert polygon_indices[1] == [1, 5, 6, 2]
        assert polygon_indices[2] == [0, 1, 4, 3, 2]

    def test_explicit_index_sizes_validation(self):
        """Test explicit index_sizes validation."""
        # Flat indices with explicit index_sizes
        flat_indices = np.array([0, 1, 2, 1, 5, 6, 2, 0, 1, 4], dtype=np.uint32)
        # Triangle, quad, triangle
        explicit_sizes = np.array([3, 4, 3], dtype=np.uint32)

        mesh = Mesh(
            vertices=self.vertices,
            indices=flat_indices,
            index_sizes=explicit_sizes
        )

        assert mesh.index_count == 10
        assert mesh.polygon_count == 3
        assert not mesh.is_uniform_polygons
        np.testing.assert_array_equal(mesh.index_sizes, [3, 4, 3])

    def test_index_sizes_validation_mismatch(self):
        """Test that validation fails when index_sizes doesn't match indices."""
        # Indices sum to 10, but index_sizes sum to 8
        flat_indices = np.array([0, 1, 2, 1, 5, 6, 2, 0, 1, 4], dtype=np.uint32)
        wrong_sizes = np.array([3, 5], dtype=np.uint32)  # Sum = 8, not 10

        with pytest.raises(ValidationError):
            Mesh(
                vertices=self.vertices,
                indices=flat_indices,
                index_sizes=wrong_sizes
            )

    def test_explicit_vs_inferred_index_sizes(self):
        """Test validation when explicit index_sizes conflicts with inferred structure."""
        # 2D array format that infers [4, 4]
        indices = np.array([
            [0, 1, 2, 3],
            [1, 5, 6, 2]
        ], dtype=np.uint32)

        # Explicit sizes that conflict with inferred structure
        conflicting_sizes = np.array([3, 5], dtype=np.uint32)

        with pytest.raises(ValidationError):
            Mesh(
                vertices=self.vertices,
                indices=indices,
                index_sizes=conflicting_sizes
            )

    def test_index_sizes_encoding_decoding(self):
        """Test that index_sizes is preserved during encoding/decoding."""
        # Create a mixed polygon mesh
        indices = [
            [0, 1, 2],        # Triangle
            [1, 5, 6, 2],     # Quad
            [0, 3, 4]         # Another triangle
        ]

        mesh = Mesh(vertices=self.vertices, indices=indices)
        original_index_sizes = mesh.index_sizes.copy()

        # Encode the mesh - returns bytes
        encoded_mesh = mesh.encode()
        assert isinstance(encoded_mesh, bytes)
        assert len(encoded_mesh) > 0

        # Decode via zip round-trip
        buffer = BytesIO()
        mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_mesh = Mesh.load_from_zip(buffer)

        # Check that index_sizes is preserved
        np.testing.assert_array_equal(decoded_mesh.index_sizes, original_index_sizes)
        assert decoded_mesh.polygon_count == 3
        assert not decoded_mesh.is_uniform_polygons

        original_polygons = mesh.get_polygon_indices()
        decoded_polygons = decoded_mesh.get_polygon_indices()
        assert len(original_polygons) == len(decoded_polygons)
        for orig, decoded in zip(original_polygons, decoded_polygons):
            assert orig == decoded

    def test_index_sizes_serialization(self):
        """Test that index_sizes is preserved during ZIP serialization."""
        indices = [
            [0, 1, 2, 3],
            [1, 5, 6],
            [0, 3, 4, 2, 1]
        ]

        mesh = Mesh(vertices=self.vertices, indices=indices)
        original_index_sizes = mesh.index_sizes.copy()

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            mesh.save_to_zip(temp_path)
            loaded_mesh = Mesh.load_from_zip(temp_path)

            np.testing.assert_array_equal(loaded_mesh.index_sizes, original_index_sizes)
            assert loaded_mesh.polygon_count == 3
            assert not loaded_mesh.is_uniform_polygons

            original_polygons = mesh.get_polygon_indices()
            loaded_polygons = loaded_mesh.get_polygon_indices()
            assert len(original_polygons) == len(loaded_polygons)
            for orig, loaded in zip(original_polygons, loaded_polygons):
                assert orig == loaded
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_uniform_vs_mixed_polygons(self):
        """Test uniform vs mixed polygon detection."""
        triangle_indices = [[0, 1, 2], [1, 5, 6], [0, 3, 4]]
        triangle_mesh = Mesh(vertices=self.vertices, indices=triangle_indices)
        assert triangle_mesh.is_uniform_polygons
        np.testing.assert_array_equal(triangle_mesh.index_sizes, [3, 3, 3])

        quad_indices = np.array([[0, 1, 2, 3], [1, 5, 6, 2]], dtype=np.uint32)
        quad_mesh = Mesh(vertices=self.vertices, indices=quad_indices)
        assert quad_mesh.is_uniform_polygons
        np.testing.assert_array_equal(quad_mesh.index_sizes, [4, 4])

        mixed_indices = [[0, 1, 2], [1, 5, 6, 2]]
        mixed_mesh = Mesh(vertices=self.vertices, indices=mixed_indices)
        assert not mixed_mesh.is_uniform_polygons
        np.testing.assert_array_equal(mixed_mesh.index_sizes, [3, 4])

    def test_polygon_reconstruction(self):
        """Test polygon reconstruction from flat indices and index_sizes."""
        flat_indices = np.array([0, 1, 2, 1, 5, 6, 2, 0, 3, 4], dtype=np.uint32)
        sizes = np.array([3, 4, 3], dtype=np.uint32)

        mesh = Mesh(
            vertices=self.vertices,
            indices=flat_indices,
            index_sizes=sizes
        )

        uniform_indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        uniform_mesh = Mesh(vertices=self.vertices, indices=uniform_indices)
        reconstructed_uniform = uniform_mesh.get_polygon_indices()
        assert isinstance(reconstructed_uniform, np.ndarray)
        assert reconstructed_uniform.shape == (2, 3)
        np.testing.assert_array_equal(reconstructed_uniform, uniform_indices)

        mixed_polygons = mesh.get_polygon_indices()
        assert isinstance(mixed_polygons, list)
        assert len(mixed_polygons) == 3
        assert mixed_polygons[0] == [0, 1, 2]
        assert mixed_polygons[1] == [1, 5, 6, 2]
        assert mixed_polygons[2] == [0, 3, 4]


class TestIndexSizesIntegrity:
    """Test index_sizes integrity during mesh operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.5, 0.5, 1.0]
        ], dtype=np.float32)

    def test_copy_preserves_index_sizes(self):
        """Test that copying a mesh preserves index_sizes."""
        indices = [
            [0, 1, 2, 3],
            [1, 5, 6],
            [4, 5, 6, 7, 2]
        ]

        mesh = Mesh(vertices=self.vertices, indices=indices)
        copied_mesh = mesh.model_copy(deep=True)

        np.testing.assert_array_equal(copied_mesh.index_sizes, mesh.index_sizes)
        assert copied_mesh.polygon_count == mesh.polygon_count
        assert copied_mesh.is_uniform_polygons == mesh.is_uniform_polygons

        assert copied_mesh.index_sizes is not mesh.index_sizes
        assert copied_mesh.indices is not mesh.indices

    def test_optimization_with_index_sizes(self):
        """Test that mesh optimizations work correctly with index_sizes."""
        vertices = []
        indices = []

        for i in range(3):
            for j in range(3):
                base = len(vertices)
                vertices.extend([
                    [j, i, 0], [j+1, i, 0], [j+1, i+1, 0], [j, i+1, 0]
                ])

                if (i + j) % 2 == 0:
                    indices.append([base, base+1, base+2, base+3])
                else:
                    indices.extend([[base, base+1, base+2], [base, base+2, base+3]])

        mesh_vertices = np.array(vertices, dtype=np.float32)
        mesh = Mesh(vertices=mesh_vertices, indices=indices)

        original_polygon_count = mesh.polygon_count
        original_is_uniform = mesh.is_uniform_polygons

        optimized_cache = mesh.optimize_vertex_cache()
        optimized_overdraw = mesh.optimize_overdraw()

        assert optimized_cache.polygon_count == original_polygon_count
        assert optimized_cache.is_uniform_polygons == original_is_uniform
        assert optimized_overdraw.polygon_count == original_polygon_count
        assert optimized_overdraw.is_uniform_polygons == original_is_uniform

        assert mesh.polygon_count == original_polygon_count
        assert mesh.is_uniform_polygons == original_is_uniform


class TestCellTypes:
    """Test cell_types functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.5, 0.5, 1.0]
        ], dtype=np.float32)

    def test_cell_types_auto_inference(self):
        """Test automatic inference of cell_types from index_sizes."""
        indices = [
            [0, 1],
            [0, 1, 2],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4]
        ]

        mesh = Mesh(vertices=self.vertices, indices=indices)

        expected_cell_types = [3, 5, 9, 14]
        np.testing.assert_array_equal(mesh.cell_types, expected_cell_types)
        np.testing.assert_array_equal(mesh.index_sizes, [2, 3, 4, 5])

    def test_cell_types_explicit(self):
        """Test explicit cell_types validation."""
        indices = [[0, 1, 2], [1, 2, 3, 4]]
        explicit_cell_types = [5, 9]

        mesh = Mesh(
            vertices=self.vertices,
            indices=indices,
            cell_types=explicit_cell_types
        )

        np.testing.assert_array_equal(mesh.cell_types, [5, 9])
        np.testing.assert_array_equal(mesh.index_sizes, [3, 4])

    def test_cell_types_length_mismatch(self):
        """Test that validation fails when cell_types length doesn't match index_sizes."""
        indices = [[0, 1, 2], [1, 2, 3, 4]]
        wrong_cell_types = [5]

        with pytest.raises(ValidationError):
            Mesh(
                vertices=self.vertices,
                indices=indices,
                cell_types=wrong_cell_types
            )

    def test_cell_types_encoding_decoding(self):
        """Test that cell_types is preserved during encoding/decoding."""
        indices = [[0, 1, 2], [1, 2, 3, 4], [0, 4, 5]]
        explicit_cell_types = [5, 9, 5]

        mesh = Mesh(
            vertices=self.vertices,
            indices=indices,
            cell_types=explicit_cell_types
        )

        encoded_mesh = mesh.encode()
        assert isinstance(encoded_mesh, bytes)
        assert len(encoded_mesh) > 0

        buffer = BytesIO()
        mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_mesh = Mesh.load_from_zip(buffer)

        np.testing.assert_array_equal(decoded_mesh.cell_types, explicit_cell_types)
        np.testing.assert_array_equal(decoded_mesh.index_sizes, [3, 4, 3])

    def test_cell_types_serialization(self):
        """Test that cell_types is preserved during ZIP serialization."""
        indices = [[0], [0, 1], [0, 1, 2], [1, 2, 3, 4]]
        explicit_cell_types = [1, 3, 5, 9]

        mesh = Mesh(
            vertices=self.vertices,
            indices=indices,
            cell_types=explicit_cell_types
        )

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            mesh.save_to_zip(temp_path)
            loaded_mesh = Mesh.load_from_zip(temp_path)

            np.testing.assert_array_equal(loaded_mesh.cell_types, explicit_cell_types)
            np.testing.assert_array_equal(loaded_mesh.index_sizes, [1, 2, 3, 4])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_cell_types_copy_preservation(self):
        """Test that cell_types is preserved during mesh copying."""
        indices = [[0, 1, 2, 3], [1, 2, 4]]
        explicit_cell_types = [9, 5]

        mesh = Mesh(
            vertices=self.vertices,
            indices=indices,
            cell_types=explicit_cell_types
        )

        copied_mesh = mesh.model_copy(deep=True)

        np.testing.assert_array_equal(copied_mesh.cell_types, mesh.cell_types)
        np.testing.assert_array_equal(copied_mesh.index_sizes, mesh.index_sizes)

        assert copied_mesh.cell_types is not mesh.cell_types
        assert copied_mesh.index_sizes is not mesh.index_sizes

    @pytest.mark.parametrize("indices_list,expected_cell_types", [
        ([0], [1]),
        ([0, 1], [3]),
        ([0, 1, 2], [5]),
        ([0, 1, 2, 3], [9]),
        ([0, 1, 2, 3, 4], [14]),
        ([0, 1, 2, 3, 4, 5], [13]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [12]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 0], [7]),
    ])
    def test_cell_types_vtk_inference(self, indices_list, expected_cell_types):
        """Test VTK cell type inference for various polygon sizes."""
        mesh = Mesh(vertices=self.vertices, indices=[indices_list])
        np.testing.assert_array_equal(mesh.cell_types, expected_cell_types)
