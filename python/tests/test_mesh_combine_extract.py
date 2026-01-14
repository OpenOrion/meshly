"""
Tests for mesh combine and extract_by_marker methods.
"""

import pytest
import numpy as np
from meshly.mesh import Mesh


class TestMeshCombine:
    """Test cases for Mesh.combine() method."""

    def test_combine_simple_meshes(self):
        """Test combining two simple meshes without markers."""
        mesh1 = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        mesh2 = Mesh(
            vertices=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        combined = Mesh.combine([mesh1, mesh2])

        assert combined.vertex_count == 6
        assert np.array_equal(combined.vertices[:3], mesh1.vertices)
        assert np.array_equal(combined.vertices[3:], mesh2.vertices)

        expected_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        assert np.array_equal(combined.indices, expected_indices)

    def test_combine_with_marker_names(self):
        """Test combining meshes with marker names."""
        mesh1 = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        mesh2 = Mesh(
            vertices=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        combined = Mesh.combine([mesh1, mesh2], marker_names=["part1", "part2"])

        assert "part1" in combined.markers
        assert "part2" in combined.markers

        assert np.array_equal(
            combined.markers["part1"], np.array([0, 1, 2], dtype=np.uint32))
        assert np.array_equal(
            combined.markers["part2"], np.array([3, 4, 5], dtype=np.uint32))

    def test_combine_preserve_existing_markers(self):
        """Test combining meshes while preserving existing markers."""
        mesh1 = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"boundary": np.array([0, 1], dtype=np.uint32)},
            marker_sizes={"boundary": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"boundary": np.array([1, 1], dtype=np.uint32)},
        )

        mesh2 = Mesh(
            vertices=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"edge": np.array([1, 2], dtype=np.uint32)},
            marker_sizes={"edge": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"edge": np.array([1, 1], dtype=np.uint32)},
        )

        combined = Mesh.combine([mesh1, mesh2], preserve_markers=True)

        assert "boundary" in combined.markers
        assert "edge" in combined.markers

        assert np.array_equal(
            combined.markers["boundary"], np.array([0, 1], dtype=np.uint32))
        assert np.array_equal(
            combined.markers["edge"], np.array([4, 5], dtype=np.uint32))

    def test_combine_with_marker_names_and_preserve(self):
        """Test that marker_names takes precedence over existing markers."""
        mesh1 = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"top": np.array([2], dtype=np.uint32)},
            marker_sizes={"top": np.array([1], dtype=np.uint32)},
            marker_cell_types={"top": np.array([1], dtype=np.uint32)},
        )

        mesh2 = Mesh(
            vertices=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        combined = Mesh.combine(
            [mesh1, mesh2], marker_names=["left", "right"], preserve_markers=True)

        assert "left" in combined.markers
        assert "right" in combined.markers
        assert "top" not in combined.markers

        assert len(combined.markers) == 2

    def test_combine_same_marker_names(self):
        """Test combining meshes with same marker names - should merge them."""
        mesh1 = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"boundary": np.array([0, 1], dtype=np.uint32)},
            marker_sizes={"boundary": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"boundary": np.array([1, 1], dtype=np.uint32)},
        )

        mesh2 = Mesh(
            vertices=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"boundary": np.array([1, 2], dtype=np.uint32)},
            marker_sizes={"boundary": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"boundary": np.array([1, 1], dtype=np.uint32)},
        )

        combined = Mesh.combine([mesh1, mesh2], preserve_markers=True)

        assert "boundary" in combined.markers
        assert len(combined.markers["boundary"]) == 4

        expected_indices = np.array([0, 1, 4, 5], dtype=np.uint32)
        assert np.array_equal(combined.markers["boundary"], expected_indices)

    def test_combine_empty_list(self):
        """Test that combining empty list raises error."""
        with pytest.raises(ValueError, match="Cannot combine empty list"):
            Mesh.combine([])

    def test_combine_marker_names_length_mismatch(self):
        """Test that mismatched marker_names length raises error."""
        mesh1 = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        with pytest.raises(ValueError, match="marker_names length"):
            Mesh.combine([mesh1], marker_names=["part1", "part2"])


class TestMeshExtract:
    """Test cases for Mesh.extract_by_marker() method."""

    def test_extract_by_marker(self):
        """Test extracting a submesh by marker name."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

        mesh = Mesh(
            vertices=vertices,
            indices=indices,
            markers={"edge": np.array([0, 1], dtype=np.uint32)},
            marker_sizes={"edge": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"edge": np.array([1, 1], dtype=np.uint32)},
        )

        extracted = mesh.extract_by_marker("edge")

        assert extracted.vertex_count == 2
        assert np.array_equal(extracted.vertices[0], vertices[0])
        assert np.array_equal(extracted.vertices[1], vertices[1])

        assert np.array_equal(extracted.indices, np.array([0, 1], dtype=np.uint32))

    def test_extract_by_marker_with_triangles(self):
        """Test extracting a submesh with triangle elements."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]], dtype=np.float32)

        mesh = Mesh(
            vertices=vertices,
            markers={"boundary": np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)},
            marker_sizes={"boundary": np.array([3, 3], dtype=np.uint32)},
            marker_cell_types={"boundary": np.array([5, 5], dtype=np.uint32)},
        )

        extracted = mesh.extract_by_marker("boundary")

        assert extracted.vertex_count == 4
        assert extracted.index_count == 6

        assert np.array_equal(
            extracted.indices, np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32))

    def test_extract_by_marker_nonexistent(self):
        """Test that extracting by nonexistent marker raises error."""
        mesh = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        with pytest.raises(ValueError, match="Marker 'nonexistent' not found"):
            mesh.extract_by_marker("nonexistent")


class TestMeshCombineAndExtract:
    """Test cases for combining and extracting meshes."""

    def test_combine_and_extract_roundtrip(self):
        """Test combining meshes and extracting them back."""
        mesh1 = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        mesh2 = Mesh(
            vertices=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        combined = Mesh.combine([mesh1, mesh2], marker_names=["part1", "part2"])

        extracted1 = combined.extract_by_marker("part1")

        assert extracted1.vertex_count == mesh1.vertex_count
        assert np.allclose(extracted1.vertices, mesh1.vertices)

        extracted2 = combined.extract_by_marker("part2")

        assert extracted2.vertex_count == mesh2.vertex_count
        assert np.allclose(extracted2.vertices, mesh2.vertices)
