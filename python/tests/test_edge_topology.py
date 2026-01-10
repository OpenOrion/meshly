"""Tests for edge topology extraction in CellTypeUtils."""

import unittest
from meshly import CellTypeUtils, VTKCellType


class TestEdgeTopology(unittest.TestCase):
    """Tests for VTK_EDGE_TOPOLOGY and edge extraction methods."""

    def test_hexahedron_has_12_edges(self):
        """A hexahedron should have exactly 12 edges."""
        topology = CellTypeUtils.get_edge_topology(VTKCellType.VTK_HEXAHEDRON)
        self.assertEqual(len(topology), 12)

    def test_tetrahedron_has_6_edges(self):
        """A tetrahedron should have exactly 6 edges."""
        topology = CellTypeUtils.get_edge_topology(VTKCellType.VTK_TETRA)
        self.assertEqual(len(topology), 6)

    def test_wedge_has_9_edges(self):
        """A wedge/prism should have exactly 9 edges."""
        topology = CellTypeUtils.get_edge_topology(VTKCellType.VTK_WEDGE)
        self.assertEqual(len(topology), 9)

    def test_pyramid_has_8_edges(self):
        """A pyramid should have exactly 8 edges."""
        topology = CellTypeUtils.get_edge_topology(VTKCellType.VTK_PYRAMID)
        self.assertEqual(len(topology), 8)

    def test_triangle_has_3_edges(self):
        """A triangle should have exactly 3 edges."""
        topology = CellTypeUtils.get_edge_topology(VTKCellType.VTK_TRIANGLE)
        self.assertEqual(len(topology), 3)

    def test_quad_has_4_edges(self):
        """A quad should have exactly 4 edges."""
        topology = CellTypeUtils.get_edge_topology(VTKCellType.VTK_QUAD)
        self.assertEqual(len(topology), 4)

    def test_unknown_type_returns_empty(self):
        """Unknown cell types should return empty edge topology."""
        topology = CellTypeUtils.get_edge_topology(999)
        self.assertEqual(len(topology), 0)


class TestGetCellEdges(unittest.TestCase):
    """Tests for get_cell_edges method."""

    def test_hexahedron_edges_normalized(self):
        """Hexahedron edges should be returned with u < v."""
        vertices = [10, 20, 30, 40, 50, 60, 70, 80]  # global vertex indices
        edges = CellTypeUtils.get_cell_edges(
            vertices, VTKCellType.VTK_HEXAHEDRON)

        self.assertEqual(len(edges), 12)
        for u, v in edges:
            self.assertLess(u, v, f"Edge ({u}, {v}) not normalized")

    def test_tetrahedron_edges(self):
        """Test tetrahedron edge extraction."""
        vertices = [0, 1, 2, 3]
        edges = CellTypeUtils.get_cell_edges(vertices, VTKCellType.VTK_TETRA)

        # Should have all 6 edges of a tetrahedron
        expected_edges = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        actual_edges = {tuple(e) for e in edges}
        self.assertEqual(actual_edges, expected_edges)

    def test_triangle_edges(self):
        """Test triangle edge extraction."""
        vertices = [5, 10, 15]
        edges = CellTypeUtils.get_cell_edges(
            vertices, VTKCellType.VTK_TRIANGLE)

        expected_edges = {(5, 10), (10, 15), (5, 15)}
        actual_edges = {tuple(e) for e in edges}
        self.assertEqual(actual_edges, expected_edges)

    def test_unknown_type_treated_as_polygon(self):
        """Unknown types should be treated as polygons (consecutive edges)."""
        vertices = [0, 1, 2, 3, 4]  # 5 vertices, unknown type
        edges = CellTypeUtils.get_cell_edges(vertices, 999)  # Unknown type

        # Should connect consecutive vertices
        expected_edges = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
        actual_edges = {tuple(e) for e in edges}
        self.assertEqual(actual_edges, expected_edges)


class TestGetEdgesFromElementSize(unittest.TestCase):
    """Tests for get_edges_from_element_size method."""

    def test_8_vertices_inferred_as_hex(self):
        """8 vertices should be inferred as hexahedron."""
        element = list(range(8))
        edges = CellTypeUtils.get_edges_from_element_size(element)
        self.assertEqual(len(edges), 12)

    def test_4_vertices_inferred_as_quad(self):
        """4 vertices should be inferred as quad (not tetrahedron)."""
        element = [0, 1, 2, 3]
        edges = CellTypeUtils.get_edges_from_element_size(element)

        # Quad has 4 edges, tetra would have 6
        self.assertEqual(len(edges), 4)

    def test_3_vertices_inferred_as_triangle(self):
        """3 vertices should be inferred as triangle."""
        element = [0, 1, 2]
        edges = CellTypeUtils.get_edges_from_element_size(element)
        self.assertEqual(len(edges), 3)

    def test_6_vertices_inferred_as_wedge(self):
        """6 vertices should be inferred as wedge."""
        element = list(range(6))
        edges = CellTypeUtils.get_edges_from_element_size(element)
        self.assertEqual(len(edges), 9)

    def test_5_vertices_inferred_as_pyramid(self):
        """5 vertices should be inferred as pyramid."""
        element = list(range(5))
        edges = CellTypeUtils.get_edges_from_element_size(element)
        self.assertEqual(len(edges), 8)

    def test_unknown_size_treated_as_polygon(self):
        """Unknown sizes should be treated as polygons."""
        element = list(range(7))  # 7 vertices - no standard cell type
        edges = CellTypeUtils.get_edges_from_element_size(element)

        # Should have 7 edges (connecting consecutive vertices in a ring)
        self.assertEqual(len(edges), 7)


if __name__ == "__main__":
    unittest.main()
