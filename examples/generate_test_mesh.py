"""
Generate a test mesh using pymeshoptimizer and save it to a zip file.
This will be used to test the TypeScript library's import functionality.
"""

import os
import numpy as np
from pymeshoptimizer import Mesh

def generate_cube_mesh():
    """Generate a simple cube mesh."""
    # Create a simple cube mesh
    vertices = np.array([
        # positions
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2, 2, 3, 0,  # front
        1, 5, 6, 6, 2, 1,  # right
        5, 4, 7, 7, 6, 5,  # back
        4, 0, 3, 3, 7, 4,  # left
        3, 2, 6, 6, 7, 3,  # top
        4, 5, 1, 1, 0, 4   # bottom
    ], dtype=np.uint32)

    # Create normals (one per vertex)
    normals = np.array([
        [0.0, 0.0, -1.0],  # front vertices
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],   # back vertices
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Create colors (RGBA, one per vertex)
    colors = np.array([
        [1.0, 0.0, 0.0, 1.0],  # red
        [0.0, 1.0, 0.0, 1.0],  # green
        [0.0, 0.0, 1.0, 1.0],  # blue
        [1.0, 1.0, 0.0, 1.0],  # yellow
        [1.0, 0.0, 1.0, 1.0],  # magenta
        [0.0, 1.0, 1.0, 1.0],  # cyan
        [0.5, 0.5, 0.5, 1.0],  # gray
        [1.0, 1.0, 1.0, 1.0]   # white
    ], dtype=np.float32)

    # Create a custom mesh class with normals and colors
    class ColoredMesh(Mesh):
        """A custom mesh class with normals and colors."""
        normals: np.ndarray
        colors: np.ndarray

    # Create the mesh
    mesh = ColoredMesh(
        vertices=vertices,
        indices=indices,
        normals=normals,
        colors=colors
    )

    return mesh

def main():
    """Generate a test mesh and save it to a zip file."""
    # Create the mesh
    mesh = generate_cube_mesh()

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'typescript', 'examples')
    os.makedirs(output_dir, exist_ok=True)

    # Save the mesh to a zip file
    output_path = os.path.join(output_dir, 'test-mesh.zip')
    mesh.save_to_zip(output_path)

    print(f"Mesh saved to {output_path}")

if __name__ == "__main__":
    main()