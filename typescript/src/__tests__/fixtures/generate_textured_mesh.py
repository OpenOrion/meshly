#!/usr/bin/env python3
"""
Generate a TexturedMesh fixture for TypeScript integration testing.

This script creates a mesh similar to the mesh_example.ipynb TexturedMesh
and saves it to a zip file that TypeScript tests can load.
"""
import sys
from pathlib import Path

# Add meshly to path
meshly_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "python"
sys.path.insert(0, str(meshly_path))

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing import ClassVar

from meshly import Mesh, Array, Packable


class MaterialProperties(BaseModel):
    """Material properties with numpy arrays."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Material name")
    diffuse: Array = Field(..., description="Diffuse color array")
    specular: Array = Field(..., description="Specular color array")
    shininess: float = Field(32.0, description="Shininess value")


class PhysicsProperties(Packable):
    """Physics properties as a nested Packable."""
    is_contained: ClassVar[bool] = True
    mass: float = Field(1.0, description="Object mass")
    friction: float = Field(0.5, description="Friction coefficient")
    inertia_tensor: Array = Field(..., description="3x3 inertia tensor")
    collision_points: Array = Field(..., description="Collision sample points")


class TexturedMesh(Mesh):
    """
    A mesh with texture coordinates and normals.
    """
    texture_coords: Array = Field(..., description="Texture coordinates")
    normals: Array | None = Field(None, description="Vertex normals")
    material_name: str = Field("default", description="Material name")
    tags: list[str] = Field(default_factory=list, description="Tags for the mesh")

    # Dictionary containing nested dictionaries with arrays
    material_data: dict[str, dict[str, Array]] = Field(
        default_factory=dict,
        description="Nested dictionary structure with arrays"
    )

    physics: PhysicsProperties | None = Field(
        None,
        description="Physics properties as a nested Packable"
    )

    material_colors: dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary with non-array values"
    )

    # Dictionary containing BaseModel instances with numpy arrays
    materials: dict[str, MaterialProperties] = Field(
        default_factory=dict,
        description="Dictionary of material name to MaterialProperties"
    )


def create_textured_mesh() -> TexturedMesh:
    """Create a simple textured cube mesh."""
    # Create vertices for a cube
    vertices = np.array([
        [-0.5, -0.5, -0.5],  # 0: bottom-left-back
        [0.5, -0.5, -0.5],   # 1: bottom-right-back
        [0.5, 0.5, -0.5],    # 2: top-right-back
        [-0.5, 0.5, -0.5],   # 3: top-left-back
        [-0.5, -0.5, 0.5],   # 4: bottom-left-front
        [0.5, -0.5, 0.5],    # 5: bottom-right-front
        [0.5, 0.5, 0.5],     # 6: top-right-front
        [-0.5, 0.5, 0.5]     # 7: top-left-front
    ], dtype=np.float32)

    # Create indices for the cube (2 triangles per face, 6 faces)
    indices = np.array([
        0, 1, 2, 2, 3, 0,  # back face
        1, 5, 6, 6, 2, 1,  # right face
        5, 4, 7, 7, 6, 5,  # front face
        4, 0, 3, 3, 7, 4,  # left face
        3, 2, 6, 6, 7, 3,  # top face
        4, 5, 1, 1, 0, 4   # bottom face
    ], dtype=np.uint32)

    # Create texture coordinates (one for each vertex)
    texture_coords = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.0, 0.0],  # 4
        [1.0, 0.0],  # 5
        [1.0, 1.0],  # 6
        [0.0, 1.0]   # 7
    ], dtype=np.float32)

    # Create normals (one for each vertex)
    normals = np.array([
        [0.0, 0.0, -1.0],  # 0: back
        [0.0, 0.0, -1.0],  # 1: back
        [0.0, 0.0, -1.0],  # 2: back
        [0.0, 0.0, -1.0],  # 3: back
        [0.0, 0.0, 1.0],   # 4: front
        [0.0, 0.0, 1.0],   # 5: front
        [0.0, 0.0, 1.0],   # 6: front
        [0.0, 0.0, 1.0]    # 7: front
    ], dtype=np.float32)

    # Create MaterialProperties instances
    cube_material = MaterialProperties(
        name="cube_material",
        diffuse=np.array([1.0, 0.5, 0.31], dtype=np.float32),
        specular=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        shininess=32.0
    )

    secondary_material = MaterialProperties(
        name="secondary_material",
        diffuse=np.array([0.2, 0.8, 0.2], dtype=np.float32),
        specular=np.array([0.3, 0.3, 0.3], dtype=np.float32),
        shininess=16.0
    )

    # Create PhysicsProperties instance (nested Packable)
    physics = PhysicsProperties(
        mass=2.5,
        friction=0.7,
        inertia_tensor=np.eye(3, dtype=np.float32) * 0.1,
        collision_points=np.array([
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32)
    )

    # Create the textured mesh
    mesh = TexturedMesh(
        vertices=vertices,
        indices=indices,
        texture_coords=texture_coords,
        normals=normals,
        physics=physics,
        material_name="cube_material",
        tags=["cube", "example", "test"],
        material_data={
            "cube_material": {
                "diffuse": np.array([1.0, 0.5, 0.31], dtype=np.float32),
                "specular": np.array([0.5, 0.5, 0.5], dtype=np.float32),
                "shininess": np.array([32.0], dtype=np.float32)
            }
        },
        material_colors={
            "cube_material": "#FF7F50",
            "secondary": "#00FF00"
        },
        materials={
            "cube_material": cube_material,
            "secondary_material": secondary_material
        },
    )
    
    return mesh


if __name__ == "__main__":
    output_path = Path(__file__).parent / "textured_mesh.zip"
    
    print(f"Creating TexturedMesh...")
    mesh = create_textured_mesh()
    
    print(f"Mesh info:")
    print(f"  Vertices: {mesh.vertex_count} ({mesh.vertices.shape})")
    print(f"  Indices: {mesh.index_count} ({mesh.indices.shape})")
    print(f"  Texture coords: {mesh.texture_coords.shape}")
    print(f"  Normals: {mesh.normals.shape}")
    print(f"  Material name: {mesh.material_name}")
    print(f"  Tags: {mesh.tags}")
    print(f"  Material colors: {mesh.material_colors}")
    print(f"  Materials: {list(mesh.materials.keys())}")
    print(f"  Physics mass: {mesh.physics.mass}")
    
    print(f"\nSaving to {output_path}...")
    mesh.save_to_zip(output_path)
    print(f"Saved! File size: {output_path.stat().st_size} bytes")
    
    # Verify we can reload it
    print("\nVerifying reload...")
    loaded = TexturedMesh.load_from_zip(output_path)
    print(f"  Reloaded vertices: {loaded.vertex_count}")
    print(f"  Reloaded texture_coords: {loaded.texture_coords.shape}")
    print("Done!")
