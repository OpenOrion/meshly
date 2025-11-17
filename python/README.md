# meshly

This package provides high-level abstractions and utilities for working with [meshoptimizer](https://github.com/zeux/meshoptimizer), making it easier to use the core functionality in common workflows.

# Installation
```bash
pip install meshly
```

## Features

### Mesh Representation

- `Mesh` class: A Pydantic-based representation of a 3D mesh with methods for optimization and simplification
- Support for custom mesh subclasses with additional attributes
- Automatic encoding/decoding of numpy array attributes, including nested arrays in dictionaries
- Enhanced polygon support with automatic `index_sizes` inference and mixed polygon mesh support
- VTK-compatible `cell_types` with automatic inference from polygon structure
- Mesh copying functionality for creating independent copies
- `EncodedMesh` class: A container for encoded mesh data

### Utility Classes

- `MeshUtils`: Static methods for mesh operations (encoding, decoding, optimization)
- `ArrayUtils`: Static methods for array operations (encoding, decoding)
### Metadata Models

- `ArrayMetadata`: Pydantic model for array metadata validation and serialization
- `MeshSize`: Pydantic model for mesh size information (vertex/index counts and sizes)
- `MeshMetadata`: Pydantic model for storing class, module, and mesh size information
- `EncodedArray`: Container for encoded array data with metadata
- `EncodedArray`: Container for encoded array data with metadata

### File I/O

- Save and load meshes to/from ZIP files with [`MeshUtils.save_to_zip()`](python/meshly/mesh.py:564) and [`MeshUtils.load_from_zip()`](python/meshly/mesh.py:677) methods
- Automatic preservation of custom attributes during serialization/deserialization
- Support for storing and loading custom mesh subclasses
- Nested directory structure for organized array storage in ZIP files
- In-memory operations with binary data

### Advanced Features

- **Nested Array Support**: Automatically encode/decode numpy arrays within nested dictionary structures
- **Flexible Polygon Formats**: Support for triangles, quads, and mixed polygon meshes with automatic `index_sizes` inference
- **Index Sizes Management**: Automatic calculation and validation of polygon vertex counts for complex mesh structures
- **VTK Cell Types**: Automatic inference and validation of VTK-compatible cell type identifiers
- **Marker Support**: Define boundary conditions, material regions, and geometric features with automatic conversion between list and flattened formats
- **Deep Copying**: Create independent mesh copies with the [`copy()`](python/meshly/mesh.py:129) method
- **Enhanced Validation**: Automatic validation and conversion of polygon structures and array data

## Usage Example

The following example demonstrates the key functionality of meshly, including custom mesh subclasses, optimization, and serialization:

```python
import numpy as np
from typing import Optional, List
from pydantic import Field
from meshly import Mesh

# Create a custom mesh subclass with additional attributes
class TexturedMesh(Mesh):
    """
    A mesh with texture coordinates and normals.
    
    This demonstrates how to create a custom Mesh subclass with additional
    numpy array attributes that will be automatically encoded/decoded.
    """
    # Add texture coordinates and normals as additional numpy arrays
    texture_coords: np.ndarray = Field(..., description="Texture coordinates")
    normals: Optional[np.ndarray] = Field(None, description="Vertex normals")
    
    # Add non-array attributes
    material_name: str = Field("default", description="Material name")
    tags: List[str] = Field(default_factory=list, description="Tags for the mesh")
    
    # Dictionary containing nested dictionaries with arrays
    material_data: dict[str, dict[str, np.ndarray]] = Field(
        default_factory=dict,
        description="Nested dictionary structure with arrays"
    )
    
    material_colors: dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary with non-array values"
    )

# Create a simple cube mesh
vertices = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
], dtype=np.float32)

indices = np.array([
    0, 1, 2, 2, 3, 0,  # back face
    1, 5, 6, 6, 2, 1,  # right face
    5, 4, 7, 7, 6, 5,  # front face
    4, 0, 3, 3, 7, 4,  # left face
    3, 2, 6, 6, 7, 3,  # top face
    4, 5, 1, 1, 0, 4   # bottom face
], dtype=np.uint32)

# Create texture coordinates and normals
texture_coords = np.array([
    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
], dtype=np.float32)

normals = np.array([
    [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
], dtype=np.float32)

# Create the textured mesh with nested dictionary data
mesh = TexturedMesh(
    vertices=vertices,
    indices=indices,
    texture_coords=texture_coords,
    normals=normals,
    material_name="cube_material",
    tags=["cube", "example"],
    material_data={
        "cube_material": {
            "diffuse": np.array([1.0, 0.5, 0.31], dtype=np.float32),
            "specular": np.array([0.5, 0.5, 0.5], dtype=np.float32),
            "shininess": np.array([32.0], dtype=np.float32)
        }
    },
    material_colors={
        "cube_material": "#FF7F50"
    }
)

# Optimize the mesh using MeshUtils static methods
from meshly import MeshUtils

# Create optimized copies of the mesh (original mesh is unchanged)
vertex_cache_optimized_mesh = MeshUtils.optimize_vertex_cache(mesh)
overdraw_optimized_mesh = MeshUtils.optimize_overdraw(mesh)
vertex_fetch_optimized_mesh = MeshUtils.optimize_vertex_fetch(mesh)
simplified_mesh = MeshUtils.simplify(mesh, target_ratio=0.8)  # Keep 80% of triangles

# Encode the mesh (includes all numpy array attributes automatically, including nested arrays)
encoded_mesh = MeshUtils.encode(mesh)
print(f"Encoded mesh: {len(encoded_mesh.vertices)} bytes for vertices")
print(f"Encoded arrays: {list(encoded_mesh.arrays.keys())}")

# Decode the mesh directly
decoded_mesh = MeshUtils.decode(TexturedMesh, encoded_mesh)
print(f"Decoded mesh has {decoded_mesh.vertex_count} vertices")

# Save the mesh to a zip file (uses encode internally)
zip_path = "textured_cube.zip"
MeshUtils.save_to_zip(mesh, zip_path)

# Load the mesh from the zip file (uses decode internally)
loaded_mesh = MeshUtils.load_from_zip(TexturedMesh, zip_path)

# Use the loaded mesh
print(f"Loaded mesh with {loaded_mesh.vertex_count} vertices")
print(f"Material name: {loaded_mesh.material_name}")
print(f"Tags: {loaded_mesh.tags}")
print(f"Texture coordinates shape: {loaded_mesh.texture_coords.shape}")
print(f"Normals shape: {loaded_mesh.normals.shape}")
print(f"Material data: {loaded_mesh.material_data}")
print(f"Material colors: {loaded_mesh.material_colors}")

# Copy the mesh to create a new instance
copied_mesh = mesh.copy()
print(f"Copied mesh has {copied_mesh.vertex_count} vertices")
```

## Array Utilities

The package also provides utilities for working with arrays:

```python
import numpy as np
from meshly import ArrayUtils

# Create a numpy array
array = np.random.random((100, 3)).astype(np.float32)

# Encode the array
encoded_array = ArrayUtils.encode_array(array)
print(f"Original size: {array.nbytes} bytes")
print(f"Encoded size: {len(encoded_array.data)} bytes")
print(f"Compression ratio: {array.nbytes / len(encoded_array.data):.2f}x")

# Decode the array
decoded_array = ArrayUtils.decode_array(encoded_array)
print(f"Decoded shape: {decoded_array.shape}")
print(f"Decoded dtype: {decoded_array.dtype}")

# Verify that the decoded array matches the original
np.testing.assert_allclose(array, decoded_array)
```

For more detailed examples, see the Jupyter notebooks in the [examples](examples/) directory:
- [array_example.ipynb](examples/array_example.ipynb): Working with arrays, compression, and file I/O
- [mesh_example.ipynb](examples/mesh_example.ipynb): Working with Pydantic-based meshes, custom subclasses, and serialization
- [markers_example.ipynb](examples/markers_example.ipynb): Working with mesh markers, cell types, and boundary conditions for finite element analysis

## Custom Mesh Subclasses

One of the key features of the Pydantic-based Mesh class is the ability to create custom subclasses with additional attributes:

```python
class SkinnedMesh(Mesh):
    """A mesh with skinning information for animation."""
    # Add bone weights and indices as additional numpy arrays
    bone_weights: np.ndarray = Field(..., description="Bone weights for each vertex")
    bone_indices: np.ndarray = Field(..., description="Bone indices for each vertex")
    
    # Add non-array attributes
    skeleton_name: str = Field("default", description="Skeleton name")
    animation_names: List[str] = Field(default_factory=list, description="Animation names")
```

### Nested Dictionary Support

Meshly now supports numpy arrays within nested dictionary structures. Arrays in nested dictionaries are automatically detected, encoded, and decoded:

```python
class MaterialMesh(Mesh):
    """A mesh with complex material data stored in nested dictionaries."""
    material_data: dict[str, dict[str, np.ndarray]] = Field(
        default_factory=dict,
        description="Nested material properties with array values"
    )
    
    material_metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Non-array material metadata"
    )

# Arrays in nested dictionaries are handled automatically
mesh = MaterialMesh(
    vertices=vertices,
    indices=indices,
    material_data={
        "wood": {
            "diffuse": np.array([0.8, 0.6, 0.4], dtype=np.float32),
            "normal": np.array([0.5, 0.5, 1.0], dtype=np.float32)
        },
        "metal": {
            "diffuse": np.array([0.7, 0.7, 0.7], dtype=np.float32),
            "roughness": np.array([0.1], dtype=np.float32)
        }
    }
)
```

Benefits of custom mesh subclasses:
- Automatic validation of required fields
- Type checking and conversion (e.g., arrays are automatically converted to the correct dtype)
- Automatic encoding/decoding of all numpy array attributes, including nested arrays in dictionaries
- Preservation of non-array attributes during serialization/deserialization
- Support for complex nested data structures with mixed array and non-array content

## Enhanced Polygon Support

Meshly provides enhanced support for different polygon types and automatically infers polygon structure through the `index_sizes` field, with optional `cell_types` for VTK compatibility:

```python
# Triangular mesh (traditional format)
triangular_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

# Quad mesh using 2D numpy array (uniform polygons)
quad_indices = np.array([
    [0, 1, 2, 3],  # First quad
    [4, 5, 6, 7]   # Second quad
], dtype=np.uint32)

# Mixed polygon mesh using list of lists
mixed_indices = [
    [0, 1, 2],        # Triangle
    [3, 4, 5, 6],     # Quad
    [7, 8, 9, 10, 11] # Pentagon
]

# All formats are automatically handled with automatic index_sizes inference
mesh1 = Mesh(vertices=vertices, indices=triangular_indices)
mesh2 = Mesh(vertices=vertices, indices=quad_indices)  # index_sizes: [4, 4]
mesh3 = Mesh(vertices=vertices, indices=mixed_indices)  # index_sizes: [3, 4, 5]

# Access polygon information
print(f"Polygon count: {mesh2.polygon_count}")
print(f"Is uniform: {mesh2.is_uniform_polygons}")
print(f"Index sizes: {mesh2.index_sizes}")  # Shows polygon sizes
print(f"Original structure: {mesh2.get_polygon_indices()}")

# You can also explicitly provide index_sizes for validation
flat_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
explicit_sizes = np.array([3, 4, 2], dtype=np.uint32)  # Triangle, quad, line
mesh4 = Mesh(
    vertices=vertices,
    indices=flat_indices,
    index_sizes=explicit_sizes
)
```

### Index Sizes Field

The [`index_sizes`](python/meshly/mesh.py:120) field stores the number of vertices for each polygon and enables support for mixed polygon meshes:

- **Automatic Inference**: When you provide 2D arrays or lists of lists, `index_sizes` is automatically calculated
- **Validation**: When explicitly provided, it validates against the inferred structure
- **Reconstruction**: Used by [`get_polygon_indices()`](python/meshly/mesh.py:172) to recreate the original polygon structure
- **Storage**: Automatically encoded and stored with the mesh data

```python
# Mixed polygon mesh with explicit index_sizes
vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0.5,0.5,1]], dtype=np.float32)
indices = np.array([0, 1, 2, 3, 4, 1, 2], dtype=np.uint32)  # Quad + triangle
index_sizes = np.array([4, 3], dtype=np.uint32)

mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes)

# Check polygon structure
print(f"Polygon count: {mesh.polygon_count}")  # 2
print(f"Index count: {mesh.index_count}")      # 7
print(f"Is uniform: {mesh.is_uniform_polygons}")  # False
print(f"Polygons: {mesh.get_polygon_indices()}")  # [[0,1,2,3], [4,1,2]]
print(f"Cell types: {mesh.cell_types}")  # [9, 5] (VTK_QUAD, VTK_TRIANGLE)
```

### Cell Types Support

The [`cell_types`](python/meshly/mesh.py:129) field provides VTK-compatible cell type identifiers for each polygon, automatically inferred from `index_sizes`:

```python
# Automatic cell type inference
mixed_indices = [
    [0],              # Vertex
    [0, 1],           # Line
    [0, 1, 2],        # Triangle
    [0, 1, 2, 3],     # Quad
    [0, 1, 2, 3, 4]   # Pentagon
]

mesh = Mesh(vertices=vertices, indices=mixed_indices)
print(f"Cell types: {mesh.cell_types}")  # [1, 3, 5, 9, 14]

# Explicit cell types
explicit_types = [1, 3, 5, 9, 14]  # VTK cell type constants
mesh_explicit = Mesh(
    vertices=vertices,
    indices=mixed_indices,
    cell_types=explicit_types
)

# Common VTK cell types:
# 1: VTK_VERTEX, 3: VTK_LINE, 5: VTK_TRIANGLE, 9: VTK_QUAD
# 10: VTK_TETRA, 12: VTK_HEXAHEDRON, 13: VTK_WEDGE, 14: VTK_PYRAMID
```

## Mesh Markers

Meshly provides comprehensive support for mesh markers, which are essential for defining boundary conditions, material regions, and other geometric features in computational meshes:

### Basic Marker Usage

```python
# Create a 2D mesh with boundary markers
vertices = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]
], dtype=np.float32)

indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

# Define markers using list-of-lists format (automatically converted)
markers = {
    "bottom_edge": [[0, 1]],      # Line marker for bottom boundary
    "right_edge": [[1, 2]],       # Line marker for right boundary
    "top_edge": [[2, 3]],         # Line marker for top boundary
    "left_edge": [[3, 0]],        # Line marker for left boundary
    "center_triangle": [[0, 1, 2]], # Triangle marker for element region
}

mesh = Mesh(
    vertices=vertices,
    indices=indices,
    markers=markers,
    dim=2  # 2D mesh dimension
)

print(f"Markers: {list(mesh.marker_indices.keys())}")
print(f"Boundary elements: {mesh.get_reconstructed_markers()['bottom_edge']}")
```

### Marker Storage and Efficiency

Markers are stored internally using an efficient flattened format that supports variable-sized elements:

```python
# Access flattened marker structure
for name, indices in mesh.marker_indices.items():
    offsets = mesh.marker_offsets[name]
    types = mesh.marker_cell_types[name]
    
    print(f"{name}:")
    print(f"  Flattened indices: {indices}")
    print(f"  Element offsets: {offsets}")
    print(f"  VTK cell types: {types}")

# Reconstruct original list format when needed
original_format = mesh.get_reconstructed_markers()
```

### Advanced Marker Features

```python
# Mixed marker types in a single mesh
mixed_markers = {
    "boundary_vertices": [[0], [2]],           # Vertex markers (VTK type 1)
    "boundary_edges": [[0, 1], [1, 2]],       # Line markers (VTK type 3)
    "material_regions": [[0, 1, 4], [2, 3, 4]], # Triangle markers (VTK type 5)
    "interface_quads": [[1, 2, 5, 4]],         # Quad markers (VTK type 9)
}

advanced_mesh = Mesh(
    vertices=vertices,
    indices=mixed_indices,
    markers=mixed_markers,
    dim=2
)

# Automatic VTK cell type detection
print(f"Marker types detected: {advanced_mesh.marker_cell_types}")
```

### Custom Mesh Classes with Markers

```python
class FiniteElementMesh(Mesh):
    """Mesh with finite element analysis features."""
    
    # Material properties for different regions
    material_properties: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Boundary condition specifications
    boundary_conditions: Dict[str, Dict[str, any]] = Field(default_factory=dict)
    
    def get_boundary_elements(self, boundary_name: str) -> List[List[int]]:
        """Get elements on a specific boundary."""
        return self.get_reconstructed_markers().get(boundary_name, [])

# Create FEM mesh with materials and boundary conditions
fem_mesh = FiniteElementMesh(
    vertices=vertices,
    indices=indices,
    markers={
        "dirichlet_bc": [[0, 3]],    # Fixed displacement boundary
        "neumann_bc": [[1, 2]],      # Applied force boundary
        "material_steel": [[0, 1, 4]], # Steel region
        "material_aluminum": [[2, 3, 4]], # Aluminum region
    },
    material_properties={
        "steel": {"young_modulus": 200e9, "poisson_ratio": 0.3},
        "aluminum": {"young_modulus": 70e9, "poisson_ratio": 0.33},
    },
    boundary_conditions={
        "dirichlet_bc": {"type": "displacement", "value": [0.0, 0.0]},
        "neumann_bc": {"type": "force", "value": [1000.0, 0.0]},
    }
)
```

### Marker Serialization

Markers are fully preserved during mesh encoding/decoding and file I/O:

```python
# Encode mesh with markers
encoded = MeshUtils.encode(fem_mesh)
print(f"Encoded marker arrays: {[k for k in encoded.arrays.keys() if 'marker' in k]}")

# Decode preserves all marker data
decoded = MeshUtils.decode(FiniteElementMesh, encoded)
assert fem_mesh.get_reconstructed_markers() == decoded.get_reconstructed_markers()

# ZIP file serialization also preserves markers
MeshUtils.save_to_zip(fem_mesh, "fem_mesh.zip")
loaded = MeshUtils.load_from_zip(FiniteElementMesh, "fem_mesh.zip")
assert loaded.material_properties == fem_mesh.material_properties
```

Key marker features:
- **Automatic conversion** between list-of-lists and efficient flattened storage
- **VTK compatibility** with standard cell type identifiers
- **Mixed element types** (points, lines, triangles, quads) in a single marker set
- **Type validation** ensures only supported element sizes (1-4 vertices)
- **Full serialization** support with encoding/decoding and ZIP file I/O
- **Easy reconstruction** back to list format for processing algorithms

Common use cases:
- **Finite element analysis**: Boundary conditions and material regions
- **Computational fluid dynamics**: Inlet/outlet boundaries and wall conditions
- **Mesh processing**: Feature identification and region marking
- **Visualization**: Highlighting specific mesh regions or boundaries

## Mesh Copying

Create independent copies of meshes with the [`copy()`](python/meshly/mesh.py:129) method:

```python
# Create a copy of the mesh
copied_mesh = mesh.copy()

# Modifications to the copy don't affect the original
copied_mesh.vertices[0] = [1.0, 1.0, 1.0]
print(f"Original vertices unchanged: {mesh.vertices[0]}")
```

## Encoding and Decoding

The package provides a clean separation between encoding/decoding and file I/O operations:

### Direct Encoding and Decoding

```python
from meshly import Mesh, MeshUtils

# Create a mesh
mesh = Mesh(vertices=vertices, indices=indices)

# Encode the mesh
encoded_mesh = MeshUtils.encode(mesh)

# Decode the mesh
decoded_mesh = MeshUtils.decode(Mesh, encoded_mesh)
```

### File I/O Using Encode and Decode

The `save_to_zip` and `load_from_zip` methods use the `encode` and `decode` functions internally:

```python
# Save to zip (uses encode internally)
MeshUtils.save_to_zip(mesh, "mesh.zip")

# Load from zip (uses decode internally)
loaded_mesh = MeshUtils.load_from_zip(Mesh, "mesh.zip")
```

This separation of concerns makes the code more maintainable and allows for more flexibility in how you work with encoded mesh data.

## Integration with Other Tools

This package is designed to work well with other tools and libraries:

- Use with NumPy for efficient array operations
- Export optimized meshes to game engines
- Store compressed mesh data efficiently
- Process large datasets with minimal memory usage
- Leverage Pydantic's validation and serialization capabilities

## Performance Considerations

- Mesh encoding significantly reduces data size (typically 3-5x compression)
- ZIP compression provides additional size reduction for file storage
- Optimized meshes render faster on GPUs through improved cache performance
- Simplified meshes maintain visual quality with fewer triangles
- Pydantic models provide efficient validation with minimal overhead
- Automatic handling of array attributes reduces boilerplate code
- Deep copying creates independent mesh instances without affecting originals
- Nested array structures are efficiently encoded with dotted path notation
- Polygon structure validation ensures data integrity across different input formats

## Development and Contributing

### Testing

Run the test suite with unittest:

```bash
python -m unittest discover
```

### Continuous Integration

This project uses GitHub Actions for continuous integration:

- Automated tests run on push to main and on pull requests
- Tests run on multiple Python versions (3.8, 3.9, 3.10, 3.11)

### Releasing to PyPI

To release a new version:

1. Update dependencies in `requirements.txt` if needed
2. Update the version number in `setup.py`
3. Create a new release on GitHub with a tag matching the version
4. The GitHub Actions workflow will automatically build and publish the package to PyPI

Note: Publishing to PyPI requires a PyPI API token stored as a GitHub secret named `PYPI_API_TOKEN`.