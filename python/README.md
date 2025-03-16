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
- Automatic encoding/decoding of numpy array attributes
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

- Save and load meshes to/from ZIP files with `save_to_zip` and `load_from_zip` methods
- Automatic preservation of custom attributes during serialization/deserialization
- Support for storing and loading custom mesh subclasses
- In-memory operations with binary data

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

# Create the textured mesh
mesh = TexturedMesh(
    vertices=vertices,
    indices=indices,
    texture_coords=texture_coords,
    normals=normals,
    material_name="cube_material",
    tags=["cube", "example"]
)

# Optimize the mesh using MeshUtils methods
from meshly import MeshUtils

# Option 1: Use MeshUtils static methods
MeshUtils.optimize_vertex_cache(mesh)
MeshUtils.optimize_overdraw(mesh)
MeshUtils.optimize_vertex_fetch(mesh)
MeshUtils.simplify(mesh, target_ratio=0.8)  # Keep 80% of triangles

# Option 2: Use convenience methods on the Mesh instance
mesh.optimize_vertex_cache()
mesh.optimize_overdraw()
mesh.optimize_vertex_fetch()
mesh.simplify(target_ratio=0.8)  # Keep 80% of triangles

# Encode the mesh (includes all numpy array attributes automatically)
encoded_mesh = mesh.encode()
print(f"Encoded mesh: {len(encoded_mesh.vertices)} bytes for vertices")
print(f"Encoded arrays: {list(encoded_mesh.arrays.keys())}")

# You can also decode the mesh directly
decoded_mesh = MeshUtils.decode(TexturedMesh, encoded_mesh)
print(f"Decoded mesh has {decoded_mesh.vertex_count} vertices")

# Save the mesh to a zip file (uses encode internally)
zip_path = "textured_cube.zip"
mesh.save_to_zip(zip_path)

# Load the mesh from the zip file (uses decode internally)
loaded_mesh = TexturedMesh.load_from_zip(zip_path)

# Use the loaded mesh
print(f"Loaded mesh with {loaded_mesh.vertex_count} vertices")
print(f"Material name: {loaded_mesh.material_name}")
print(f"Tags: {loaded_mesh.tags}")
print(f"Texture coordinates shape: {loaded_mesh.texture_coords.shape}")
print(f"Normals shape: {loaded_mesh.normals.shape}")
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

Benefits of custom mesh subclasses:
- Automatic validation of required fields
- Type checking and conversion (e.g., arrays are automatically converted to the correct dtype)
- Automatic encoding/decoding of all numpy array attributes
- Preservation of non-array attributes during serialization/deserialization

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
mesh.save_to_zip("mesh.zip")

# Load from zip (uses decode internally)
loaded_mesh = Mesh.load_from_zip("mesh.zip")
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
- ZIP compression provides additional size reduction
- Optimized meshes render faster on GPUs
- Simplified meshes maintain visual quality with fewer triangles
- Pydantic models provide efficient validation with minimal overhead
- Automatic handling of array attributes reduces boilerplate code

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