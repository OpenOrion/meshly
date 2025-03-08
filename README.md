# PyMeshoptimizer

This package provides high-level abstractions and utilities for working with [meshoptimizer](https://github.com/zeux/meshoptimizer), making it easier to use the core functionality in common workflows.

# Installation
```bash
pip install pymeshoptimizer
```

## Dependencies
- NumPy: For efficient array operations
- meshoptimizer: Core mesh optimization functionality
- Pydantic: For data validation and serialization


## Features

### Mesh Representation

- `Mesh` class: A high-level representation of a 3D mesh with methods for optimization and simplification
- `EncodedMesh` class: A container for encoded mesh data
- Encoding and decoding functions for meshes

### Metadata Models

- `ArrayMetadata`: Pydantic model for array metadata validation and serialization
- `MeshMetadata`: Pydantic model for mesh metadata validation and serialization
- `ArraysMetadata`: Container for multiple array metadata entries

### File I/O

- Save and load arrays to/from files
- Save and load meshes to/from ZIP files
- Save and load multiple arrays to/from ZIP files
- Combined storage of meshes and arrays in a single ZIP file
- In-memory operations with binary data

## Usage Example

The following example demonstrates the key functionality of pymeshoptimizer, including mesh optimization, metadata handling, and combined storage:

```python
import numpy as np
from pymeshoptimizer import Mesh, encode_mesh, decode_array, encode_array
from pymeshoptimizer.io import (
    MeshMetadata, ArrayMetadata,
    save_combined_data_to_zip, get_combined_data_as_bytes, load_combined_data_from_zip
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

# Create and optimize a mesh
mesh = Mesh(vertices, indices)
mesh.optimize_vertex_cache()
mesh.optimize_overdraw()
mesh.optimize_vertex_fetch()
mesh.simplify(target_ratio=0.8)  # Keep 80% of triangles

# Create additional data (normals and colors)
normals = np.random.random((mesh.vertex_count, 3)).astype(np.float32)
colors = np.random.random((mesh.vertex_count, 4)).astype(np.float32)

# Encode the mesh and arrays
encoded_mesh = mesh.encode()
encoded_normals = encode_array(normals)
encoded_colors = encode_array(colors)


# Save to file
save_combined_data_to_zip(
    encoded_mesh=encoded_mesh,
    encoded_arrays={"normals": encoded_normals, "colors": encoded_colors},
    metadata={"name": "Cube", "version": "1.0"},
    zip_path="cube_with_data.zip"
)

# Get as bytes (for in-memory operations)
combined_data_bytes = get_combined_data_as_bytes(
    encoded_mesh=encoded_mesh,
    encoded_arrays={"normals": encoded_normals, "colors": encoded_colors},
    metadata={"name": "Cube", "version": "1.0"}
)

# Load from file or bytes
loaded_mesh, loaded_arrays, loaded_metadata = load_combined_data_from_zip("cube_with_data.zip")
# Or: load_combined_data_from_zip(combined_data_bytes)

# Use the loaded data
print(f"Loaded mesh with {loaded_mesh.vertex_count} vertices")
print(f"Loaded arrays: {list(loaded_arrays.keys())}")
print(f"Metadata: {loaded_metadata}")
```

For more detailed examples, see the Jupyter notebooks in the [examples](examples/) directory:
- [array_example.ipynb](examples/array_example.ipynb): Working with arrays, compression, and file I/O
- [mesh_example.ipynb](examples/mesh_example.ipynb): Working with meshes, optimization, and metadata

## Integration with Other Tools

This package is designed to work well with other tools and libraries:

- Use with NumPy for efficient array operations
- Export optimized meshes to game engines
- Store compressed mesh data efficiently
- Process large datasets with minimal memory usage

## Performance Considerations

- Mesh encoding significantly reduces data size (typically 3-5x compression)
- ZIP compression provides additional size reduction
- Optimized meshes render faster on GPUs
- Simplified meshes maintain visual quality with fewer triangles
- Pydantic models provide efficient validation with minimal overhead

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
3. Create a new release on GitHub with a tag matching the version (e.g., `v0.1.2`)
4. The GitHub Actions workflow will automatically build and publish the package to PyPI

Note: Publishing to PyPI requires a PyPI API token stored as a GitHub secret named `PYPI_API_TOKEN`.