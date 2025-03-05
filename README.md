# PyMeshoptimizer

This package provides high-level abstractions and utilities for working with [meshoptimizer](https://github.com/zeux/meshoptimizer), making it easier to use the core functionality in common workflows.

# Installation
`pip install pymeshoptimizer`


## Features

### Mesh Representation

- `Mesh` class: A high-level representation of a 3D mesh with methods for optimization and simplification
- `EncodedMesh` class: A container for encoded mesh data
- Encoding and decoding functions for meshes

### File I/O

- Save and load arrays to/from files
- Save and load meshes to/from ZIP files
- Save and load multiple arrays to/from ZIP files
- Combined storage of meshes and arrays in a single ZIP file

## Usage Examples

### Working with Meshes

```python
from meshoptimizer.export import Mesh, encode_mesh, decode_mesh

# Create a mesh
mesh = Mesh(vertices, indices)

# Optimize the mesh
mesh.optimize_vertex_cache()
mesh.optimize_overdraw()
mesh.optimize_vertex_fetch()

# Simplify the mesh
mesh.simplify(target_ratio=0.5)  # Keep 50% of triangles

# Encode the mesh for storage or transmission
encoded_mesh = mesh.encode()
# or
encoded_mesh = encode_mesh(vertices, indices)

# Decode the mesh
decoded_mesh = Mesh.decode(encoded_mesh)
# or
decoded_vertices, decoded_indices = decode_mesh(encoded_mesh)
```

### File I/O

```python
from meshoptimizer.export import (
    save_array_to_file, load_array_from_file,
    save_mesh_to_zip, load_mesh_from_zip,
    save_arrays_to_zip, load_arrays_from_zip
)

# Save and load arrays to/from files
save_array_to_file(array, "array.bin")
loaded_array = load_array_from_file("array.bin")

# Save and load meshes to/from ZIP files
save_mesh_to_zip(mesh, "mesh.zip")
loaded_mesh = load_mesh_from_zip(Mesh, "mesh.zip")

# Save and load multiple arrays to/from ZIP files
arrays = {"positions": positions, "normals": normals, "colors": colors}
save_arrays_to_zip(arrays, "arrays.zip")
loaded_arrays = load_arrays_from_zip("arrays.zip")
```

### Combined Storage

```python
from meshoptimizer.export import save_combined_data_to_zip, load_combined_data_from_zip

# Save mesh and arrays together
save_combined_data_to_zip(
    encoded_mesh=encoded_mesh,
    encoded_arrays={"normals": encoded_normals, "colors": encoded_colors},
    metadata={"name": "MyMesh", "version": "1.0"},
    zip_path="combined_data.zip"
)

# Load combined data
loaded_mesh, loaded_arrays, loaded_metadata = load_combined_data_from_zip("combined_data.zip")
```

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

1. Update the version number manually in `setup.py`
2. Create a new release on GitHub with a tag matching the version (e.g., `v0.1.0`)
3. The GitHub Actions workflow will automatically build and publish the package to PyPI

Note: Publishing to PyPI requires a PyPI API token stored as a GitHub secret named `PYPI_API_TOKEN`.