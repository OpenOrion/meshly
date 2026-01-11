# meshly

A Python library for efficient 3D mesh serialization using [meshoptimizer](https://github.com/zeux/meshoptimizer) compression.

## Installation

```bash
pip install meshly
```

## Features

### Core Classes

- **`Packable`**: Base class for automatic numpy/JAX array serialization to zip files
- **`Mesh`**: 3D mesh representation extending Packable with meshoptimizer encoding for vertices/indices

### Key Capabilities

- Automatic encoding/decoding of numpy array attributes, including nested dictionaries
- Custom subclasses with additional array fields are automatically serialized
- Enhanced polygon support with `index_sizes` and VTK-compatible `cell_types`
- Mesh markers for boundary conditions, material regions, and geometric features
- Mesh operations: triangulate, optimize, simplify, combine, extract
- Optional JAX array support for GPU-accelerated workflows

## Quick Start

### Basic Mesh Usage

```python
import numpy as np
from meshly import Mesh

# Create a simple triangle mesh
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
], dtype=np.float32)

indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

mesh = Mesh(vertices=vertices, indices=indices)

# Save to zip (uses meshoptimizer compression)
mesh.save_to_zip("mesh.zip")

# Load from zip
loaded = Mesh.load_from_zip("mesh.zip")
print(f"Loaded {loaded.vertex_count} vertices")
```

### Custom Mesh Subclasses

Create custom mesh types with additional array attributes:

```python
from pydantic import Field
from typing import Optional

class TexturedMesh(Mesh):
    """Mesh with texture coordinates and material data."""
    texture_coords: np.ndarray = Field(..., description="UV coordinates")
    normals: Optional[np.ndarray] = Field(None, description="Vertex normals")
    material_name: str = Field("default", description="Material name")
    
    # Nested dictionaries with arrays are automatically handled
    material_data: dict[str, dict[str, np.ndarray]] = Field(default_factory=dict)

# All array fields are automatically encoded/decoded
mesh = TexturedMesh(
    vertices=vertices,
    indices=indices,
    texture_coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
    material_data={
        "wood": {
            "diffuse": np.array([0.8, 0.6, 0.4], dtype=np.float32),
            "specular": np.array([0.2, 0.2, 0.2], dtype=np.float32),
        }
    }
)

mesh.save_to_zip("textured.zip")
loaded = TexturedMesh.load_from_zip("textured.zip")
```

## Architecture

### Class Hierarchy

```
Packable (base class)
├── Mesh (3D mesh with meshoptimizer encoding)
└── Your custom classes...
```

### Metadata Classes

```
PackableMetadata (base metadata)
└── MeshMetadata (extends with MeshSizeInfo)
```

The `Packable` base class provides:
- `save_to_zip()` / `load_from_zip()` - File I/O with compression
- `encode()` - In-memory serialization
- `load_metadata()` - Generic metadata loading with type parameter
- `_create_metadata()` - Override point for custom metadata

### Zip File Structure

```
mesh.zip
├── metadata.json           # PackableMetadata or MeshMetadata
├── mesh/                   # Mesh-specific (meshoptimizer encoded)
│   ├── vertices.bin
│   └── indices.bin
└── arrays/                 # Additional arrays
    ├── texture_coords/
    │   ├── array.bin
    │   └── metadata.json
    └── material_data/wood/diffuse/
        ├── array.bin
        └── metadata.json
```

## Polygon Support

Meshly supports various polygon formats with automatic inference:

```python
# 2D array → uniform polygons
quad_indices = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint32)
mesh = Mesh(vertices=vertices, indices=quad_indices)
print(mesh.index_sizes)  # [4, 4]

# List of lists → mixed polygons
mixed_indices = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11]]
mesh = Mesh(vertices=vertices, indices=mixed_indices)
print(mesh.index_sizes)  # [3, 4, 5]
print(mesh.cell_types)   # [5, 9, 7] (VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON)
```

### VTK Cell Types

| Type | Constant | Vertices |
|------|----------|----------|
| Vertex | 1 | 1 |
| Line | 3 | 2 |
| Triangle | 5 | 3 |
| Quad | 9 | 4 |
| Tetrahedron | 10 | 4 |
| Hexahedron | 12 | 8 |
| Wedge | 13 | 6 |
| Pyramid | 14 | 5 |

## Mesh Markers

Define boundary conditions and regions:

```python
mesh = Mesh(
    vertices=vertices,
    indices=indices,
    markers={
        "inlet": [[0, 1]],           # Line elements
        "outlet": [[2, 3]],
        "wall": [[0, 1, 2], [1, 2, 3]],  # Triangle elements
    },
    dim=2
)

# Access flattened storage
print(mesh.markers)        # Flattened indices
print(mesh.marker_sizes)   # Element sizes
print(mesh.marker_cell_types)  # VTK cell types

# Reconstruct original format
original = mesh.get_reconstructed_markers()
```

## Mesh Operations

### Triangulation

```python
# Convert mixed polygons to triangles
triangulated = mesh.triangulate()
print(f"Original: {mesh.polygon_count} polygons")
print(f"Triangulated: {triangulated.polygon_count} triangles")
```

### Optimization

```python
# Optimize for GPU rendering
optimized = mesh.optimize_vertex_cache()
optimized = mesh.optimize_overdraw()
optimized = mesh.optimize_vertex_fetch()
```

### Simplification

```python
# Reduce triangle count
simplified = mesh.simplify(target_ratio=0.5)  # Keep 50% of triangles
```

### Combining Meshes

```python
combined = Mesh.combine([mesh1, mesh2], marker_names=["part1", "part2"])
```

### Extracting by Marker

```python
boundary_mesh = mesh.extract_by_marker("inlet")
```

### Loading Individual Arrays

Load a single array without loading the entire object (useful for large files):

```python
# Load just the normals array
normals = Mesh.load_array("mesh.zip", "normals")

# Load nested arrays using dotted notation
inlet_indices = Mesh.load_array("mesh.zip", "markers.inlet")
```

## JAX Support

Optional GPU-accelerated arrays:

```python
import jax.numpy as jnp
from meshly import Mesh

# Create with JAX arrays
mesh = Mesh(
    vertices=jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32),
    indices=jnp.array([0, 1, 2], dtype=jnp.uint32)
)

# Load with JAX arrays
mesh = Mesh.load_from_zip("mesh.zip", use_jax=True)
```

## API Reference

### Packable (Base Class)

```python
class Packable(BaseModel):
    def save_to_zip(self, destination, date_time=None) -> None
    @classmethod
    def load_from_zip(cls, source, use_jax=False) -> T
    
    @staticmethod
    def load_array(source, name, use_jax=False) -> Array
    
    def encode(self) -> EncodedData
    
    @classmethod
    def load_metadata(cls, zipf, metadata_cls=PackableMetadata) -> M
    
    def _create_metadata(self, field_data) -> PackableMetadata  # Override point
```

### Mesh

```python
class Mesh(Packable):
    vertices: Array           # Required
    indices: Optional[Array]  # Optional
    index_sizes: Optional[Array]  # Auto-inferred
    cell_types: Optional[Array]   # Auto-inferred
    dim: Optional[int]        # Auto-computed
    markers: Dict[str, Array]
    marker_sizes: Dict[str, Array]
    marker_cell_types: Dict[str, Array]
    
    # Properties
    vertex_count: int
    index_count: int
    polygon_count: int
    is_uniform_polygons: bool
    
    # Methods
    def triangulate(self) -> Mesh
    def optimize_vertex_cache(self) -> Mesh
    def optimize_overdraw(self, threshold=1.05) -> Mesh
    def optimize_vertex_fetch(self) -> Mesh
    def simplify(self, target_ratio=0.25, target_error=0.01) -> Mesh
    def get_polygon_indices(self) -> Array | list
    def get_reconstructed_markers(self) -> Dict[str, List[List[int]]]
    def extract_by_marker(self, marker_name) -> Mesh
    
    @staticmethod
    def combine(meshes, marker_names=None, preserve_markers=True) -> Mesh
    
    def _create_metadata(self, field_data) -> MeshMetadata  # Returns MeshMetadata
```

### Metadata Classes

```python
class PackableMetadata(BaseModel):
    class_name: str
    module_name: str
    field_data: Dict[str, Any]

class MeshSizeInfo(BaseModel):
    vertex_count: int
    vertex_size: int
    index_count: Optional[int]
    index_size: int = 4

class MeshMetadata(PackableMetadata):
    mesh_size: MeshSizeInfo
```

## Examples

See the [examples/](examples/) directory:
- [array_example.ipynb](examples/array_example.ipynb) - Array compression and I/O
- [mesh_example.ipynb](examples/mesh_example.ipynb) - Mesh operations and custom classes
- [markers_example.ipynb](examples/markers_example.ipynb) - Markers and boundary conditions

## Development

```bash
# Run tests
python -m unittest discover tests -v

# Run specific test
python -m unittest tests.test_mesh -v
```

## License

MIT
