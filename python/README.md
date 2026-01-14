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
- **`CustomFieldConfig`**: Configuration for custom field encoding/decoding
- **`ArrayUtils`**: Utility class for encoding/decoding individual arrays
- **`DataHandler`**: Unified interface for reading and writing files or zip archives

### Key Capabilities

- Automatic encoding/decoding of numpy array attributes, including nested dictionaries
- Custom subclasses with additional array fields are automatically serialized
- Custom field encoding via `_get_custom_fields()` override
- Enhanced polygon support with `index_sizes` and VTK-compatible `cell_types`
- Mesh markers for boundary conditions, material regions, and geometric features
- Mesh operations: triangulate, optimize, simplify, combine, extract
- Optional JAX array support for GPU-accelerated workflows

## Quick Start

### Standalone Array Compression

Compress individual arrays without creating a Packable:

```python
import numpy as np
from meshly import ArrayUtils

# Create an array
data = np.random.randn(1000, 3).astype(np.float32)

# Save to zip
ArrayUtils.save_to_zip(data, "array.zip")

# Load from zip
loaded = ArrayUtils.load_from_zip("array.zip")
```

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

# Or use encode/decode for in-memory operations
encoded = mesh.encode()  # Returns bytes
decoded = Mesh.decode(encoded)
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

### Custom Field Encoding

For fields that need special encoding (like meshoptimizer for vertices/indices), override `_get_custom_fields()`:

```python
from meshly import Packable, CustomFieldConfig
from typing import Dict

class CompressedData(Packable):
    """Example with custom field encoding."""
    data: np.ndarray
    
    @staticmethod
    def _encode_data(data: np.ndarray, instance: "CompressedData") -> bytes:
        # Custom encoding logic
        return custom_compress(data)
    
    @staticmethod
    def _decode_data(encoded: bytes, metadata, array_type) -> np.ndarray:
        # Custom decoding logic
        return custom_decompress(encoded)
    
    @classmethod
    def _get_custom_fields(cls) -> Dict[str, CustomFieldConfig]:
        return {
            'data': CustomFieldConfig(
                file_name='data',
                encode=cls._encode_data,
                decode=cls._decode_data,
                optional=False
            ),
        }
```

### Dict of Pydantic BaseModel Objects

You can also use dictionaries containing Pydantic `BaseModel` instances with numpy arrays:

```python
from pydantic import BaseModel, ConfigDict, Field

class MaterialProperties(BaseModel):
    """Material with numpy arrays - automatically serialized."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    diffuse: np.ndarray
    specular: np.ndarray
    shininess: float = 32.0

class SceneMesh(Mesh):
    """Mesh with dict of BaseModel materials."""
    materials: dict[str, MaterialProperties] = Field(default_factory=dict)

# Create mesh with materials
mesh = SceneMesh(
    vertices=vertices,
    indices=indices,
    materials={
        "wood": MaterialProperties(
            name="wood",
            diffuse=np.array([0.8, 0.6, 0.4], dtype=np.float32),
            specular=np.array([0.2, 0.2, 0.2], dtype=np.float32),
        ),
        "metal": MaterialProperties(
            name="metal", 
            diffuse=np.array([0.7, 0.7, 0.8], dtype=np.float32),
            specular=np.array([0.9, 0.9, 0.9], dtype=np.float32),
            shininess=64.0
        )
    }
)

mesh.save_to_zip("scene.zip")
loaded = SceneMesh.load_from_zip("scene.zip")
# loaded.materials["wood"] is a MaterialProperties instance
```

### Nested Packables

Fields that are themselves `Packable` subclasses are automatically handled:

```python
class PhysicsProperties(Packable):
    """Physics data as a nested Packable."""
    mass: float = 1.0
    inertia_tensor: np.ndarray  # 3x3 matrix

class PhysicsMesh(Mesh):
    """Mesh with nested Packable field."""
    physics: Optional[PhysicsProperties] = None

# Nested Packables use their own encode/decode methods
mesh = PhysicsMesh(
    vertices=vertices,
    indices=indices,
    physics=PhysicsProperties(
        mass=2.5,
        inertia_tensor=np.eye(3, dtype=np.float32)
    )
)

mesh.save_to_zip("physics_mesh.zip")
loaded = PhysicsMesh.load_from_zip("physics_mesh.zip")
print(loaded.physics.mass)  # 2.5
```

### Caching Nested Packables

For large projects with shared nested Packables, use caching to deduplicate data using SHA256 content-addressable storage:

```python
from meshly import DataHandler

# Create cache handlers from a directory path
cache_handler = DataHandler.create("/path/to/cache")

# Save with caching - nested Packables stored separately by hash
mesh.save_to_zip("mesh.zip", cache_handler=cache_handler)

# Load with caching - nested Packables loaded from cache
loaded = PhysicsMesh.load_from_zip("mesh.zip", cache_handler=cache_handler)
```

**Deduplication example:**

```python
# Two meshes sharing identical physics properties
shared_physics = PhysicsProperties(mass=1.0, inertia_tensor=np.eye(3))

mesh1 = PhysicsMesh(vertices=v1, indices=i1, physics=shared_physics)
mesh2 = PhysicsMesh(vertices=v2, indices=i2, physics=shared_physics)

# Save both with the same cache handler - physics stored only once!
mesh1.save_to_zip("mesh1.zip", cache_handler=cache_handler)
mesh2.save_to_zip("mesh2.zip", cache_handler=cache_handler)
```

**Custom cache handlers:**

You can implement custom `DataHandler` subclasses for different storage backends:

```python
from meshly.data_handler import DataHandler
from typing import Optional, List
from pathlib import Path

class RedisDataHandler(DataHandler):
    """Data handler backed by Redis."""
    def __init__(self, redis_client, prefix="packable:"):
        super().__init__(source="", rel_path="")
        self.redis = redis_client
        self.prefix = prefix
    
    def read_binary(self, subpath) -> bytes:
        data = self.redis.get(f"{self.prefix}{subpath}")
        if data is None:
            raise FileNotFoundError(f"Key not found: {self.prefix}{subpath}")
        return data
    
    def read_text(self, subpath, encoding="utf-8") -> str:
        return self.read_binary(subpath).decode(encoding)
    
    def list_files(self, subpath="", recursive=False) -> List[Path]:
        raise NotImplementedError("File listing not supported")


class RedisWriteHandler(WriteHandler):
    """Write handler backed by Redis."""
    def __init__(self, redis_client, prefix="packable:"):
        super().__init__(destination="", rel_path="")
        self.redis = redis_client
        self.prefix = prefix
    
    def write_binary(self, subpath, content, executable=False) -> None:
        data = content if isinstance(content, bytes) else content.read()
        self.redis.set(f"{self.prefix}{subpath}", data)
    
    def write_text(self, subpath, content, executable=False) -> None:
        self.redis.set(f"{self.prefix}{subpath}", content.encode('utf-8'))


# Usage with Redis
cache_writer = RedisWriteHandler(redis_client)
cache_reader = RedisReadHandler(redis_client)

mesh.save_to_zip("mesh.zip", cache_handler=cache_writer)
loaded = PhysicsMesh.load_from_zip("mesh.zip", cache_handler=cache_reader)
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
- `encode()` / `decode()` - In-memory serialization to/from bytes
- `convert_to()` - Convert arrays between numpy and JAX
- `_get_custom_fields()` - Override point for custom field encoding
- `load_metadata()` - Generic metadata loading with type parameter

### Zip File Structure

```
mesh.zip
├── metadata.json           # PackableMetadata or MeshMetadata
├── vertices.bin            # Meshoptimizer-encoded (custom field)
├── indices.bin             # Meshoptimizer-encoded (custom field, optional)
└── arrays/                 # Standard arrays (auto-compressed)
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
mesh = Mesh.load_from_zip("mesh.zip", array_type="jax")

# Convert between array types
numpy_mesh = mesh.convert_to("numpy")
jax_mesh = mesh.convert_to("jax")
```

## API Reference

### ArrayUtils

```python
class ArrayUtils:
    # Encode/decode arrays
    @staticmethod
    def encode_array(array: Array) -> EncodedArray
    @staticmethod
    def decode_array(encoded: EncodedArray) -> np.ndarray
    
    # File I/O for single arrays
    @staticmethod
    def save_to_zip(array: Array, destination: PathLike | BytesIO) -> None
    @staticmethod
    def load_from_zip(source: PathLike | BytesIO, array_type=None) -> Array
    
    # Array type utilities
    @staticmethod
    def is_array(obj) -> bool
    @staticmethod
    def detect_array_type(array: Array) -> ArrayType
    @staticmethod
    def convert_array(array: Array, array_type: ArrayType) -> Array
```

### CustomFieldConfig

```python
@dataclass
class CustomFieldConfig(Generic[V, M]):
    file_name: str                                    # File name in zip (without .bin)
    encode: Callable[[V, Any], bytes]                 # (value, instance) -> bytes
    decode: Callable[[bytes, M, Optional[ArrayType]], V]  # (bytes, metadata, array_type) -> value
    optional: bool = False                            # Won't throw if missing
```

### Packable (Base Class)

```python
class Packable(BaseModel):
    # File I/O
    def save_to_zip(self, destination, cache_saver=None) -> None
    @classmethod
    def load_from_zip(cls, source, array_type=None, cache_loader=None) -> T
    
    # In-memory serialization
    def encode(self, cache_saver=None) -> bytes
    @classmethod
    def decode(cls, buf: bytes, array_type=None, cache_loader=None) -> T
    
    # Array conversion
    def convert_to(self, array_type: ArrayType) -> T
    
    # Single array loading
    @staticmethod
    def load_array(source, name, array_type=None) -> Array
    
    # Metadata
    @classmethod
    def load_metadata(cls, handler, metadata_cls=PackableMetadata) -> M
    
    # Custom field encoding (override in subclasses)
    @classmethod
    def _get_custom_fields(cls) -> Dict[str, CustomFieldConfig]
```

### Mesh

```python
class Mesh(Packable):
    # Fields
    vertices: Array           # Required (meshoptimizer encoded)
    indices: Optional[Array]  # Optional (meshoptimizer encoded)
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
    
    # Custom field encoding for meshoptimizer
    @classmethod
    def _get_custom_fields(cls) -> Dict[str, CustomFieldConfig]
```

### Metadata Classes

```python
class PackableMetadata(BaseModel):
    class_name: str
    module_name: str
    field_data: Dict[str, Any]
    packable_refs: Dict[str, str]  # SHA256 hash refs for cached packables

class MeshSizeInfo(BaseModel):
    vertex_count: int
    vertex_size: int
    index_count: Optional[int]
    index_size: int = 4

class MeshMetadata(PackableMetadata):
    mesh_size: MeshSizeInfo
    array_type: ArrayType = "numpy"  # "numpy" or "jax"
```

### Cache Types

```python
# Type aliases for cache callbacks
CacheLoader = Callable[[str], Optional[bytes]]  # hash -> bytes or None
CacheSaver = Callable[[str, bytes], None]       # hash, bytes -> None

# Factory methods to create cache functions from paths
ReadHandler.create_cache_loader(source: PathLike) -> CacheLoader
WriteHandler.create_cache_saver(destination: PathLike) -> CacheSaver
```

### Data Handlers

The `data_handler` module provides abstract interfaces for reading and writing data, supporting both regular files and zip archives.

```python
from meshly import ReadHandler, WriteHandler

# ReadHandler - Abstract base for reading files
class ReadHandler:
    def __init__(self, source: PathLike | BytesIO, rel_path: str = "")
    
    # Abstract methods (implemented by FileReadHandler, ZipReadHandler)
    def read_text(self, subpath: PathLike, encoding: str = "utf-8") -> str
    def read_binary(self, subpath: PathLike) -> bytes
    def list_files(self, subpath: PathLike = "", recursive: bool = False) -> List[Path]
    
    # Navigate to subdirectory
    def to_path(self, rel_path: str) -> ReadHandler
    
    # Factory method - automatically creates FileReadHandler or ZipReadHandler
    @staticmethod
    def create_handler(source: PathLike | BytesIO, rel_path: str = "") -> ReadHandler
    
    # Create cache loader for nested Packables
    @staticmethod
    def create_cache_loader(source: PathLike | BytesIO) -> CacheLoader

# WriteHandler - Abstract base for writing files
class WriteHandler:
    def __init__(self, destination: PathLike | BytesIO, rel_path: str = "")
    
    # Abstract methods (implemented by FileWriteHandler, ZipWriteHandler)
    def write_text(self, subpath: PathLike, content: str, executable: bool = False) -> None
    def write_binary(self, subpath: PathLike, content: bytes | BytesIO, executable: bool = False) -> None
    
    # Navigate to subdirectory
    def to_path(self, rel_path: str) -> WriteHandler
    
    # Factory method - automatically creates FileWriteHandler or ZipWriteHandler
    @staticmethod
    def create_handler(destination: PathLike | BytesIO, rel_path: str = "") -> WriteHandler
    
    # Create cache saver for nested Packables
    @staticmethod
    def create_cache_saver(destination: PathLike | BytesIO) -> CacheSaver
    
    # Close resources (important for ZipWriteHandler)
    def finalize(self) -> None
```

#### Concrete Implementations

```python
# FileReadHandler - Read from filesystem
handler = FileReadHandler("/path/to/directory")
data = handler.read_binary("subdir/file.bin")
files = handler.list_files("subdir", recursive=True)

# ZipReadHandler - Read from zip archives
with open("archive.zip", "rb") as f:
    handler = ZipReadHandler(BytesIO(f.read()))
    metadata = handler.read_text("metadata.json")
    array_data = handler.read_binary("arrays/vertices/array.bin")

# FileWriteHandler - Write to filesystem
handler = FileWriteHandler("/path/to/output")
handler.write_text("config.json", '{"version": 1}')
handler.write_binary("data.bin", compressed_bytes)

# ZipWriteHandler - Write to zip archives
buf = BytesIO()
handler = ZipWriteHandler(buf)
handler.write_text("metadata.json", json_string)
handler.write_binary("data.bin", array_bytes)
handler.finalize()  # Important: closes the zip file
zip_bytes = buf.getvalue()
```

#### Advanced Usage

```python
# Use handlers for custom storage backends
class S3ReadHandler(ReadHandler):
    """Custom handler for reading from S3."""
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix
    
    def read_binary(self, subpath: PathLike) -> bytes:
        key = f"{self.prefix}/{subpath}" if self.prefix else str(subpath)
        return s3_client.get_object(Bucket=self.bucket, Key=key)['Body'].read()
    
    # ... implement other methods

# Deterministic zip output (ZipWriteHandler uses fixed timestamps)
# This ensures identical content produces identical zip files
handler = ZipWriteHandler(buf)
# All files get timestamp (2020, 1, 1, 0, 0, 0) for reproducibility
```

## Examples

See the [examples/](examples/) directory:
- [array_example.ipynb](examples/array_example.ipynb) - Array compression and I/O
- [mesh_example.ipynb](examples/mesh_example.ipynb) - Mesh operations and custom classes
- [markers_example.ipynb](examples/markers_example.ipynb) - Markers and boundary conditions

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_mesh.py -v

# Run tests with coverage
pytest --cov=meshly --cov-report=html
```

## License

MIT
