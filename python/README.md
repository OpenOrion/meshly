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
- **`CachedAssetLoader`**: Asset loader with disk cache for content-addressable storage
- **`LazyModel`**: Lazy proxy that defers asset loading until field access
- **`ResourceRef` / `Resource`**: File reference that serializes by content checksum
- **`SerializedPackableData`**: Result of extracting a Packable (data + assets)
- **`ExtractedAssets`**: Result of extracting assets from multiple values

### Key Capabilities

- Automatic encoding/decoding of numpy array attributes, including nested dictionaries
- Custom subclasses with additional array fields are automatically serialized
- Custom field encoding via `_get_custom_fields()` override
- **Extract/Reconstruct API** for content-addressable storage with deduplication
- **Lazy loading** with `LazyModel` for deferred asset resolution
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

### Extract and Reconstruct API

For content-addressable storage with deduplication, use the `extract()` and `reconstruct()` static methods:

```python
from meshly import Packable

class SimulationResult(Packable):
    """Simulation data with arrays."""
    time: float
    temperature: np.ndarray
    velocity: np.ndarray

result = SimulationResult(
    time=0.5,
    temperature=np.array([300.0, 301.0, 302.0], dtype=np.float32),
    velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
)

# Extract to serializable data + assets
extracted = Packable.extract(result)
# extracted.data = {"time": 0.5, "temperature": {"$ref": "abc123..."}, "velocity": {"$ref": "def456..."}}
# extracted.assets = {"abc123...": <bytes>, "def456...": <bytes>}

# Data is JSON-serializable
import json
json.dumps(extracted.data)  # Works!

# Reconstruct from data + assets (eager loading)
rebuilt = Packable.reconstruct(SimulationResult, extracted.data, extracted.assets)
assert rebuilt.time == 0.5
```

### Resource References for File Handling

Use `ResourceRef` (or `Resource` alias) to include file paths as Pydantic fields that automatically serialize by content checksum:

```python
from meshly import Packable, Resource, ResourceRef
from pydantic import BaseModel

class SimulationCase(BaseModel):
    name: str
    geometry: Resource      # File path that gets serialized by checksum
    config: Resource
    description: str

# Create with file paths
case = SimulationCase(
    name="wind_tunnel",
    geometry="models/wing.stl",
    config="configs/turbulent.json",
    description="High-speed flow analysis"
)

# Extract automatically reads files and computes checksums
extracted = Packable.extract(case)
# extracted.data = {
#     "name": "wind_tunnel",
#     "geometry": {"$ref": "a1b2c3d4", "ext": ".stl"},
#     "config": {"$ref": "e5f6g7h8", "ext": ".json"},
#     "description": "High-speed flow analysis"
# }
# extracted.assets = {
#     "a1b2c3d4": <stl file bytes>,
#     "e5f6g7h8": <json file bytes>
# }

# Reconstruct creates ResourceRef instances with checksums
case2 = Packable.reconstruct(SimulationCase, extracted.data, extracted.assets)
assert isinstance(case2.geometry, ResourceRef)
assert case2.geometry.checksum == "a1b2c3d4"
assert case2.geometry.ext == ".stl"

# ResourceRef can resolve paths from MESHLY_RESOURCE_PATH environment variable
# (useful for remote execution where files are cached by checksum)
path = case2.geometry.resolve_path()  # Finds file by checksum
data = case2.geometry.read_bytes()    # Read file contents
```

### Extracting Assets from Multiple Values

Use `extract_assets()` to extract all assets from multiple values at once - useful for collecting all dependencies before remote execution:

```python
from meshly import Packable
import numpy as np

# Example: extracting assets from function arguments
geometry = ResourceRef(path="model.stl")
initial_temp = np.array([300.0, 301.0, 302.0], dtype=np.float32)
config = {"solver": "cfd", "timesteps": 100}

# Extract all assets in one call
extracted = Packable.extract_assets(geometry, initial_temp, config)
# extracted.assets = {
#     "checksum1": <stl file bytes>,
#     "checksum2": <encoded temperature array>
# }
# extracted.extensions = {
#     "checksum1": ".stl"
# }

# Works with any combination of arrays, Packables, ResourceRefs, BaseModels, dicts, lists
class Pipeline(Packable):
    steps: list[np.ndarray]

pipeline = Pipeline(steps=[np.array([1, 2]), np.array([3, 4])])
user_files = [ResourceRef(path="input1.dat"), ResourceRef(path="input2.dat")]

# Extract from all of them
all_assets = Packable.extract_assets(pipeline, user_files, geometry, initial_temp)
# Automatically deduplicates identical content by checksum
```

### Lazy Loading with Callable Assets

When assets is a callable (or `CachedAssetLoader`), `reconstruct()` returns a `LazyModel` that defers loading:

```python
from meshly import Packable, CachedAssetLoader, DataHandler
from meshly.packable import LazyModel

# Define a fetch function (e.g., from cloud storage)
def fetch_asset(checksum: str) -> bytes:
    return cloud_storage.download(checksum)

# Reconstruct with callable - returns LazyModel
lazy = Packable.reconstruct(SimulationResult, data, fetch_asset)
assert isinstance(lazy, LazyModel)

# No assets loaded yet!
print(lazy.time)         # Primitive field - no fetch needed
print(lazy.temperature)  # NOW temperature asset is fetched
print(lazy.velocity)     # NOW velocity asset is fetched

# Get full Pydantic model
model = lazy.resolve()
assert isinstance(model, SimulationResult)
```

### CachedAssetLoader for Disk Persistence

Use `CachedAssetLoader` to cache fetched assets to disk:

```python
from meshly import CachedAssetLoader, DataHandler

# Create disk cache
cache = DataHandler.create("/path/to/cache")
loader = CachedAssetLoader(fetch_asset, cache)

# First access fetches and caches
lazy = Packable.reconstruct(SimulationResult, data, loader)
temp = lazy.temperature  # Fetches from source, saves to cache

# Subsequent access reads from cache
lazy2 = Packable.reconstruct(SimulationResult, data, loader)
temp2 = lazy2.temperature  # Reads from cache, no fetch!
```

### Deduplication with Extract

Since assets are keyed by SHA256 checksum, identical arrays automatically deduplicate:

```python
# Two results with same temperature data
result1 = SimulationResult(time=0.0, temperature=shared_temp, velocity=v1)
result2 = SimulationResult(time=1.0, temperature=shared_temp, velocity=v2)

extracted1 = Packable.extract(result1)
extracted2 = Packable.extract(result2)

# Same checksum for temperature - deduplicated!
assert extracted1.data["temperature"] == extracted2.data["temperature"]
```

**Note**: Direct Packable fields inside another Packable are not supported. Use `extract()` and `reconstruct()` for composing Packables, or embed Packables inside typed dicts:

```python
from typing import Dict

class Container(Packable):
    name: str
    # Dict of Packables is allowed - extract() handles them
    items: Dict[str, SimulationResult] = Field(default_factory=dict)
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
- `extract()` / `reconstruct()` - Content-addressable storage with `$ref` checksums
- `convert_to()` - Convert arrays between numpy and JAX
- `_get_custom_fields()` - Override point for custom field encoding
- `load_metadata()` - Generic metadata loading with type parameter
- `checksum` - Computed SHA256 checksum property

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
    def save_to_zip(self, destination, cache_handler=None) -> None
    @classmethod
    def load_from_zip(cls, source, array_type=None, cache_handler=None) -> T
    
    # In-memory serialization
    def encode(self, cache_handler=None) -> bytes
    @classmethod
    def decode(cls, buf: bytes, array_type=None, cache_handler=None) -> T
    
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
    
    # Extract/Reconstruct for content-addressable storage
    @staticmethod
    def extract(obj: BaseModel) -> SerializedPackableData
    @staticmethod
    def reconstruct(model_class, data, assets, array_type=None) -> T | LazyModel[T]
    
    # Extract assets from multiple values
    @staticmethod
    def extract_assets(*values) -> ExtractedAssets
    
    # Checksum computation
    @staticmethod
    def compute_checksum(obj) -> str
```

### SerializedPackableData

```python
@dataclass
class SerializedPackableData:
    """Result of extracting a Packable for serialization."""
    data: Dict[str, Any]           # Serializable dict with $ref for arrays/Packables
    assets: Dict[str, bytes]       # Map of checksum -> encoded bytes
```

### ExtractedAssets

```python
@dataclass
class ExtractedAssets:
    """Result of extracting assets from values.
    
    Contains binary assets and their file extensions (for ResourceRefs).
    """
    assets: Dict[str, bytes]       # Map of checksum -> encoded bytes
    extensions: Dict[str, str]     # Map of checksum -> file extension (e.g., '.stl')
```

### LazyModel

```python
class LazyModel(Generic[T]):
    """Lazy proxy for a Pydantic BaseModel that defers asset loading.
    
    Returned by Packable.reconstruct() when assets is a callable or CachedAssetLoader.
    """
    def __getattr__(self, name: str) -> Any  # Loads field on first access
    def resolve(self) -> T                    # Returns fully-loaded model
    def __repr__(self) -> str                 # Shows loaded/pending fields
```

### ResourceRef (Resource)

```python
class ResourceRef(BaseModel):
    """Reference to a file resource that serializes by content checksum.
    
    Use as a Pydantic field type for file paths. When extracted via Packable.extract(),
    files are read and stored by checksum. When reconstructed, files can be retrieved
    from a cache directory via MESHLY_RESOURCE_PATH environment variable.
    """
    path: Optional[str]                  # Original file path (if available)
    ext: Optional[str]                   # File extension (e.g., '.stl')
    
    @cached_property
    def checksum(self) -> Optional[str]  # Computed from file content
    
    def read_bytes(self) -> bytes        # Read file content
    def resolve_path(self) -> Path       # Resolve file path (checks cache)
    
# Alias for convenience
Resource = ResourceRef
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

### DataHandler

The `data_handler` module provides a unified interface for reading and writing data, supporting both regular files and zip archives.

```python
from meshly import DataHandler

# DataHandler - Unified interface for file I/O
class DataHandler:
    def __init__(self, source: PathLike | BytesIO, rel_path: str = "")
    
    # Abstract methods (implemented by FileHandler, ZipHandler)
    def read_text(self, subpath: PathLike, encoding: str = "utf-8") -> str
    def read_binary(self, subpath: PathLike) -> bytes
    def write_text(self, subpath: PathLike, content: str, executable: bool = False) -> None
    def write_binary(self, subpath: PathLike, content: bytes | BytesIO, executable: bool = False) -> None
    def list_files(self, subpath: PathLike = "", recursive: bool = False) -> List[Path]
    def exists(self, subpath: PathLike) -> bool
    def remove_file(self, subpath: PathLike) -> None  # FileHandler only; raises NotImplementedError for ZipHandler
    
    # Navigate to subdirectory
    def to_path(self, rel_path: str) -> DataHandler
    
    # Factory method - automatically creates FileHandler or ZipHandler
    @staticmethod
    def create(source: PathLike | BytesIO, rel_path: str = "") -> DataHandler
    
    # Close resources (important for ZipHandler)
    def finalize(self) -> None
    
    # Context manager support (calls finalize() on exit)
    def __enter__(self) -> DataHandler
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool
```

#### Concrete Implementations

```python
# FileHandler - Read/write from filesystem
handler = DataHandler.create("/path/to/directory")
data = handler.read_binary("subdir/file.bin")
handler.write_text("config.json", '{"version": 1}')
files = handler.list_files("subdir", recursive=True)

# ZipHandler - Read/write from zip archives (using context manager)
buf = BytesIO()
with DataHandler.create(buf) as handler:
    handler.write_text("metadata.json", json_string)
    handler.write_binary("data.bin", array_bytes)
# finalize() is automatically called when exiting the context
zip_bytes = buf.getvalue()

# Reading from existing zip
with open("archive.zip", "rb") as f:
    with DataHandler.create(BytesIO(f.read())) as handler:
        metadata = handler.read_text("metadata.json")
        array_data = handler.read_binary("arrays/vertices/array.bin")
```

#### Advanced Usage

```python
# Use handlers for custom storage backends
class S3DataHandler(DataHandler):
    """Custom handler for reading/writing from S3."""
    def __init__(self, bucket: str, prefix: str = ""):
        super().__init__(source="", rel_path="")
        self.bucket = bucket
        self.prefix = prefix
    
    def read_binary(self, subpath: PathLike) -> bytes:
        key = f"{self.prefix}/{subpath}" if self.prefix else str(subpath)
        return s3_client.get_object(Bucket=self.bucket, Key=key)['Body'].read()
    
    def write_binary(self, subpath: PathLike, content: bytes | BytesIO, executable: bool = False) -> None:
        if isinstance(content, BytesIO):
            content.seek(0)
            content = content.read()
        key = f"{self.prefix}/{subpath}" if self.prefix else str(subpath)
        s3_client.put_object(Bucket=self.bucket, Key=key, Body=content)
    
    def exists(self, subpath: PathLike) -> bool:
        key = f"{self.prefix}/{subpath}" if self.prefix else str(subpath)
        try:
            s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False
    
    # ... implement other methods

# Deterministic zip output (ZipHandler uses fixed timestamps)
# This ensures identical content produces identical zip files
handler = DataHandler.create(buf)
# All files get timestamp (2020, 1, 1, 0, 0, 0) for reproducibility

# Automatic mode switching for ZipHandler
handler = DataHandler.create(BytesIO())
# Handler starts in write mode for empty buffer
handler.write_binary("file1.bin", data1)
# Automatically switches to read mode when needed
content = handler.read_binary("file1.bin")
# Switches back to write mode
handler.write_binary("file2.bin", data2)
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
