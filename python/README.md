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
- **`ArrayUtils`**: Utility class for extracting/reconstructing individual arrays
- **`PackableStore`**: File-based store for persistent storage with deduplication
- **`LazyModel`**: Lazy proxy that defers asset loading until field access
- **`Resource`**: Binary data reference that serializes by content checksum
- **`ExtractedPackable`**: Result of extracting a Packable (data + json_schema + assets)
- **`ExtractedArray`**: Result of extracting an array (data + metadata + encoding)
- **`TMesh`**: Type variable for typing custom mesh operations (e.g., `def combine(meshes: List[TMesh]) -> TMesh`)

### Array Type Annotations

- **`Array`**: Generic array type with meshoptimizer compression
- **`VertexBuffer`**: Optimized encoding for 3D vertex data (N × components)
- **`IndexSequence`**: Optimized encoding for mesh indices (1D array)

### Key Capabilities

- Automatic encoding/decoding of numpy array attributes via `Array`, `VertexBuffer`, `IndexSequence` type annotations
- Custom subclasses with additional array fields are automatically serialized
- **Extract/Reconstruct API** for content-addressable storage with deduplication
- **PackableStore** for file-based persistent storage with automatic deduplication
- **JSON Schema** generation with `x-module` for cross-language compatibility
- **Lazy loading** with `LazyModel` for deferred asset resolution
- **Nested Packables** supported - `is_contained` class variable controls serialization strategy
- Enhanced polygon support with `index_sizes` and VTK-compatible `cell_types`
- Mesh markers for boundary conditions, material regions, and geometric features
- Mesh operations: triangulate, optimize, simplify, combine, extract
- **VTK export** via PyVista with `to_pyvista()` and `save_vtk()` methods
- Optional JAX array support for GPU-accelerated workflows

## Quick Start

### Standalone Array Compression

Compress individual arrays without creating a Packable:

```python
import numpy as np
from meshly import ArrayUtils

# Create an array
data = np.random.randn(1000, 3).astype(np.float32)

# Extract to ExtractedArray (with meshoptimizer compression)
extracted = ArrayUtils.extract(data)

# Reconstruct back to numpy array
reconstructed = ArrayUtils.reconstruct(extracted)
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

Create custom mesh types with additional array attributes using `Array` type annotation:

```python
from pydantic import Field
from typing import Optional
from meshly import Mesh, Array

class TexturedMesh(Mesh):
    """Mesh with texture coordinates and material data."""
    texture_coords: Array = Field(..., description="UV coordinates")
    normals: Optional[Array] = Field(None, description="Vertex normals")
    material_name: str = Field("default", description="Material name")
    
    # Nested dictionaries with arrays are automatically handled
    material_data: dict[str, dict[str, Array]] = Field(default_factory=dict)

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

### Array Type Annotations

Use specialized array types for optimized encoding:

```python
from meshly import Packable, Array, VertexBuffer, IndexSequence
from pydantic import Field

class OptimizedMesh(Packable):
    """Mesh with optimized array encodings."""
    
    # VertexBuffer: optimized for 3D vertex data
    vertices: VertexBuffer = Field(..., description="Vertex positions")
    
    # IndexSequence: optimized for mesh indices
    indices: IndexSequence = Field(..., description="Triangle indices")
    
    # Array: generic compression for other data
    normals: Array = Field(..., description="Vertex normals")
    uvs: Array = Field(..., description="Texture coordinates")
```

### Dict of Pydantic BaseModel Objects

You can also use dictionaries containing Pydantic `BaseModel` instances with numpy arrays:

```python
from pydantic import BaseModel, ConfigDict, Field
from meshly import Array

class MaterialProperties(BaseModel):
    """Material with numpy arrays - automatically serialized."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    diffuse: Array
    specular: Array
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
from meshly import Packable, Array

class SimulationResult(Packable):
    """Simulation data with arrays."""
    time: float
    temperature: Array
    velocity: Array

result = SimulationResult(
    time=0.5,
    temperature=np.array([300.0, 301.0, 302.0], dtype=np.float32),
    velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
)

# Extract to serializable data + assets + schema (instance method)
extracted = result.extract()
# extracted.data = {"time": 0.5, "temperature": {"$ref": "abc123...", ...}, ...}
# extracted.json_schema = {...}  # JSON Schema with x-module for class identification
# extracted.assets = {"abc123...": <bytes>, "def456...": <bytes>}

# Data is JSON-serializable
import json
json.dumps(extracted.data)  # Works!

# Reconstruct from extracted (eager loading)
rebuilt = SimulationResult.reconstruct(extracted)
assert rebuilt.time == 0.5
```

### Nested Packables

Packables can contain other Packables. By default, nested Packables are "expanded" (their fields inlined with `$refs` for arrays). Set `is_contained = True` to make a Packable serialize as a single zip blob:

```python
from typing import ClassVar
from meshly import Packable, Mesh, Array

# Self-contained: extracts as a single $ref (zip blob)
class Mesh(Packable):
    is_contained: ClassVar[bool] = True
    vertices: Array
    faces: Array

# Default: fields are expanded inline
class SimulationResult(Packable):
    time: float
    mesh: Mesh  # Will be a single $ref since Mesh.is_contained = True
    temperature: Array  # Will be its own $ref
```

### Resource References for Binary Data

Use `Resource` to include binary data that serializes by checksum:

```python
from meshly import Packable, Resource

class SimulationCase(Packable):
    name: str
    geometry: Resource   # Binary data serialized by checksum
    config: Resource

# Create from file paths
case = SimulationCase(
    name="wind_tunnel",
    geometry=Resource.from_path("models/wing.stl"),
    config=Resource.from_path("configs/turbulent.json"),
)

# Or create from bytes directly
case = SimulationCase(
    name="wind_tunnel",
    geometry=Resource(data=stl_bytes, ext=".stl"),
    config=Resource(data=json_bytes, ext=".json"),
)

# Extract automatically computes checksums (instance method)
extracted = case.extract()
# extracted.metadata.data = {
#     "name": "wind_tunnel",
#     "geometry": {"$ref": "a1b2c3d4", "ext": ".stl"},
#     "config": {"$ref": "e5f6g7h8", "ext": ".json"},
# }
# extracted.assets = {
#     "a1b2c3d4": <gzip-compressed stl bytes>,
#     "e5f6g7h8": <gzip-compressed json bytes>,
# }

# Reconstruct creates Resource instances with data
case2 = Packable.reconstruct(SimulationCase, extracted.metadata.data, extracted.assets)
assert isinstance(case2.geometry, Resource)
assert case2.geometry.data == stl_bytes
assert case2.geometry.ext == ".stl"
```

### Lazy Loading with Callable Assets

When using `is_lazy=True` with `reconstruct()`, returns a `LazyModel` that defers loading:

```python
from meshly import Packable
from meshly.utils.dynamic_model import LazyModel

# Define a fetch function (e.g., from cloud storage)
def fetch_asset(checksum: str) -> bytes:
    return cloud_storage.download(checksum)

# Reconstruct with is_lazy=True - returns LazyModel
lazy = SimulationResult.reconstruct(extracted, assets=fetch_asset, is_lazy=True)
assert isinstance(lazy, LazyModel)

# No assets loaded yet!
print(lazy.time)         # Primitive field - no fetch needed
print(lazy.temperature)  # NOW temperature asset is fetched
print(lazy.velocity)     # NOW velocity asset is fetched

# Get the fully resolved model
model = lazy.resolve(SimulationResult)
```

### Schema-Based Decoding

When decoding using the base `Packable` class (instead of a subclass), the embedded JSON schema is used to build a dynamic model. The `x-base` hint in the schema determines the base class (`Mesh` or `Packable`):

```python
# Decode using base Packable - uses embedded schema
decoded = Packable.decode(encoded_bytes)
# Returns a dynamic model inheriting from Mesh if x-base is "Mesh",
# or from Packable if x-base is "Packable"

# If the original was a Mesh subclass, the decoded object has Mesh methods
if isinstance(decoded, Mesh):
    decoded.triangulate()  # Works!
    decoded.to_pyvista()   # Works!

# Decode using specific class - returns that class type
loaded = SimulationResult.decode(encoded_bytes)
print(loaded.time)  # Access fields normally

# Also works with load_from_zip
loaded = Packable.load_from_zip("result.zip")
print(loaded.time)  # Access fields normally
```

### Deduplication with Extract

Since assets are keyed by SHA256 checksum, identical arrays automatically deduplicate:

```python
# Two results with same temperature data
result1 = SimulationResult(time=0.0, temperature=shared_temp, velocity=v1)
result2 = SimulationResult(time=1.0, temperature=shared_temp, velocity=v2)

extracted1 = result1.extract()
extracted2 = result2.extract()

# Same checksum for temperature - deduplicated!
assert extracted1.metadata.data["temperature"]["$ref"] == extracted2.metadata.data["temperature"]["$ref"]
```

### File-Based Packable Store

Use `PackableStore` for persistent file-based storage with automatic deduplication:

```python
from pathlib import Path
from meshly import PackableStore, Mesh

# Create a store with separate paths for assets and extracted data
store = PackableStore(
    assets_path=Path("/data/assets"),      # Binary blobs stored by checksum
    extracted_path=Path("/data/runs")      # Extracted JSON stored at user keys
)

# Save a packable (auto-generates key from content checksum)
key = mesh.save(store)

# Or save with explicit key for organization
mesh.save(store, "experiment_001/geometry")
result.save(store, "experiment_001/result")

# Load from store
loaded = Mesh.load(store, key)
result = SimulationResult.load(store, "experiment_001/result")

# Lazy loading supported
lazy = Mesh.load(store, key, is_lazy=True)
```

Directory structure:
```
assets_path/
    <checksum1>.bin
    <checksum2>.bin
extracted_path/
    experiment_001/
        geometry.json   # Contains data + json_schema
        result.json
```

## Architecture

### Class Hierarchy

```
Packable (base class)
├── Mesh (3D mesh with meshoptimizer encoding via VertexBuffer/IndexSequence)
└── Your custom classes...
```

### Array Type Annotations

```
Array          → Generic meshoptimizer compression
VertexBuffer   → Optimized for 3D vertex data
IndexSequence  → Optimized for mesh indices
```

The `Packable` base class provides:
- `save_to_zip()` / `load_from_zip()` - File I/O with compression
- `save()` / `load()` - File-based asset store with deduplication
- `extract()` / `encode()` - Instance methods for serialization
- `decode()` / `reconstruct()` - Static methods for deserialization
- `convert_to()` - Convert arrays between numpy and JAX
- `is_contained` - Class variable controlling nested serialization behavior

### Zip File Structure

```
mesh.zip
├── extracted.json          # ExtractedPackable (data + json_schema)
└── assets/                 # Encoded arrays by checksum
    ├── abc123.bin
    └── def456.bin
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

### VTK Export

Export meshes to VTK formats for visualization and analysis (requires pyvista):

```python
# Convert to PyVista UnstructuredGrid
pv_mesh = mesh.to_pyvista()
pv_mesh.plot()

# Save directly to VTK formats
mesh.save_vtk("output.vtu")  # VTK unstructured grid
mesh.save_vtk("output.stl")  # STL format
mesh.save_vtk("output.ply")  # PLY format
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

### Array Type Annotations

```python
from meshly import Array, VertexBuffer, IndexSequence

# Use in Pydantic models for automatic encoding
class MyData(Packable):
    generic_data: Array           # Generic meshoptimizer compression
    vertices: VertexBuffer        # Optimized for 3D vertex data
    indices: IndexSequence        # Optimized for mesh indices
```

### ArrayUtils

```python
class ArrayUtils:
    # Extract/reconstruct arrays
    @staticmethod
    def extract(array: Array, encoding: ArrayEncoding = "array") -> ExtractedArray
    @staticmethod
    def reconstruct(extracted: ExtractedArray, array_type: ArrayType = "numpy") -> np.ndarray
    
    # Zip file I/O
    @staticmethod
    def save_to_zip(array: Array, destination, name: str = "array") -> None
    @staticmethod
    def load_from_zip(source, name: str = "array", array_type: ArrayType = "numpy") -> Array
    
    # Zip file helpers (for use within zipfile contexts)
    @staticmethod
    def save_array(zf: ZipFile, name: str, extracted: ExtractedArray) -> None
    @staticmethod
    def decode(zf: ZipFile, name: str, encoding: ArrayEncoding, array_type: ArrayType = "numpy") -> Array
    
    # Array type utilities
    @staticmethod
    def is_array(obj) -> bool
    @staticmethod
    def detect_array_type(array: Array) -> ArrayType
    @staticmethod
    def convert_array(array: Array, array_type: ArrayType) -> Array
    @staticmethod
    def get_array_encoding(type_hint) -> ArrayEncoding  # Get encoding from type annotation
```

### Packable (Base Class)

```python
class Packable(BaseModel):
    # Class variable for nested serialization behavior
    is_contained: ClassVar[bool] = False  # If True, extracts as single zip blob
    
    # File I/O
    def save_to_zip(self, destination) -> None
    @classmethod
    def load_from_zip(cls, source, array_type="numpy") -> T
    
    # PackableStore I/O
    def save(self, store: PackableStore, key: str = None) -> str  # Returns key
    @classmethod
    def load(cls, store: PackableStore, key: str, array_type="numpy", is_lazy=False) -> T
    
    # Array conversion
    def convert_to(self, array_type: ArrayType) -> T
    
    # Extract/Encode (instance methods)
    def extract(self) -> ExtractedPackable  # Cached for efficiency
    def encode(self) -> bytes  # Calls extract() internally
    def get_checksum(self) -> str  # SHA256 checksum of encoded bytes
    
    # Decode/Reconstruct
    @classmethod
    def decode(cls, buf: bytes, array_type="numpy") -> T  # Decodes and reconstructs
    @classmethod
    def reconstruct(
        cls,
        extracted: ExtractedPackable,       # Contains data, json_schema, and assets
        assets: AssetProvider = None,       # Optional override for asset provider
        array_type: ArrayType = "numpy",
        is_lazy: bool = False,              # If True, returns LazyModel
    ) -> T | LazyModel
    @staticmethod
    def reconstruct_polymorphic(
        model_classes: list[type[T]],       # Candidate classes to match against
        extracted: ExtractedPackable,
        assets: AssetProvider = None,
        array_type: ArrayType = "numpy",
        is_lazy: bool = False,
    ) -> T | LazyModel  # Matches x-module in schema against class list
```

### ExtractedPackable

```python
class ExtractedPackable(BaseModel):
    """Result of extracting a Packable for serialization.
    
    The json_schema contains 'x-module' with the fully qualified class path
    for automatic class resolution during reconstruction, and 'x-base' with
    the base class hint ('Mesh', 'Packable', or 'BaseModel').
    Use model_dump() to get a JSON-serializable dict (assets are excluded).
    """
    data: Dict[str, Any]           # Serializable dict with $ref for arrays/Packables
    json_schema: Optional[Dict[str, Any]]  # JSON Schema with x-module and x-base hints
    assets: Dict[str, bytes]       # Map of checksum -> encoded bytes (excluded from model_dump)
    
    def extract_checksums(self) -> list[str]  # Extract checksums from self.data
```

### ExtractedArray

```python
class ExtractedArray(BaseModel):
    """Result of extracting an array."""
    data: bytes                    # Meshoptimizer-compressed array data
    info: ArrayRefInfo             # Metadata (shape, dtype, itemsize, etc.)
    encoding: ArrayEncoding        # Encoding used ("array", "vertex_buffer", "index_sequence")
```

### LazyModel

```python
class LazyModel:
    """Lazy proxy for a Pydantic BaseModel that defers asset loading.
    
    Returned by Packable.reconstruct() when is_lazy=True.
    """
    def __getattr__(self, name: str) -> Any  # Loads field on first access
    def resolve(self, model_class: type[T]) -> T  # Returns fully-loaded model
    def __repr__(self) -> str                 # Shows loaded/pending fields
```

### Resource

```python
class Resource(BaseModel):
    """Reference to binary data that serializes by content checksum.
    
    Use as a Pydantic field type for binary data. When extracted via .extract(),
    data is gzip-compressed and stored by checksum. When reconstructed, data is
    decompressed and restored.
    """
    data: bytes                       # Binary data (excluded from serialization)
    ext: str = ""                     # File extension (e.g., '.stl')
    name: str = ""                    # Optional name
    
    @property
    def checksum(self) -> str         # Computed from data content
    
    @staticmethod
    def from_path(path: str | Path) -> Resource  # Create from file path
```

### PackableStore

```python
class PackableStore(BaseModel):
    """Configuration for file-based Packable asset storage.
    
    Assets (binary blobs) are stored by their SHA256 checksum, enabling deduplication.
    Extracted packable data is stored at user-specified keys as JSON files.
    """
    
    assets_path: Path              # Directory for binary assets
    extracted_path: Path = None    # Directory for extracted JSON files (defaults to assets_path)
    
    # Path helpers
    def asset_file(self, checksum: str) -> Path       # Get path for binary asset
    def get_extracted_path(self, key: str) -> Path    # Get path for extracted JSON
    
    # Asset operations (binary blobs by checksum)
    def save_asset(self, data: bytes, checksum: str) -> None
    def load_asset(self, checksum: str) -> bytes
    def asset_exists(self, checksum: str) -> bool
    
    # Extracted packable operations (JSON by key)
    def save_extracted(self, key: str, extracted: ExtractedPackable) -> None
    def load_extracted(self, key: str) -> ExtractedPackable
    def extracted_exists(self, key: str) -> bool
```

### Mesh

```python
class Mesh(Packable):
    # Class variable
    is_contained: ClassVar[bool] = True  # Mesh extracts as single zip blob
    
    # Fields with specialized encoding via type annotations
    vertices: VertexBuffer        # Required (meshoptimizer vertex buffer encoding)
    indices: Optional[IndexSequence]  # Optional (meshoptimizer index sequence encoding)
    index_sizes: Optional[Array]  # Auto-inferred
    cell_types: Optional[Array]   # Auto-inferred
    dim: Optional[int]            # Auto-computed
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
    
    # VTK export (requires pyvista)
    def to_pyvista(self) -> pv.UnstructuredGrid  # Convert to PyVista mesh
    def save_vtk(self, path: str | Path) -> None # Save to VTK/STL/PLY formats
    
    @classmethod
    def combine(cls, meshes: List[TMesh], marker_names=None, preserve_markers=True) -> TMesh
```

## Examples

See the [examples/](examples/) directory:
- [array_example.ipynb](examples/array_example.ipynb) - Array compression and I/O
- [mesh_example.ipynb](examples/mesh_example.ipynb) - Mesh operations and custom classes
- [markers_example.ipynb](examples/markers_example.ipynb) - Markers and boundary conditions
- [extract_reconstruct_example.ipynb](examples/extract_reconstruct_example.ipynb) - Extract/reconstruct API

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
