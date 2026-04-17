# meshly

A Python library for efficient 3D mesh serialization using [meshoptimizer](https://github.com/zeux/meshoptimizer) compression.

## Installation

```bash
pip install meshly

# With unit conversion support (pint)
pip install meshly[units]
```

## Features

### Core Classes

- **`Packable`**: Base class for automatic numpy/JAX array serialization to zip files
- **`Mesh`**: 3D mesh representation extending Packable with meshoptimizer encoding. Use factory methods: `from_triangles()`, `from_polygons()`, `create()`
- **`ArrayUtils`**: Utility class for extracting/reconstructing individual arrays
- **`Param`**: Unit-aware parameter field for Pydantic models (drop-in replacement for `Field`)
- **`PackableStore`**: File-based store for persistent storage with deduplication
- **`LazyModel`**: Lazy proxy that defers asset loading until field access
- **`Resource`**: Binary data reference that serializes by content checksum
- **`ExtractedPackable`**: Result of extracting a Packable (data + json_schema + assets)
- **`ExtractedArray`**: Result of extracting an array (data + metadata + encoding)
- **`TMesh`**: Type variable for typing custom mesh operations (e.g., `def combine(meshes: List[TMesh]) -> TMesh`)

### Array Type Annotations

- **`Array`**: Generic array type with meshoptimizer compression
- **`IndexSequence`**: Optimized encoding for mesh indices (1D array)
- **`InlineArray`**: Array serialized as inline JSON list (no binary compression)

### Unit-Aware Parameters

- **`Param()`**: Drop-in replacement for `pydantic.Field()` that adds `units`, `shape`, and `example` metadata to the JSON schema
- **`ParamInfo`**: `FieldInfo` subclass backing `Param()` — carries units/shape/example through Pydantic's schema generation
- **`Packable.to_example()`**: Class method that builds an instance from `Param()` example/default values
- **`Packable.with_units()`**: Returns a clone with numeric/array fields converted to `pint.Quantity` objects

### Key Capabilities

- Automatic encoding/decoding of numpy array attributes via `Array`, `IndexSequence`, `InlineArray` type annotations
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

# Reconstruct back to numpy array (preserves original shape)
reconstructed = ArrayUtils.reconstruct(extracted)
print(reconstructed.shape)  # (1000, 3)

# Reconstruct as flat 1-D array (e.g. for GPU vertex buffers)
flat = ArrayUtils.reconstruct(extracted, flat=True)
print(flat.shape)  # (3000,)
```

### Basic Mesh Usage

Use factory methods to create meshes from various input formats:

```python
import numpy as np
from meshly import Mesh

# Vertices shared by all examples
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
], dtype=np.float32)

# Method 1: from_triangles() - Nx3 triangle index array (most common)
triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
mesh = Mesh.from_triangles(vertices=vertices, triangles=triangles)

# Method 2: from_polygons() - list of polygon vertex indices (mixed sizes OK)
polygons = [[0, 1, 2], [0, 2, 3]]  # Can mix triangles, quads, etc.
mesh = Mesh.from_polygons(vertices=vertices, polygons=polygons)

# Method 3: create() - flexible convenience wrapper (auto-detects format)
mesh = Mesh.create(vertices=vertices, indices=triangles)  # 2D array
mesh = Mesh.create(vertices=vertices, indices=polygons)   # List of lists

# Save to zip (uses meshoptimizer compression)
mesh.save_to_zip("mesh.zip")

# Load from zip
loaded = Mesh.load_from_zip("mesh.zip")
print(f"Loaded {loaded.vertex_count} vertices, {loaded.polygon_count} polygons")

# Or use encode/decode for in-memory operations
encoded = mesh.encode()  # Returns bytes
decoded = Mesh.decode(encoded)
```

#### Direct Constructor (Advanced)

The direct constructor requires canonical fields with pre-computed `index_sizes` and `cell_types`:

```python
from meshly.cell_types import VTKCellType

# Direct constructor for maximum performance (no validation overhead)
mesh = Mesh(
    vertices=vertices,
    indices=np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32),  # Flattened 1D
    index_sizes=np.array([3, 3], dtype=np.uint8),           # 2 triangles
    cell_types=np.array([VTKCellType.VTK_TRIANGLE, VTKCellType.VTK_TRIANGLE], dtype=np.uint8),
)
```

> **Note:** Use factory methods (`from_triangles`, `from_polygons`, `create`) for convenience. 
> Use the direct constructor only when you need maximum performance with pre-processed data.

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
from meshly import Packable, Array, IndexSequence, InlineArray
from pydantic import Field

class OptimizedMesh(Packable):
    """Mesh with optimized array encodings."""
    
    # Array: generic compression for vertex/normal data
    vertices: Array = Field(..., description="Vertex positions")
    normals: Array = Field(..., description="Vertex normals")
    uvs: Array = Field(..., description="Texture coordinates")
    
    # IndexSequence: optimized for mesh indices
    indices: IndexSequence = Field(..., description="Triangle indices")
    
    # InlineArray: small arrays serialized inline (no binary compression)
    color: InlineArray = Field(..., description="RGB color")
```

> **Note:** All dtypes are supported. Arrays with non-4-byte dtypes (e.g., `float16`, `int8`, `uint8`) are automatically padded to 4-byte alignment during encoding and unpadded during decoding (meshoptimizer requirement). For best performance, prefer 4-byte aligned dtypes like `float32`, `int32`, or `float64`.

### Unit-Aware Parameters with Param()

`Param()` is a drop-in replacement for `pydantic.Field()` that adds units, shape, and example metadata to the JSON schema. It works on any Pydantic `BaseModel` or `Packable` field, including `InlineArray`:

```python
from meshly import Packable, Param, InlineArray

class Simulation(Packable):
    velocity: InlineArray = Param(units="m/s", example=[30.0, 0, 0], shape=(3,),
                                  description="Flow velocity vector [vx, vy, vz]")
    temperature: float = Param(300.0, units="K", description="Fluid temperature")
    pressure: float = Param(101325.0, units="Pa", description="Outlet pressure")
    name: str = Param("default", units="dimensionless", description="Simulation name")

# Units appear in the JSON schema
schema = Simulation.model_json_schema()
print(schema["properties"]["temperature"])
# {'default': 300.0, 'description': 'Fluid temperature', 'title': 'Temperature',
#  'type': 'number', 'units': 'K'}

# Create from example values
sim = Simulation.to_example()
print(sim.velocity)     # [30.  0.  0.]
print(sim.temperature)  # 300.0

# Convert to pint Quantities (requires `pip install meshly[units]`)
sim_units = sim.with_units()
print(sim_units.velocity)                # [30.0 0.0 0.0] meter / second
print(sim_units.velocity.to("km/h"))     # [108.0 0.0 0.0] kilometer / hour
print(sim_units.temperature.to("degC"))  # 26.85 degree_Celsius

# Convert to SI base units
sim_base = sim.with_units(base_units=True)
print(sim_base.pressure)  # 101325.0 kilogram / meter / second ** 2
```

`Param()` requires either a default value or an `example`:
```python
# With default
velocity: float = Param(10.0, units="m/s")

# With example (no default, field is required)
velocity: float = Param(units="m/s", example=10.0)

# Error: neither default nor example
velocity: float = Param(units="m/s")  # ValueError!
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

# Create a store with a root directory and extracted subdirectory
store = PackableStore(
    root_dir=Path("/data/my_project"),
    extracted_dir="runs"               # Subdirectory for extracted JSON
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
root_dir/
    assets/               # Binary blobs stored by checksum
        <checksum1>.bin
        <checksum2>.bin
    runs/                 # Extracted JSON at user keys
        experiment_001/
            geometry.json
            result.json
```

### In-Memory Caching with PackableCache

Use `PackableCache` for high-performance two-tier caching (in-memory LRU + disk):

```python
from meshly import PackableCache, PackableStore, Mesh
from pathlib import Path

# Create a cache backed by a PackableStore
store = PackableStore(root_dir=Path("/data/cache"), extracted_dir="meshes")
cache = PackableCache(
    store=store,
    decoder=Mesh,           # Type to decode from disk
    prefix="meshes",        # Namespace within the store
    max_memory=10_000,      # Max entries in memory LRU
)

# Single item operations
cache.put("my_mesh", mesh)
loaded = cache.get("my_mesh")  # Memory -> disk -> None

# Batch operations (parallelized disk I/O via ForkPool)
cache.put_many({"mesh1": m1, "mesh2": m2, "mesh3": m3})
found = cache.get_many({"mesh1", "mesh2", "mesh4"})  # Returns dict of found items
```

Lookup order: memory cache → disk → miss. Disk I/O uses `ForkPool` for parallel reads/writes on POSIX systems.

## Architecture

### Class Hierarchy

```
Packable (base class)
├── Mesh (3D mesh with meshoptimizer encoding via Array/IndexSequence)
└── Your custom classes...
```

### Array Type Annotations

```
Array          → Generic meshoptimizer compression
IndexSequence  → Optimized for mesh indices
InlineArray    → Serialized as inline JSON list (no binary $ref)
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
from meshly import Array, IndexSequence, InlineArray

# Use in Pydantic models for automatic encoding
class MyData(Packable):
    generic_data: Array           # Generic meshoptimizer compression
    vertices: Array               # All arrays use meshoptimizer compression
    indices: IndexSequence        # Optimized for mesh indices
    color: InlineArray            # Small arrays as inline JSON (no $ref)
```

### Param

```python
def Param(
    default: Any = ...,
    *,
    units: str,                    # Required: unit string (e.g., "m/s", "Pa", "dimensionless")
    shape: tuple[int, ...] = None, # Optional: expected array shape
    example: Any = None,           # Optional: example value for to_example()
    description: str = None,       # Optional: field description
    # ... all other pydantic.Field kwargs supported (gt, ge, lt, le, etc.)
) -> ParamInfo
```

### ParamInfo

```python
class ParamInfo(FieldInfo):
    """FieldInfo subclass that adds units, shape, and example to the JSON schema.
    
    Works on any Pydantic BaseModel. When used with InlineArray, the units
    are preserved in the JSON schema output via json_schema_extra.
    """
    units: str
    shape: tuple[int, ...] | None
    example: Any
```

### ArrayUtils

```python
class ArrayUtils:
    # Extract/reconstruct arrays
    @staticmethod
    def extract(array: Array, encoding: ArrayEncoding = "array") -> ExtractedArray
    @staticmethod
    def reconstruct(
        extracted: ExtractedArray,
        array_type: ArrayType = "numpy",
        flat: bool = False,           # If True, return flat 1-D array (e.g. for GPU buffers)
    ) -> "np.ndarray | list[np.ndarray]"
    
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
    
    # Param-aware helpers
    @classmethod
    def to_example(cls) -> T         # Build instance from Param() example/default values
    def with_units(self, base_units: bool = False) -> T  # Clone with pint Quantities (requires pint)
    
    # Extract/Encode (instance methods)
    def extract(self) -> ExtractedPackable  # Cached for efficiency
    def encode(self) -> bytes  # Calls extract() internally
    
    # Checksum (final property, cannot be overridden)
    @cached_property
    def checksum(self) -> str  # SHA256 checksum of encoded bytes
    
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
    
    @staticmethod
    def extract_checksums(data: dict) -> list[str]  # Extract $ref checksums from any data dict
```

### ExtractedArray

```python
class ExtractedArray(BaseModel):
    """Result of extracting an array."""
    data: bytes                    # Meshoptimizer-compressed array data
    info: ArrayRefInfo             # Metadata (shape, dtype, itemsize, etc.)
    encoding: ArrayEncoding        # Encoding used ("array" or "index_sequence")
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
    
    root_dir: Path                     # Root directory for all storage
    extracted_dir: str = "runs"        # Subdirectory name for extracted JSON files
    
    # Properties
    @property
    def assets_path(self) -> Path      # root_dir / "assets"
    @property
    def extracted_path(self) -> Path   # root_dir / extracted_dir
    
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

### PackableCache

```python
class PackableCache(Generic[T]):
    """Two-tier LRU cache: in-memory + disk via PackableStore.
    
    Lookup order: memory -> disk -> miss.
    New entries written to both tiers.
    Disk I/O uses ForkPool for parallelism on batch operations.
    """
    
    def __init__(
        self,
        store: PackableStore,     # Underlying disk storage
        decoder: type[T],         # Packable subclass for decoding
        prefix: str = "",         # Key namespace within store
        max_memory: int = 10_000, # Max in-memory entries
    ): ...
    
    # Single item operations
    def get(self, key: str) -> T | None
    def put(self, key: str, value: T) -> None
    
    # Batch operations (parallelized disk I/O)
    def get_many(self, keys: set[str]) -> dict[str, T]
    def put_many(self, items: dict[str, T]) -> None
    
    def clear(self) -> None                # Clear in-memory cache
    def __len__(self) -> int               # Number of in-memory entries
```

### Mesh

```python
class Mesh(Packable):
    # Class variable
    is_contained: ClassVar[bool] = True  # Mesh extracts as single zip blob
    
    # Fields with specialized encoding via type annotations
    vertices: Array               # Required (meshoptimizer array encoding)
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
