# meshly

A cross-platform library for efficient 3D mesh serialization and transport between Python and TypeScript/JavaScript applications.

## What is meshly?

**meshly** enables you to:

1. **Serialize 3D meshes efficiently** - Compress mesh data (vertices, indices, normals, etc.) using [meshoptimizer](https://github.com/zeux/meshoptimizer) for optimal GPU-friendly storage
2. **Transport meshes from Python to the browser** - Create meshes in Python (NumPy/JAX) and load them in TypeScript/JavaScript for WebGL/THREE.js rendering
3. **Extend with custom data** - Inherit from `Packable` or `Mesh` to add your own array attributes that are automatically serialized

### Use Cases

- **Web-based 3D visualization** - Generate meshes server-side in Python, serve compressed zip files, render in browser with THREE.js
- **Simulation pipelines** - Store simulation results with mesh geometry and field data in a single portable format
- **CAD/CAM workflows** - Exchange mesh data between Python tools and web-based viewers
- **Machine learning** - Serialize mesh datasets with associated feature arrays for training pipelines

## Project Structure

This repository contains two libraries that work together:

### Python Library (`meshly`)

```bash
pip install meshly
```

- Create and manipulate 3D meshes with NumPy/JAX arrays
- Serialize meshes to compressed zip files using meshoptimizer
- Extend with custom array attributes via Pydantic models
- Mesh operations: triangulate, optimize, simplify, combine, extract

### TypeScript Library (`meshly`)

```bash
npm install meshly
# or
pnpm add meshly
```

- Decode Python-generated mesh zip files in the browser
- Convert to THREE.js BufferGeometry for WebGL rendering
- Full TypeScript type definitions

## Quick Example

**Python (server-side):**
```python
import numpy as np
from meshly import Mesh

# Create a mesh
mesh = Mesh(
    vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
    indices=np.array([0, 1, 2], dtype=np.uint32)
)

# Save compressed (uses meshoptimizer)
mesh.save_to_zip("mesh.zip")
```

**TypeScript (browser):**
```typescript
import { Mesh } from 'meshly'
import * as THREE from 'three'

// Load and decode
const response = await fetch('mesh.zip')
const mesh = await Mesh.decode(await response.arrayBuffer())

// Render with THREE.js
const geometry = mesh.toBufferGeometry()
const material = new THREE.MeshStandardMaterial({ color: 0x2194ce })
scene.add(new THREE.Mesh(geometry, material))
```

## Documentation

- [Python README](python/README.md) - Full Python API documentation
- [TypeScript README](typescript/README.md) - Full TypeScript API documentation
- [Python Examples](python/examples/) - Jupyter notebooks with usage examples

## Architecture

### Zip File Format

```
mesh.zip
├── metadata.json       # Class info + non-array fields
├── vertices.bin        # Meshoptimizer-encoded vertices
├── indices.bin         # Meshoptimizer-encoded indices (optional)
└── arrays/             # Standard compressed arrays
    ├── normals/
    │   ├── array.bin
    │   └── metadata.json
    └── ...
```

### Custom Field Encoding

Both Python and TypeScript support custom field encoding via `_get_custom_fields()`:

```python
# Python
@classmethod
def _get_custom_fields(cls) -> Dict[str, CustomFieldConfig]:
    return {
        'vertices': CustomFieldConfig(
            file_name='vertices',
            encode=Mesh._encode_vertices,
            decode=Mesh._decode_vertices,
        ),
    }
```

```typescript
// TypeScript
protected static override getCustomFields(): Record<string, CustomFieldConfig> {
  return {
    vertices: { fileName: 'vertices', decode: Mesh._decodeVertices },
  }
}
```

## License

MIT