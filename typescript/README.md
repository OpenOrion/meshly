# meshly

A TypeScript library for decoding Python meshly zip files and converting to THREE.js geometries.

## Installation

```bash
npm install meshly
# or
pnpm add meshly
```

## Features

- Decode meshes from Python meshly zip files
- Decode meshoptimizer-compressed vertex and index buffers
- Convert to THREE.js BufferGeometry
- Support for polygon meshes with automatic triangulation
- Marker extraction for boundary conditions and regions
- Custom field decoding via `getCustomFields()` override
- **Reconstruct API** for resolving `$ref` asset references
- **CachedAssetLoader** for disk-cached asset loading
- Full TypeScript type definitions

## Quick Start

### Decode Mesh from Zip

```typescript
import { Mesh } from 'meshly'
import * as THREE from 'three'

// Fetch and decode mesh
const response = await fetch('mesh.zip')
const zipData = await response.arrayBuffer()
const mesh = await Mesh.decode(zipData)

// Convert to THREE.js geometry
const geometry = mesh.toBufferGeometry()
const material = new THREE.MeshStandardMaterial({ color: 0x2194ce })
const threeMesh = new THREE.Mesh(geometry, material)
```

### Work with Mesh Data

```typescript
import { Mesh } from 'meshly'

const mesh = await Mesh.decode(zipData)

// Access mesh properties
console.log(`Vertices: ${mesh.vertices.length / 3}`)
console.log(`Indices: ${mesh.indices?.length}`)
console.log(`Polygons: ${mesh.getPolygonCount()}`)
console.log(`Uniform: ${mesh.isUniformPolygons()}`)

// Access additional arrays
console.log(`Has normals: ${!!mesh.normals}`)
console.log(`Dimension: ${mesh.dim}`)
```

## Architecture

### Class Hierarchy

```
Packable<TData> (base class)
└── Mesh<MeshData> (3D mesh with meshoptimizer decoding)
```

### Custom Field Decoding

Subclasses can override `getCustomFields()` to define custom decoders:

```typescript
protected static override getCustomFields(): Record<string, CustomFieldConfig> {
  return {
    vertices: {
      fileName: 'vertices',
      decode: (data, metadata) => Mesh._decodeVertices(data, metadata),
      optional: false
    },
    indices: {
      fileName: 'indices', 
      decode: (data, metadata) => Mesh._decodeIndices(data, metadata),
      optional: true
    }
  }
}
```

### Metadata Interfaces

```typescript
// Base metadata (matches Python PackableMetadata)
interface PackableMetadata {
  field_data?: Record<string, unknown>
}

// Mesh-specific metadata (extends PackableMetadata)
interface MeshMetadata extends PackableMetadata {
  mesh_size: MeshSize
}

interface MeshSize {
  vertex_count: number
  vertex_size: number
  index_count: number | null
  index_size: number
}
```

### Zip File Structure

```
mesh.zip
├── metadata.json           # MeshMetadata (extends PackableMetadata)
├── vertices.bin            # Meshoptimizer-encoded (custom field)
├── indices.bin             # Meshoptimizer-encoded (custom field, optional)
└── arrays/                 # Standard arrays
    ├── indexSizes/
    │   ├── array.bin
    │   └── metadata.json
    └── normals/
        ├── array.bin
        └── metadata.json
```

## Polygon Support

Meshly handles various polygon types with automatic triangulation for THREE.js:

```typescript
const mesh = await Mesh.decode(zipData)

// Check polygon structure
console.log(`Polygon count: ${mesh.getPolygonCount()}`)
console.log(`Uniform polygons: ${mesh.isUniformPolygons()}`)
console.log(`Index sizes: ${mesh.indexSizes}`)
console.log(`Cell types: ${mesh.cellTypes}`)

// Get polygon indices
const polygons = mesh.getPolygonIndices()

// Convert to BufferGeometry (auto-triangulates)
const geometry = mesh.toBufferGeometry()
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

Extract submeshes by marker name:

```typescript
const mesh = await Mesh.decode(zipData)

// Check available markers
console.log('Markers:', Object.keys(mesh.markerIndices || {}))

// Extract marker as new mesh
const boundaryMesh = mesh.extractByMarker('inlet')
console.log(`Boundary vertices: ${boundaryMesh.vertices.length / 3}`)

// Convert directly to BufferGeometry
const geometry = mesh.extractMarkerAsBufferGeometry('inlet')
```

## Loading Individual Arrays

Load a single array without decoding the entire mesh (useful for large files):

```typescript
// Load just the normals array
const normals = await Mesh.loadArray(zipData, 'normals')

// Load nested arrays using dotted notation
const inletIndices = await Mesh.loadArray(zipData, 'markerIndices.inlet')
```

## API Reference

### DataHandler

```typescript
// DataHandler interface for loading and saving data from various sources
interface DataHandler {
  // Read binary content from a file
  readBinary(path: string): Promise<ArrayBuffer | Uint8Array | undefined>
  // Write binary content to a file (optional)
  writeBinary?(path: string, content: Uint8Array | ArrayBuffer): Promise<void>
  // Check if a file exists (optional)
  exists?(path: string): Promise<boolean>
}

// Asset fetch function type
type AssetFetcher = (checksum: string) => Promise<Uint8Array | ArrayBuffer>

// Asset provider: either a dict of assets or a fetcher function
type AssetProvider = Record<string, Uint8Array | ArrayBuffer> | AssetFetcher
```

### CachedAssetLoader

```typescript
// Asset loader with optional disk cache for persistence
class CachedAssetLoader {
  constructor(
    fetch: AssetFetcher,  // Function that fetches asset bytes by checksum
    cache: DataHandler    // DataHandler for caching fetched assets
  )
  
  // Get asset bytes, checking cache first then fetching if needed
  async getAsset(checksum: string): Promise<Uint8Array | ArrayBuffer>
}
```

### CustomFieldConfig

```typescript
// Custom decoder function type
type CustomDecoder<T, M extends PackableMetadata> = (data: Uint8Array, metadata: M) => T

// Custom field configuration
interface CustomFieldConfig<T = unknown, M extends PackableMetadata = PackableMetadata> {
  fileName: string           // File name in zip (without .bin)
  decode: CustomDecoder<T, M>  // Decoder function
  optional?: boolean         // Won't throw if missing (default: false)
}
```

### Packable (Base Class)

```typescript
class Packable<TData> {
  constructor(data: TData)
  
  // Decode from zip data
  static async decode<TData>(zipData: ArrayBuffer | Uint8Array): Promise<Packable<TData>>
  
  // Reconstruct from extracted data and assets
  static async reconstruct<T>(
    data: Record<string, unknown>,
    assets: AssetProvider | CachedAssetLoader,
    schema?: ReconstructSchema
  ): Promise<T>
  
  // Decode packed array format (metadata + data bytes)
  static _decodePackedArray(packed: Uint8Array | ArrayBuffer): TypedArray
  
  // Load single array
  static async loadArray(zipData: ArrayBuffer | Uint8Array, name: string): Promise<TypedArray>
  
  // Load metadata
  static async loadMetadata<T extends PackableMetadata>(zip: JSZip): Promise<T>
  
  // Custom field configuration (override in subclasses)
  protected static getCustomFields(): Record<string, CustomFieldConfig>
}
```

### Mesh

```typescript
class Mesh<TData extends MeshData = MeshData> extends Packable<TData> {
  // Properties (via declare)
  vertices: Float32Array
  indices?: Uint32Array
  indexSizes?: Uint32Array
  cellTypes?: Uint32Array
  dim?: number
  markerIndices?: Record<string, Uint32Array>
  markerOffsets?: Record<string, Uint32Array>
  markerTypes?: Record<string, Uint8Array>
  
  // Utility methods
  getPolygonCount(): number
  isUniformPolygons(): boolean
  getPolygonIndices(): Uint32Array[] | Uint32Array
  
  // Decoding
  static async decode(zipData: ArrayBuffer | Uint8Array): Promise<Mesh>
  
  // Marker extraction
  extractByMarker(markerName: string): Mesh
  
  // THREE.js integration
  toBufferGeometry(): THREE.BufferGeometry
  extractMarkerAsBufferGeometry(markerName: string): THREE.BufferGeometry
  
  // Custom field configuration for meshoptimizer decoding
  protected static override getCustomFields(): Record<string, CustomFieldConfig<unknown, MeshMetadata>>
}
}
```

### MeshData Interface

```typescript
interface MeshData {
  vertices: Float32Array
  indices?: Uint32Array
  indexSizes?: Uint32Array
  cellTypes?: Uint32Array
  dim?: number
  markerIndices?: Record<string, Uint32Array>
  markerOffsets?: Record<string, Uint32Array>
  markerTypes?: Record<string, Uint8Array>
  markers?: Record<string, number[][]>
}
```

### Metadata Interfaces

```typescript
// Base metadata for all Packable types
interface PackableMetadata {
  field_data?: Record<string, unknown>
}

// Mesh-specific metadata extending base
interface MeshMetadata extends PackableMetadata {
  mesh_size: MeshSize
  array_type?: "numpy" | "jax"  // For Python compatibility
}

interface MeshSize {
  vertex_count: number
  vertex_size: number
  index_count: number | null
  index_size: number
}
```

### Reconstruct Schema Types

```typescript
// Decoder function for Packable types
type PackableDecoder<T> = (data: Uint8Array | ArrayBuffer) => Promise<T> | T

// Schema for a single field
type FieldSchema = 
  | { type: 'array'; element?: FieldSchema }      // TypedArray or Array of items
  | { type: 'packable'; decode: PackableDecoder<unknown> }  // Nested Packable
  | { type: 'dict'; value?: FieldSchema }         // Dict with uniform value type
  | { type: 'object'; fields?: ReconstructSchema } // Object with known field types

// Schema mapping field names to their types
type ReconstructSchema = Record<string, FieldSchema>

// Result of Python's Packable.extract()
interface SerializedPackableData {
  data: Record<string, unknown>      // Serializable dict with $ref references
  assets: Record<string, Uint8Array> // Map of checksum -> encoded bytes
}
```

### Reconstruct Example

```typescript
import { Packable, CachedAssetLoader, ReconstructSchema } from 'meshly'

// Simple case - all $refs are arrays
const result = await Packable.reconstruct(data, assets)

// With nested Packables - define schema for type hints
const schema: ReconstructSchema = {
  mesh: { type: 'packable', decode: (bytes) => Mesh.decode(bytes) },
  snapshots: { 
    type: 'array', 
    element: { type: 'packable', decode: (bytes) => Mesh.decode(bytes) }
  }
}
const result = await Packable.reconstruct(data, assets, schema)

// With CachedAssetLoader for disk caching
const loader = new CachedAssetLoader(
  async (checksum) => fetch(`/api/assets/${checksum}`).then(r => r.arrayBuffer()),
  myDataHandler
)
const result = await Packable.reconstruct(data, loader, schema)
```

### Utility Classes

```typescript
// Array encoding/decoding
class ArrayUtils {
  static decodeArray(data: Uint8Array, metadata: ArrayMetadata): Float32Array | Uint32Array
  static async loadArray(zip: JSZip, name: string): Promise<TypedArray>
}

// Array metadata
interface ArrayMetadata {
  shape: number[]
  dtype: string
  itemsize: number
  array_type?: "numpy" | "jax"  // For Python compatibility
}
```

## Python Compatibility

This library is designed to decode meshes created by the Python meshly library:

```python
# Python: Save mesh
from meshly import Mesh
mesh = Mesh(vertices=vertices, indices=indices)
mesh.save_to_zip("mesh.zip")
```

```typescript
// TypeScript: Decode mesh
import { Mesh } from 'meshly'
const mesh = await Mesh.decode(zipData)
const geometry = mesh.toBufferGeometry()
```

## Development

```bash
# Install dependencies
pnpm install

# Build
pnpm build

# Test
pnpm test
```

## License

MIT
