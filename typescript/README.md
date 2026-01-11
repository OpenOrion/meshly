# meshly

A TypeScript library for loading Python meshoptimizer zip files and converting to THREE.js geometries.

## Installation

```bash
npm install meshly
# or
pnpm add meshly
```

## Features

- Load meshes from Python meshly zip files
- Decode meshoptimizer-compressed vertex and index buffers
- Convert to THREE.js BufferGeometry
- Support for polygon meshes with automatic triangulation
- Marker extraction for boundary conditions and regions
- Full TypeScript type definitions

## Quick Start

### Load Mesh from Zip

```typescript
import { Mesh } from 'meshly'
import * as THREE from 'three'

// Load mesh from zip file
const response = await fetch('mesh.zip')
const zipData = await response.arrayBuffer()
const mesh = await Mesh.loadFromZip(zipData)

// Convert to THREE.js geometry
const geometry = mesh.toBufferGeometry()
const material = new THREE.MeshStandardMaterial({ color: 0x2194ce })
const threeMesh = new THREE.Mesh(geometry, material)
```

### Work with Mesh Data

```typescript
import { Mesh } from 'meshly'

const mesh = await Mesh.loadFromZip(zipData)

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

### Metadata Interfaces

```typescript
// Base metadata (matches Python PackableMetadata)
interface PackableMetadata {
  class_name: string
  module_name: string
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
├── mesh/                   # Meshoptimizer encoded
│   ├── vertices.bin
│   └── indices.bin
└── arrays/                 # Additional arrays
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
const mesh = await Mesh.loadFromZip(zipData)

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
const mesh = await Mesh.loadFromZip(zipData)

// Check available markers
console.log('Markers:', Object.keys(mesh.markerIndices || {}))

// Extract marker as new mesh
const boundaryMesh = mesh.extractByMarker('inlet')
console.log(`Boundary vertices: ${boundaryMesh.vertices.length / 3}`)

// Convert directly to BufferGeometry
const geometry = mesh.extractMarkerAsBufferGeometry('inlet')
```

## Decoding Encoded Meshes

Decode meshes from the EncodedMesh format:

```typescript
import { Mesh, EncodedMesh } from 'meshly'

// Decode an encoded mesh
const encodedMesh: EncodedMesh = {
  vertices: encodedVertexBuffer,
  indices: encodedIndexBuffer,
  vertex_count: 100,
  vertex_size: 12,  // 3 floats × 4 bytes
  index_count: 300,
  index_size: 4,
  arrays: { /* additional encoded arrays */ }
}

const mesh = Mesh.decode(encodedMesh)
```

## API Reference

### Packable (Base Class)

```typescript
class Packable<TData> {
  constructor(data: TData)
  
  static async loadMetadata<T extends PackableMetadata>(zip: JSZip): Promise<T>
  static async loadFromZip<TData>(zipData: ArrayBuffer | Uint8Array): Promise<Packable<TData>>
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
  static decode(encodedMesh: EncodedMesh): Mesh
  static async loadFromZip(zipData: ArrayBuffer | Uint8Array): Promise<Mesh>
  
  // Marker extraction
  extractByMarker(markerName: string): Mesh
  
  // THREE.js integration
  toBufferGeometry(): THREE.BufferGeometry
  extractMarkerAsBufferGeometry(markerName: string): THREE.BufferGeometry
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
  class_name: string
  module_name: string
  field_data?: Record<string, unknown>
}

// Mesh-specific metadata extending base
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

### Utility Classes

```typescript
// Array encoding/decoding
class ArrayUtils {
  static decodeArray(data: Uint8Array, metadata: ArrayMetadata): Float32Array | Uint32Array
}

// Zip file utilities
class ZipUtils {
  static async loadArrays(zip: JSZip): Promise<Record<string, unknown>>
  static mergeFieldData(data: Record<string, unknown>, fieldData?: Record<string, unknown>): void
}
```

## Python Compatibility

This library is designed to load meshes created by the Python meshly library:

```python
# Python: Save mesh
from meshly import Mesh
mesh = Mesh(vertices=vertices, indices=indices)
mesh.save_to_zip("mesh.zip")
```

```typescript
// TypeScript: Load mesh
import { Mesh } from 'meshly'
const mesh = await Mesh.loadFromZip(zipData)
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
