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
- **Reconstruct API** for resolving `$ref` asset references
- **Lazy loading** with `LazyModel` for on-demand field resolution
- **Dynamic model building** with `DynamicModelBuilder` from JSON schema
- **Asset caching** with IndexedDB-backed `AssetCache` (browser)
- **Checksum utilities** with `ChecksumUtils` for SHA256 hashing (bytes and dicts)
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

> **Note:** `Mesh.decode` always returns flat `TypedArray` fields (vertices, indices, etc.)
> even for 2-D shapes. This is because mesh data is consumed as interleaved GPU buffers.
> For general `Packable` reconstruction, 2-D arrays are split into per-column arrays by default.

### Work with Mesh Data

```typescript
import { Mesh } from 'meshly'

const mesh = await Mesh.decode(zipData)

// Access mesh properties
console.log(`Vertices: ${mesh.vertices.length / 3}`)
console.log(`Indices: ${mesh.indices?.length}`)
console.log(`Polygons: ${mesh.getPolygonCount()}`)
console.log(`Uniform: ${mesh.isUniformPolygons()}`)

// Access additional arrays (from subclasses)
console.log(`Dimension: ${mesh.dim}`)
```

### Parse Zip (Raw Extraction)

Extract raw data and assets without reconstruction, useful for validation and storage:

```typescript
import { Packable } from 'meshly'

// Parse zip without reconstructing arrays
const extracted = await Packable.parseZip(zipData)

// Access raw data dict with $ref references
console.log(extracted.data)
console.log(extracted.json_schema)

// Get checksums referenced in data
const checksums = Packable.extractChecksums(extracted.data)

// Access binary assets by checksum
for (const [checksum, bytes] of Object.entries(extracted.assets)) {
  console.log(`Asset ${checksum}: ${bytes.length} bytes`)
}
```

### Encode to Zip

Create a packable zip from data and assets:

```typescript
import { Packable, ExtractedPackable } from 'meshly'

// Build an ExtractedPackable
const extracted: ExtractedPackable = {
  data: {
    name: "test",
    vertices: { $ref: "abc123", shape: [100, 3], dtype: "float32", itemsize: 4 }
  },
  json_schema: { type: "object", "x-module": "my.module.MyClass" },
  assets: {
    "abc123": new Uint8Array([/* vertex data */])
  }
}

// Encode to zip bytes
const zipBytes = await Packable.encode(extracted)

// Round-trip: decode back
const decoded = await Packable.parseZip(zipBytes)
```

## Architecture

### Class Hierarchy

```
Packable<TData> (base class)
└── Mesh<MeshData> (3D mesh with meshoptimizer decoding)
```

### Zip File Format

The format stores data as:

```
mesh.zip
├── extracted.json          # ExtractedPackable (data + json_schema)
└── assets/
    ├── {checksum}.bin      # Encoded vertices
    ├── {checksum}.bin      # Encoded indices
    └── ...                 # Other arrays and nested Packables
```

### extracted.json Structure

```json
{
  "data": {
    "vertices": {
      "$ref": "abc123def456",
      "shape": [100, 3],
      "dtype": "float32",
      "itemsize": 4
    },
    "indices": {
      "$ref": "789xyz012abc",
      "shape": [200],
      "dtype": "uint32",
    "itemsize": 4
  },
  "dim": 3,
  "material_name": "default"
  },
  "json_schema": {
    "x-module": "meshly.mesh.Mesh",
    "x-base": "Mesh",
    "properties": {
      "vertices": { "type": "array" },
      "indices": { "type": "index_sequence" }
    }
  }
}
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
console.log('Markers:', Object.keys(mesh.markers || {}))

// Extract marker as new mesh
const boundaryMesh = mesh.extractByMarker('inlet')
console.log(`Boundary vertices: ${boundaryMesh.vertices.length / 3}`)

// Convert directly to BufferGeometry
const geometry = mesh.extractMarkerAsBufferGeometry('inlet')
```

## Lazy Loading

Load fields on-demand for large meshes:

```typescript
import { LazyModel } from 'meshly'

// Create lazy model from zip
const lazy = await LazyModel.fromZip(zipData)

// No arrays loaded yet
console.log(lazy.$loaded) // []

// Load vertices on access
const vertices = await lazy.vertices

// Now it's loaded
console.log(lazy.$loaded) // ['vertices']

// Resolve all fields at once
const resolved = await lazy.$resolve()
```

## Asset Caching

Cache assets for repeated access (browser environments):

```typescript
import { getDefaultCache } from 'meshly'

// Get the default cache and create a cached provider
const cache = await getDefaultCache()
const cachedFetcher = cache.createProvider(async (checksum) => {
  const response = await fetch(`/api/assets/${checksum}`)
  return new Uint8Array(await response.arrayBuffer())
})

// Use with reconstruct
const result = await Packable.reconstruct(data, cachedFetcher, schema)
```

## Checksum Computation

Compute SHA256 checksums for data validation and deduplication:

```typescript
import { ChecksumUtils } from 'meshly'

// Checksum for binary data (returns full 64-char SHA-256 hex string)
const bytes = new Uint8Array([1, 2, 3, 4])
const checksum = await ChecksumUtils.computeBytesChecksum(bytes)
console.log(checksum.length) // 64

// Checksum for a dictionary (returns 16-char truncated hex string)
const data = { name: 'mesh', vertices: { $ref: 'abc123...' } }
const dictChecksum = await ChecksumUtils.computeDictChecksum(data)
console.log(dictChecksum.length) // 16

// Dict checksum with assets included in the hash
const assets = { 'abc123...': new Uint8Array([...]) }
const dictChecksumWithAssets = await ChecksumUtils.computeDictChecksum(data, assets)
```

## Web Worker Offloading

Offload CPU-intensive Packable reconstruction to a background thread:

```typescript
import { PackableWorkerClient } from 'meshly'

// Create a worker client (handles worker lifecycle)
const client = new PackableWorkerClient()

// Reconstruct in background thread
const assets = new Map<string, Uint8Array>()
// ... populate assets ...
const result = await client.reconstruct(data, assets, jsonSchema)

// Decode a full zip in the worker
const decoded = await client.decode(zipArrayBuffer)

// Decode individual arrays
const vertices = await client.decodeArray(vertexBuffer, {
  shape: [100, 3],
  dtype: 'float32',
  itemsize: 4
})

// Clean up when done
client.terminate()
```

### Worker Entry Point

Import the worker entry point in your bundler:

```typescript
// Vite/Rollup
const worker = new Worker(
  new URL('meshly/worker', import.meta.url), 
  { type: 'module' }
)

// Or use the worker directly
import { initPackableWorker } from 'meshly'
// In worker file:
initPackableWorker()
```

## API Reference

### AssetProvider

```typescript
// Asset fetch function type - supports sync/async and nullable returns
// Matches Python: Callable[[str], Union[bytes, None, Awaitable[Optional[bytes]]]]
type AssetFetcher = (checksum: string) => 
    | Uint8Array | ArrayBuffer | null | undefined
    | Promise<Uint8Array | ArrayBuffer | null | undefined>

// Asset provider: either a dict of assets or a fetcher function
type AssetProvider = Record<string, Uint8Array | ArrayBuffer> | AssetFetcher

// Get asset bytes from a provider (dict or callable)
// Matches Python's SerializationUtils.get_asset()
async function getAsset(assets: AssetProvider, checksum: string): Promise<Uint8Array>
```

### Packable (Base Class)

```typescript
class Packable<TData> {
  constructor(data: TData)
  
  // Decode from zip data
  static async decode<TData>(zipData: ArrayBuffer | Uint8Array): Promise<Packable<TData>>
  
  // Encode ExtractedPackable to zip bytes
  static async encode(extracted: ExtractedPackable): Promise<Uint8Array>
  
  // Parse zip without reconstruction (raw access to data/assets)
  static async parseZip(zipData: ArrayBuffer | Uint8Array): Promise<ExtractedPackable>
  
  // Reconstruct from extracted data and assets
  static async reconstruct<T>(
    data: Record<string, unknown>,
    assets: AssetProvider,
    schema?: JsonSchema,
    fieldSchemas?: ReconstructSchema
  ): Promise<T>
  
  // Extract all checksums from data
  static extractChecksums(data: Record<string, unknown>): string[]
}
```

### Mesh

```typescript
class Mesh<TData extends MeshData = MeshData> extends Packable<TData> {
  // Properties
  vertices: Float32Array
  indices?: Uint32Array
  indexSizes?: TypedArray
  cellTypes?: TypedArray
  dim?: number
  markers?: Record<string, Uint32Array>
  markerSizes?: Record<string, TypedArray>
  markerCellTypes?: Record<string, TypedArray>
  
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
}
```

### MeshData Interface

```typescript
interface MeshData {
  vertices: Float32Array
  indices?: Uint32Array
  indexSizes?: TypedArray
  cellTypes?: TypedArray
  dim?: number
  markers?: Record<string, Uint32Array>
  markerSizes?: Record<string, TypedArray>
  markerCellTypes?: Record<string, TypedArray>
}
```

### ArrayRefInfo

```typescript
// Reference to an encoded array in assets
interface ArrayRefInfo {
  $ref: string        // Checksum of the asset
  shape: number[]     // Array shape
  dtype: string       // Data type ('float32', 'uint32', etc.)
  itemsize: number    // Bytes per element
}
```

> **Note:** All dtypes are supported. Arrays with non-4-byte dtypes (e.g., `float16`, `int8`, `uint8`) are automatically padded during encoding and unpadded during decoding. For best performance, prefer 4-byte aligned dtypes like `float32`, `int32`, or `float64`.

### JSON Schema Types

```typescript
// JSON Schema with encoding info
interface JsonSchema {
  properties?: Record<string, JsonSchemaProperty>
  $defs?: Record<string, JsonSchemaProperty>
  // ... standard JSON Schema fields
}

interface JsonSchemaProperty {
  type?: string | 'array' | 'index_sequence'
  items?: JsonSchemaProperty
  properties?: Record<string, JsonSchemaProperty>
  $ref?: string
  // ... standard JSON Schema fields
}

// Array encoding types
type ArrayEncoding = 'array' | 'index_sequence'
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
```

### Reconstruct Example

```typescript
import { Packable, ReconstructSchema, Mesh } from 'meshly'

// Simple case - all $refs are arrays (uses JSON schema for encoding info)
const result = await Packable.reconstruct(data, assets, jsonSchema)

// With nested Packables - define field schemas for type hints
const fieldSchemas: ReconstructSchema = {
  mesh: { type: 'packable', decode: (bytes) => Mesh.decode(bytes) },
  snapshots: { 
    type: 'array', 
    element: { type: 'packable', decode: (bytes) => Mesh.decode(bytes) }
  }
}
const result = await Packable.reconstruct(data, assets, jsonSchema, fieldSchemas)

// With async fetcher
const fetcher = async (checksum: string) => {
  const response = await fetch(`/api/assets/${checksum}`)
  return response.arrayBuffer()
}
const result = await Packable.reconstruct(data, fetcher, jsonSchema)
```

### Utility Classes

```typescript
// Array encoding/decoding
class ArrayUtils {
  // Reconstruct array from ExtractedArray.
  // By default, 2-D+ arrays (e.g. shape [N, 3]) are split into per-column
  // TypedArray[] (one array per column). Pass flat=true to always return a
  // single flat TypedArray (e.g. for GPU vertex buffers).
  static reconstruct(extracted: ExtractedArray, flat?: boolean): TypedArray | TypedArray[]
  
  // Decode array from zip file
  static async decode(zip: JSZip, name: string, encoding?: ArrayEncoding): Promise<TypedArray | TypedArray[]>
}

// Checksum computation utilities
class ChecksumUtils {
  // Full 64-char SHA-256 checksum for bytes
  static async computeBytesChecksum(data: Uint8Array | ArrayBuffer): Promise<string>
  
  // SHA-256 checksum for a data dict, optionally including asset bytes.
  // Returns 16-char truncated hex string. Uses compact JSON with sorted keys
  // to match the Python ChecksumUtils.compute_dict_checksum implementation.
  static async computeDictChecksum(
    data: Record<string, unknown>,
    assets?: Record<string, Uint8Array>
  ): Promise<string>
  
  // Convert object to compact JSON with recursively sorted keys
  static toSortedJson(obj: unknown): string
}


// Extracted array with data and metadata (matches Python's ExtractedArray)
interface ExtractedArray {
  data: Uint8Array
  info: ArrayRefInfo
  encoding?: ArrayEncoding
}

// Array reference info (matches Python's ArrayRefInfo)
interface ArrayRefInfo {
  $ref?: string      // Checksum reference
  shape: number[]
  dtype: string
  itemsize: number   // Must be multiple of 4 (meshoptimizer requirement)
}
```

### ExportConstants

```typescript
// File paths for the new format
const ExportConstants = {
  DATA_FILE: 'metadata/data.json',
  SCHEMA_FILE: 'metadata/schema.json',
  ASSETS_DIR: 'assets',
  ASSET_EXT: '.bin',
  
  // Get asset path from checksum
  assetPath(checksum: string): string,
  
  // Extract checksum from asset path
  checksumFromPath(path: string): string
}
```

### LazyModel

```typescript
// Lazy loading proxy for deferred field resolution
class LazyModel {
  // Create lazy proxy from data and assets
  static create<T>(
    data: Record<string, unknown>,
    assets: AssetProvider,
    schema?: JsonSchema,
    fieldSchemas?: ReconstructSchema
  ): LazyModelProps<T> & T
  
  // Create from zip data
  static async fromZip<T>(zipData: ArrayBuffer | Uint8Array): Promise<LazyModelProps<T> & T>
}

// Special properties exposed by LazyModel proxy
interface LazyModelProps<T> {
  readonly $loaded: string[]   // Loaded field names
  readonly $pending: string[]  // Currently loading fields
  readonly $fields: string[]   // All available field names
  $resolve(): Promise<T>       // Resolve all fields
  $get(name: string): Promise<unknown>  // Get specific field
}
```

### DynamicModelBuilder

```typescript
// Dynamic model building from JSON schema
class DynamicModelBuilder {
  // Instantiate model from schema and data
  static async instantiate<T>(
    schema: JsonSchema,
    data: Record<string, unknown>,
    assets: AssetProvider,
    options?: InstantiateOptions
  ): Promise<DynamicModel<T> | LazyModelProps<T>>
  
  // Load from zip data
  static async fromZip<T>(
    zipData: ArrayBuffer | Uint8Array,
    options?: InstantiateOptions
  ): Promise<DynamicModel<T> | LazyModelProps<T>>
}

interface InstantiateOptions {
  isLazy?: boolean              // Return lazy proxy instead
  fieldSchemas?: ReconstructSchema  // Nested Packable decoders
}
```

### PackableCache

```typescript
// Two-tier LRU cache: in-memory + disk via PackableStore
// Matches Python's PackableCache
class PackableCache {
  constructor(config?: PackableCacheConfig)
  
  // Single item operations
  async get(key: string): Promise<Uint8Array | undefined>
  async put(key: string, value: Uint8Array): Promise<void>
  
  // Batch operations
  async getMany(keys: Set<string>): Promise<Map<string, Uint8Array>>
  async putMany(items: Record<string, Uint8Array>): Promise<void>
  
  // Cache management
  clear(): void           // Clear memory cache (disk not affected)
  get size(): number      // Items in memory cache
  has(key: string): boolean
  
  // Create cached asset fetcher with upstream provider
  createProvider(upstream: AssetFetcher): AssetFetcher
}

interface PackableCacheConfig {
  store?: PackableStore  // Disk storage (null for memory-only)
  prefix?: string        // Key prefix for namespacing
  maxMemory?: number     // Max items in memory (default: 10_000)
}

// Get the default browser cache instance (IndexedDB-backed)
async function getDefaultCache(): Promise<PackableCache>
```

### PackableStore

```typescript
// File-based storage for Packable assets and extracted data
// Matches Python's PackableStore
class PackableStore {
  constructor(config: PackableStoreConfig)
  
  // Paths
  rootDir: string
  assetsPath: string
  extractedPath: string
  
  // Asset operations
  async saveAsset(data: Uint8Array, checksum: string): Promise<void>
  async loadAsset(checksum: string): Promise<Uint8Array>
  async assetExists(checksum: string): Promise<boolean>
  
  // Extracted packable operations
  async saveExtracted(key: string, extracted: ExtractedPackable): Promise<void>
  async loadExtracted(key: string): Promise<ExtractedPackable>
  async extractedExists(key: string): Promise<boolean>
}

interface PackableStoreConfig {
  rootDir: string         // Base directory for storage
  fs?: FileSystemAdapter  // Custom fs implementation (default: node fs/promises)
}

// Create browser-compatible store with IndexedDB via lightning-fs
async function createBrowserStore(config: BrowserStoreConfig): Promise<PackableStore>

interface BrowserStoreConfig {
  rootDir: string    // Virtual path in IndexedDB
  dbName?: string    // IndexedDB database name
}
```

### PackableWorkerClient

```typescript
// Client for offloading Packable reconstruction to a Web Worker
class PackableWorkerClient {
  constructor(workerUrl?: URL)
  
  // Reconstruct data with pre-fetched assets (runs in worker)
  async reconstruct<T>(
    data: Record<string, unknown>,
    assets: Map<string, Uint8Array> | Record<string, Uint8Array>,
    jsonSchema?: JsonSchema
  ): Promise<T>
  
  // Decode a Packable zip blob (runs in worker)
  async decode<T>(zipData: ArrayBuffer): Promise<T>
  
  // Decode an array from binary data (runs in worker, always returns flat)
  async decodeArray(
    data: ArrayBuffer,
    info: ArrayRefInfo
  ): Promise<Float32Array | Int32Array | Uint32Array>
  
  // Decode a mesh zip, triangulate per marker, and compute normals (runs in worker)
  async decodeMesh(
    zipData: ArrayBuffer
  ): Promise<Record<string, { positions: Float32Array; indices: Uint32Array; normals: Float32Array }>>
  
  // Terminate the worker
  terminate(): void
}

// Initialize the worker-side message handler (call in worker file)
function initPackableWorker(): void
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

### Custom Mesh Subclasses

Python subclasses with additional arrays are fully supported:

```python
# Python: Custom mesh with texture coordinates
class TexturedMesh(Mesh):
    texture_coords: Array
    normals: Array
    material_name: str

mesh = TexturedMesh(
    vertices=vertices,
    indices=indices,
    texture_coords=tex_coords,
    normals=normals,
    material_name="wood"
)
mesh.save_to_zip("textured.zip")
```

```typescript
// TypeScript: Access custom fields
interface TexturedMeshData extends MeshData {
  texture_coords: Float32Array
  normals: Float32Array
  material_name: string
}

const mesh = await Mesh.decode(zipData) as Mesh & TexturedMeshData
console.log(mesh.texture_coords)  // Float32Array
console.log(mesh.material_name)   // "wood"
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
