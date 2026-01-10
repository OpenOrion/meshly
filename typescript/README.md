# Meshly

A TypeScript library for decoding Python meshoptimizer zip files into THREE.js geometries.

## Installation

```bash
npm install meshly
```

## Usage

### Basic Usage

```typescript
import * as THREE from 'three';
import { MeshUtils } from 'meshly';

// Load a mesh from a zip file
async function loadMesh(zipData: ArrayBuffer) {
  const geometry = await MeshUtils.loadZipAsBufferGeometry(zipData, {
    normalize: true,
    computeNormals: true
  });
  
  // Create a THREE.js mesh
  const material = new THREE.MeshStandardMaterial({ color: 0x2194ce });
  const mesh = new THREE.Mesh(geometry, material);
  
  return mesh;
}

// Example: Load a mesh from a URL
async function loadMeshFromURL(url: string) {
  const response = await fetch(url);
  const zipData = await response.arrayBuffer();
  return loadMesh(zipData);
}
```

### Advanced Usage

You can also use the lower-level functions to extract and decode the mesh data:

```typescript
import { MeshUtils } from 'meshly';

async function loadMeshManually(zipData: ArrayBuffer) {
  // Extract the mesh data from the zip file
  const decodedMesh = await MeshUtils.loadMeshFromZip(zipData);
  
  // Convert the decoded mesh to a THREE.js geometry
  const geometry = MeshUtils.convertToBufferGeometry(decodedMesh, {
    normalize: true,
    computeNormals: true
  });
  
  return geometry;
}
```

### Enhanced Polygon Support

Meshly provides enhanced support for different polygon types through `indexSizes` and `cellTypes`, with automatic triangulation for THREE.js compatibility:

```typescript
import { MeshUtils, Mesh } from 'meshly';

// Create a mesh with mixed polygons
const mesh: Mesh = {
  vertices: new Float32Array([
    0, 0, 0,    // 0
    1, 0, 0,    // 1
    1, 1, 0,    // 2
    0, 1, 0,    // 3
    0.5, 0.5, 1 // 4
  ]),
  indices: new Uint32Array([
    0, 1, 2,        // Triangle
    1, 2, 3, 0,     // Quad
    0, 1, 4, 3, 2   // Pentagon
  ]),
  indexSizes: new Uint32Array([3, 4, 5]),
  cellTypes: new Uint32Array([5, 9, 7]), // VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON
  normals: new Float32Array([
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1
  ])
};

// Convert to THREE.js BufferGeometry (automatically triangulates non-triangular polygons)
const geometry = MeshUtils.convertToBufferGeometry(mesh);

// Check polygon information
console.log(`Polygon count: ${MeshUtils.getPolygonCount(mesh)}`); // 3
console.log(`Is uniform: ${MeshUtils.isUniformPolygons(mesh)}`);   // false
console.log(`Cell types: ${MeshUtils.getCellTypes(mesh)}`);       // [5, 9, 7]

// The geometry indices will be triangulated:
// Triangle: unchanged [0, 1, 2]
// Quad: becomes [1, 2, 3, 1, 3, 0]
// Pentagon: becomes [0, 1, 4, 0, 4, 3, 0, 3, 2]
```

### Encoding and Decoding Meshes

You can use the `encode` and `decode` functions to encode and decode meshes directly:

```typescript
import { MeshUtils, Mesh, EncodedMesh } from 'meshly';

// Create a mesh with vertices, indices, and additional arrays
const mesh: Mesh = {
  vertices: new Float32Array([
    -0.5, -0.5, -0.5,
    0.5, -0.5, -0.5,
    0.5, 0.5, -0.5,
    -0.5, 0.5, -0.5
  ]),
  indices: new Uint32Array([0, 1, 2, 2, 3, 0]),
  normals: new Float32Array([
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1
  ])
};

// Encode the mesh
const encodedMesh: EncodedMesh = MeshUtils.encode(mesh);
console.log(`Encoded vertices size: ${encodedMesh.vertices.byteLength} bytes`);
console.log(`Encoded indices size: ${encodedMesh.indices?.byteLength} bytes`);
console.log(`Additional arrays: ${Object.keys(encodedMesh.arrays || {})}`);

// Decode the mesh
const decodedMesh = MeshUtils.decode(encodedMesh);
console.log(`Decoded vertices: ${decodedMesh.vertices.length} elements`);
console.log(`Decoded indices: ${decodedMesh.indices?.length} elements`);
console.log(`Decoded normals: ${decodedMesh.normals?.length} elements`);
```

### VTK Cell Types Support

The library supports VTK-compatible cell type identifiers with automatic inference:

```typescript
import { MeshUtils } from 'meshly';

// Automatic cell type inference from polygon sizes
const indexSizes = new Uint32Array([2, 3, 4, 5, 6, 8]);
const cellTypes = MeshUtils.inferCellTypes(indexSizes);
console.log(cellTypes); // [3, 5, 9, 14, 13, 12]

// Common VTK cell types:
// 1: VTK_VERTEX, 3: VTK_LINE, 5: VTK_TRIANGLE, 9: VTK_QUAD
// 10: VTK_TETRA, 12: VTK_HEXAHEDRON, 13: VTK_WEDGE, 14: VTK_PYRAMID
// 7: VTK_POLYGON (generic)
```

### Marker Auto-Calculation

Meshly automatically calculates `markerOffsets` when only `markerTypes` is provided, eliminating the need to manually specify both:

```typescript
import { MeshUtils } from 'meshly';

// Create a mesh with marker types - offsets calculated automatically
const mesh = {
  vertices: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
  indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  markerIndices: {
    "boundary": new Uint32Array([0, 1, 2, 3]) // Flattened marker indices
  },
  markerTypes: {
    "boundary": new Uint8Array([3, 3]) // Two lines (VTK_LINE = 3)
  }
  // markerOffsets automatically calculated as [0, 2] during decode
};

// Encode and decode to trigger auto-calculation
const encoded = MeshUtils.encode(mesh);
const decoded = MeshUtils.decode(encoded);

console.log(decoded.markerOffsets!.boundary); // Uint32Array([0, 2])

// Works with mixed cell types too
const mixedMesh = {
  vertices: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 1]),
  indices: new Uint32Array([0, 1, 2]),
  markerIndices: {
    "mixed": new Uint32Array([0, 1, 2, 0, 1, 4]) // vertex + line + triangle
  },
  markerTypes: {
    "mixed": new Uint8Array([1, 3, 5]) // VTK_VERTEX, VTK_LINE, VTK_TRIANGLE
  }
  // markerOffsets automatically calculated as [0, 1, 3]
};
```

The auto-calculation uses the VTK cell type mapping through utility functions:

```typescript
// Direct utility usage
const cellTypes = new Uint8Array([1, 3, 5]); // vertex, line, triangle
const sizes = MeshUtils.inferSizesFromCellTypes(cellTypes);        // [1, 2, 3]
const offsets = MeshUtils.sizesToOffsets(sizes);                   // [0, 1, 3]

// VTK cell type to size mapping:
// VTK_VERTEX (1) → 1 vertex, VTK_LINE (3) → 2 vertices
// VTK_TRIANGLE (5) → 3 vertices, VTK_QUAD (9) → 4 vertices
// VTK_TETRA (10) → 4 vertices, VTK_PYRAMID (14) → 5 vertices
// VTK_WEDGE (13) → 6 vertices, VTK_HEXAHEDRON (12) → 8 vertices
```

### Extracting Meshes by Marker

You can extract submeshes by marker name and convert them to THREE.js geometries:

```typescript
import { MeshUtils, Mesh } from 'meshly';

// Load a mesh with markers
const zipData = await fetch('mesh-with-markers.zip').then(r => r.arrayBuffer());
const mesh = await MeshUtils.loadMeshFromZip(zipData);

console.log('Available markers:', Object.keys(mesh.markerIndices!));

// Extract a specific marker as a new mesh
const boundaryMesh = MeshUtils.extractByMarker(mesh, 'boundary');
console.log(`Boundary mesh has ${boundaryMesh.vertices.length / 3} vertices`);

// Convert directly to BufferGeometry
const boundaryGeometry = MeshUtils.extractMarkerAsBufferGeometry(mesh, 'boundary', {
  computeNormals: true
});

// Use in THREE.js
const boundaryMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000 });
const boundaryMeshObj = new THREE.Mesh(boundaryGeometry, boundaryMaterial);

// Extract multiple markers
const markers = ['inlet', 'outlet', 'walls'];
const geometries = markers.map(name => 
  MeshUtils.extractMarkerAsBufferGeometry(mesh, name)
);
```

### Array Utilities

The library also provides utilities for working with arrays:

```typescript
import { ArrayUtils } from 'meshly';

// Encode a Float32Array
const array = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
const encoded = ArrayUtils.encodeArray(array);
console.log(`Original size: ${array.byteLength} bytes`);
console.log(`Encoded size: ${encoded.data.byteLength} bytes`);

// Decode an array
const decoded = ArrayUtils.decodeArray(encoded.data, {
  shape: encoded.shape,
  dtype: encoded.dtype,
  itemsize: encoded.itemsize
});
console.log(`Decoded array length: ${decoded.length}`);
```

### Snapshot Utilities

Load simulation snapshots saved from Python containing multiple fields of time-series data:

```typescript
import { SnapshotUtils } from 'meshly';

// Load a snapshot from a zip file
async function loadSnapshot(zipData: ArrayBuffer) {
  // Get metadata without loading field data
  const metadata = await SnapshotUtils.loadMetadata(zipData);
  console.log(`Snapshot at t=${metadata.time}`);
  console.log(`Available fields: ${Object.keys(metadata.fields)}`);
  
  // Load a specific field
  const velocity = await SnapshotUtils.loadField(zipData, "velocity");
  console.log(`Velocity shape: ${velocity.shape}`);
  console.log(`Velocity units: ${velocity.units}`);
  console.log(`Velocity data: ${velocity.data.length} elements`);
  
  // Load multiple fields
  const fields = await SnapshotUtils.loadFields(zipData, ["velocity", "pressure"]);
  
  // Or load entire snapshot
  const snapshot = await SnapshotUtils.loadFromZip(zipData);
  console.log(`Time: ${snapshot.time}`);
  console.log(`Fields loaded: ${Object.keys(snapshot.fields)}`);
  
  return snapshot;
}

// Quick helpers
const time = await SnapshotUtils.getTime(zipData);
const fieldNames = await SnapshotUtils.getFieldNames(zipData);
```

#### Snapshot Zip Structure

Snapshots are stored as compressed zip files:

```
snapshot.zip
├── metadata.json      # Contains time and field metadata
└── fields/
    ├── velocity.bin   # meshopt-encoded array
    ├── pressure.bin   # meshopt-encoded array
    └── temperature.bin
```

## API Reference

### MeshUtils

#### `MeshUtils.loadZipAsBufferGeometry(zipData: ArrayBuffer, options?: DecodeMeshOptions): Promise<THREE.BufferGeometry>`

Loads a mesh from a zip file and converts it to a THREE.js BufferGeometry.

##### Parameters

- `zipData`: The zip file data as an ArrayBuffer
- `options`: Options for decoding the mesh
  - `normalize`: Whether to normalize the mesh to fit within a unit cube (default: `false`)
  - `computeNormals`: Whether to compute normals if they don't exist (default: `true`)

##### Returns

A Promise that resolves to a THREE.js BufferGeometry.

#### `MeshUtils.loadMeshFromZip(zipData: ArrayBuffer): Promise<Mesh>`

Extracts and decodes a mesh from a zip file.

##### Parameters

- `zipData`: The zip file data as an ArrayBuffer

##### Returns

A Promise that resolves to a Mesh object.

#### `MeshUtils.convertToBufferGeometry(mesh: Mesh, options?: DecodeMeshOptions): THREE.BufferGeometry`

Converts a decoded mesh to a THREE.js BufferGeometry. Automatically triangulates non-triangular polygons for THREE.js compatibility.

##### Parameters

- `mesh`: The decoded mesh data
- `options`: Options for the conversion
  - `normalize`: Whether to normalize the mesh to fit within a unit cube (default: `false`)
  - `computeNormals`: Whether to compute normals if they don't exist (default: `true`)

##### Returns

A THREE.js BufferGeometry with triangulated indices.

#### `MeshUtils.getPolygonCount(mesh: Mesh): number`

Gets the number of polygons in a mesh.

##### Parameters

- `mesh`: The mesh to analyze

##### Returns

Number of polygons, or 0 if no indexSizes information is available.

#### `MeshUtils.isUniformPolygons(mesh: Mesh): boolean`

Checks if all polygons in a mesh have the same number of vertices.

##### Parameters

- `mesh`: The mesh to analyze

##### Returns

True if all polygons are uniform, false otherwise.

#### `MeshUtils.inferCellTypes(indexSizes: Uint32Array): Uint32Array`

Infers VTK cell types from polygon sizes.

##### Parameters

- `indexSizes`: Array of polygon sizes

##### Returns

Array of VTK cell type identifiers.

#### `MeshUtils.getCellTypes(mesh: Mesh): Uint32Array | undefined`

Gets or infers cell types for a mesh.

##### Parameters

- `mesh`: The mesh to get cell types for

##### Returns

Array of VTK cell type identifiers, or undefined if no polygon information is available.

#### `MeshUtils.inferSizesFromCellTypes(cellTypes: Uint8Array | Uint32Array): Uint32Array`

Converts VTK cell types to element sizes. This is used for automatic marker offset calculation.

##### Parameters

- `cellTypes`: Array of VTK cell type identifiers

##### Returns

Array of element sizes corresponding to each cell type.

#### `MeshUtils.sizesToOffsets(sizes: Uint32Array): Uint32Array`

Converts element sizes to offset arrays. This is used for automatic marker offset calculation.

##### Parameters

- `sizes`: Array of element sizes

##### Returns

Array of offset indices where each element starts in the flattened marker data.

#### `MeshUtils.decodeVertexBuffer(vertexCount: number, vertexSize: number, data: Uint8Array): Float32Array`

Decodes a vertex buffer using the meshoptimizer algorithm.

##### Parameters

- `vertexCount`: Number of vertices
- `vertexSize`: Size of each vertex in bytes
- `data`: Encoded vertex buffer

##### Returns

Decoded vertex buffer as a Float32Array.

#### `MeshUtils.encode(mesh: Mesh): EncodedMesh`

Encodes a mesh for efficient transmission. This function encodes the mesh's vertices, indices, and any additional arrays.

##### Parameters

- `mesh`: The mesh to encode

##### Returns

An EncodedMesh object containing the encoded vertices, indices, and arrays.

#### `MeshUtils.decode(MeshClass: any, encodedMesh: EncodedMesh): Mesh`

Decodes an encoded mesh. This function decodes the mesh's vertices, indices, and any additional arrays.

##### Parameters

- `MeshClass`: The mesh class constructor (not used in TypeScript version, included for API compatibility with Python)
- `encodedMesh`: The EncodedMesh object to decode

##### Returns

A decoded Mesh object.

#### `MeshUtils.getPolygonIndices(mesh: Mesh): Uint32Array[] | Uint32Array`

Gets indices in their original polygon structure.

##### Parameters

- `mesh`: The mesh to get polygon indices from

##### Returns

For uniform polygons: Uint32Array with flat indices. For mixed polygons: Array of Uint32Arrays where each represents a polygon.

#### `MeshUtils.normalizeVertices(vertices: Float32Array): Float32Array`

Normalizes vertices to fit within a unit cube centered at the origin.

##### Parameters

- `vertices`: The vertices to normalize

##### Returns

Normalized vertices as Float32Array.

#### `MeshUtils.triangulateIndices(mesh: Mesh): Uint32Array` (private)

Converts mesh indices to triangulated indices for THREE.js compatibility. This is used internally by `convertToBufferGeometry`.

##### Parameters

- `mesh`: The mesh to triangulate

##### Returns

Triangulated indices suitable for THREE.js rendering.

#### `MeshUtils.extractByMarker(mesh: Mesh, markerName: string): Mesh`

Extracts a submesh containing only the elements referenced by a specific marker.

##### Parameters

- `mesh`: The source mesh
- `markerName`: Name of the marker to extract

##### Returns

A new Mesh object containing only the vertices/elements from the marker.

##### Throws

Error if marker doesn't exist or is missing offset/type information.

#### `MeshUtils.extractMarkerAsBufferGeometry(mesh: Mesh, markerName: string, options?: DecodeMeshOptions): THREE.BufferGeometry`

Extracts a submesh by marker and converts it to a THREE.js BufferGeometry. This is a convenience method combining `extractByMarker` and `convertToBufferGeometry`.

##### Parameters

- `mesh`: The source mesh
- `markerName`: Name of the marker to extract
- `options`: Options for the conversion to BufferGeometry

##### Returns

THREE.js BufferGeometry containing only the marker elements.

##### Throws

Error if marker doesn't exist or is missing offset/type information.

### ArrayUtils

#### `ArrayUtils.encodeArray(data: Float32Array | Uint32Array): EncodedArray`

Encodes a Float32Array or Uint32Array using the meshoptimizer algorithm.

##### Parameters

- `data`: Float32Array or Uint32Array to encode

##### Returns

EncodedArray object containing the encoded data and metadata.

#### `ArrayUtils.decodeArray(data: Uint8Array, metadata: ArrayMetadata): Float32Array | Uint32Array`

Decodes an encoded array using the meshoptimizer algorithm.

##### Parameters

- `data`: Encoded array data
- `metadata`: Array metadata

##### Returns

Decoded array as a Float32Array or Uint32Array based on the metadata dtype.

#### `ArrayUtils.loadFromZip<T>(zipInput: JSZip | ArrayBuffer | Uint8Array, loadCustomMetadata?: boolean): Promise<ArrayResult<T>>`

Loads an array from a zip buffer containing array.bin and metadata.json.

##### Parameters

- `zipInput`: The zip buffer, JSZip instance, or raw data
- `loadCustomMetadata`: Whether to load custom metadata if present (default: false)

##### Returns

Promise resolving to ArrayResult containing the decoded array and optional custom metadata.

## Interfaces and Types

### Mesh Interface

```typescript
interface Mesh {
  vertices: Float32Array;
  indices?: Uint32Array;
  indexSizes?: Uint32Array;
  cellTypes?: Uint32Array;
  dim?: number;
  markerIndices?: Record<string, Uint32Array>;
  markerOffsets?: Record<string, Uint32Array>;
  markerTypes?: Record<string, Uint8Array>;
  markers?: Record<string, number[][]>;
  [key: string]: any; // Additional arrays and properties
}
```

### EncodedMesh Interface

```typescript
interface EncodedMesh {
  vertices: Uint8Array;
  indices?: Uint8Array;
  vertex_count: number;
  vertex_size: number;
  index_count?: number | null;
  index_size: number;
  indexSizes?: Uint8Array;
  arrays?: Record<string, EncodedArray>;
}
```

### EncodedArray Interface

```typescript
interface EncodedArray {
  data: Uint8Array;
  shape: number[];
  dtype: string;
  itemsize: number;
}
```

### ArrayMetadata Interface

```typescript
interface ArrayMetadata {
  shape: number[];
  dtype: string;
  itemsize: number;
}
```

### DecodeMeshOptions Interface

```typescript
interface DecodeMeshOptions {
  normalize?: boolean;    // Default: false
  computeNormals?: boolean; // Default: true
}
```

## Python Mesh Format

This library is designed to work with the Python meshoptimizer library's zip format. The zip file should contain:

- `mesh/vertices.bin`: Encoded vertex buffer
- `mesh/indices.bin`: Encoded index buffer
- `arrays/`: Additional arrays (normals, colors, etc.)
- `metadata.json`: General metadata about the mesh, including mesh size information

The metadata.json file contains:
- `class_name`: Name of the mesh class
- `module_name`: Name of the module containing the mesh class
- `field_data`: Dictionary of model fields that aren't numpy arrays
- `mesh_size`: Size metadata for the encoded mesh (vertex_count, vertex_size, index_count, index_size)



### Setup

```bash
# Clone the repository
git clone https://github.com/OpenOrion/meshly.git
cd meshly

# Install dependencies
pnpm install
```

### Build

The project uses TypeScript and builds to the `dist` directory:

```bash
pnpm run build
```

### Test

```bash
pnpm test
```

### CI/CD

This project uses GitHub Actions for continuous integration and deployment:

- **CI**: On every push and pull request, the code is built and tested.
- **Testing**: A dedicated test workflow runs tests on multiple Node.js versions (16.x, 18.x, 20.x) to ensure compatibility across different environments.
- **CD**: When a new release is created on GitHub, the package is automatically published to npm.

The GitHub Actions workflows are located in the root `.github/workflows` directory:
- `npm-publish.yml`: Handles building, testing, and publishing the package
- `npm-test.yml`: Runs tests across multiple Node.js versions

To publish a new version:

1. Update the version in `package.json`
2. Create a new release on GitHub
3. GitHub Actions will automatically build and publish the package to npm



## License

MIT