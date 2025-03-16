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
const decodedMesh = MeshUtils.decode(null, encodedMesh);
console.log(`Decoded vertices: ${decodedMesh.vertices.length} elements`);
console.log(`Decoded indices: ${decodedMesh.indices?.length} elements`);
console.log(`Decoded normals: ${decodedMesh.normals?.length} elements`);
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

Converts a decoded mesh to a THREE.js BufferGeometry.

##### Parameters

- `mesh`: The decoded mesh data
- `options`: Options for the conversion
  - `normalize`: Whether to normalize the mesh to fit within a unit cube (default: `false`)
  - `computeNormals`: Whether to compute normals if they don't exist (default: `true`)

##### Returns

A THREE.js BufferGeometry.

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


### ArrayUtils

#### `ArrayUtils.encodeArray(data: Float32Array): EncodedArray`

Encodes a Float32Array using the meshoptimizer algorithm.

##### Parameters

- `data`: Float32Array to encode

##### Returns

EncodedArray object containing the encoded data and metadata.

#### `ArrayUtils.decodeArray(data: Uint8Array, metadata: ArrayMetadata): Float32Array`

Decodes an encoded array using the meshoptimizer algorithm.

##### Parameters

- `data`: Encoded array data
- `metadata`: Array metadata

##### Returns

Decoded array as a Float32Array.

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