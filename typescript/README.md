# Meshly

A TypeScript library for decoding Python meshoptimizer zip files into THREE.js geometries.

## Installation

```bash
npm install meshly
```

## Development

## Usage

### Basic Usage

```typescript
import * as THREE from 'three';
import { loadZipAsBufferGeometry } from 'meshly';

// Load a mesh from a zip file
async function loadMesh(zipData: ArrayBuffer) {
  const geometry = await loadZipAsBufferGeometry(zipData, {
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
import { extractMeshFromZip, convertToBufferGeometry } from 'meshly';

async function loadMeshManually(zipData: ArrayBuffer) {
  // Extract the mesh data from the zip file
  const decodedMesh = await extractMeshFromZip(zipData);
  
  // Convert the decoded mesh to a THREE.js geometry
  const geometry = convertToBufferGeometry(decodedMesh, {
    normalize: true,
    computeNormals: true
  });
  
  return geometry;
}
```

## API Reference

### `loadZipAsBufferGeometry(zipData: ArrayBuffer, options?: DecodeMeshOptions): Promise<THREE.BufferGeometry>`

Loads a mesh from a zip file and converts it to a THREE.js BufferGeometry.

#### Parameters

- `zipData`: The zip file data as an ArrayBuffer
- `options`: Options for decoding the mesh
  - `normalize`: Whether to normalize the mesh to fit within a unit cube (default: `false`)
  - `computeNormals`: Whether to compute normals if they don't exist (default: `true`)

#### Returns

A Promise that resolves to a THREE.js BufferGeometry.

### `extractMeshFromZip(zipData: ArrayBuffer): Promise<DecodedMesh>`

Extracts and decodes a mesh from a zip file.

#### Parameters

- `zipData`: The zip file data as an ArrayBuffer

#### Returns

A Promise that resolves to a DecodedMesh object.

### `convertToBufferGeometry(mesh: DecodedMesh, options?: DecodeMeshOptions): THREE.BufferGeometry`

Converts a decoded mesh to a THREE.js BufferGeometry.

#### Parameters

- `mesh`: The decoded mesh data
- `options`: Options for the conversion
  - `normalize`: Whether to normalize the mesh to fit within a unit cube (default: `false`)
  - `computeNormals`: Whether to compute normals if they don't exist (default: `true`)

#### Returns

A THREE.js BufferGeometry.

## Python Mesh Format

This library is designed to work with the Python meshoptimizer library's zip format. The zip file should contain:

- `mesh/vertices.bin`: Encoded vertex buffer
- `mesh/indices.bin`: Encoded index buffer
- `mesh/metadata.json`: Metadata about the mesh
- `arrays/`: Additional arrays (normals, colors, etc.)
- `metadata.json`: General metadata about the mesh



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