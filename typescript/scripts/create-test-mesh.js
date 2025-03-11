/**
 * This script creates a test mesh file in the Python meshoptimizer format
 * for testing the TypeScript library.
 */
const fs = require('fs');
const path = require('path');
const JSZip = require('jszip');

// Create a simple cube mesh
const vertices = new Float32Array([
  // positions
  -0.5, -0.5, -0.5,
  0.5, -0.5, -0.5,
  0.5, 0.5, -0.5,
  -0.5, 0.5, -0.5,
  -0.5, -0.5, 0.5,
  0.5, -0.5, 0.5,
  0.5, 0.5, 0.5,
  -0.5, 0.5, 0.5
]);

const indices = new Uint32Array([
  0, 1, 2, 2, 3, 0,  // front
  1, 5, 6, 6, 2, 1,  // right
  5, 4, 7, 7, 6, 5,  // back
  4, 0, 3, 3, 7, 4,  // left
  3, 2, 6, 6, 7, 3,  // top
  4, 5, 1, 1, 0, 4   // bottom
]);

// Create normals (one per vertex)
const normals = new Float32Array([
  0.0, 0.0, -1.0,  // front vertices
  0.0, 0.0, -1.0,
  0.0, 0.0, -1.0,
  0.0, 0.0, -1.0,
  0.0, 0.0, 1.0,   // back vertices
  0.0, 0.0, 1.0,
  0.0, 0.0, 1.0,
  0.0, 0.0, 1.0
]);

// Create colors (RGBA, one per vertex)
const colors = new Float32Array([
  1.0, 0.0, 0.0, 1.0,  // red
  0.0, 1.0, 0.0, 1.0,  // green
  0.0, 0.0, 1.0, 1.0,  // blue
  1.0, 1.0, 0.0, 1.0,  // yellow
  1.0, 0.0, 1.0, 1.0,  // magenta
  0.0, 1.0, 1.0, 1.0,  // cyan
  0.5, 0.5, 0.5, 1.0,  // gray
  1.0, 1.0, 1.0, 1.0   // white
]);

// Create a mock encoded vertex buffer
// In a real scenario, this would be the output of the Python library's encode_vertex_buffer function
function mockEncodeVertexBuffer(vertices) {
  // Create a simple mock encoding (just prepend 0xA0 to the buffer)
  const buffer = new Uint8Array(vertices.buffer.byteLength + 1);
  buffer[0] = 0xA0; // Header for vertex buffer
  buffer.set(new Uint8Array(vertices.buffer), 1);
  return buffer;
}

// Create a mock encoded index buffer
// In a real scenario, this would be the output of the Python library's encode_index_buffer function
function mockEncodeIndexBuffer(indices) {
  // Create a simple mock encoding (just prepend 0xE0 to the buffer)
  const buffer = new Uint8Array(indices.buffer.byteLength + 1);
  buffer[0] = 0xE0; // Header for index buffer
  buffer.set(new Uint8Array(indices.buffer), 1);
  return buffer;
}

// Create a mock encoded array
function mockEncodeArray(array) {
  // Create a simple mock encoding (just prepend 0xA0 to the buffer)
  const buffer = new Uint8Array(array.buffer.byteLength + 1);
  buffer[0] = 0xA0; // Header for vertex buffer
  buffer.set(new Uint8Array(array.buffer), 1);
  return buffer;
}

async function createTestMesh() {
  // Create a new zip file
  const zip = new JSZip();

  // Encode the vertex buffer
  const encodedVertices = mockEncodeVertexBuffer(vertices);

  // Encode the index buffer
  const encodedIndices = mockEncodeIndexBuffer(indices);

  // Encode the normals and colors
  const encodedNormals = mockEncodeArray(normals);
  const encodedColors = mockEncodeArray(colors);

  // Add the encoded buffers to the zip file
  zip.file('mesh/vertices.bin', encodedVertices);
  zip.file('mesh/indices.bin', encodedIndices);

  // Add the mesh metadata
  const meshMetadata = {
    vertex_count: vertices.length / 3,
    vertex_size: 12, // 3 floats * 4 bytes
    index_count: indices.length,
    index_size: 4 // 4 bytes per index (uint32)
  };

  zip.file('mesh/metadata.json', JSON.stringify(meshMetadata, null, 2));

  // Add the general metadata
  const fileMetadata = {
    class_name: 'ColoredMesh',
    module_name: 'meshly.mesh'
  };

  zip.file('metadata.json', JSON.stringify(fileMetadata, null, 2));

  // Add the normals and colors as additional arrays
  zip.file('arrays/normals.bin', encodedNormals);
  zip.file('arrays/colors.bin', encodedColors);

  // Add the array metadata
  const normalsMetadata = {
    shape: [8, 3], // 8 vertices, 3 components per vertex
    dtype: 'float32',
    itemsize: 4 // 4 bytes per float32
  };

  const colorsMetadata = {
    shape: [8, 4], // 8 vertices, 4 components per vertex (RGBA)
    dtype: 'float32',
    itemsize: 4 // 4 bytes per float32
  };

  zip.file('arrays/normals_metadata.json', JSON.stringify(normalsMetadata, null, 2));
  zip.file('arrays/colors_metadata.json', JSON.stringify(colorsMetadata, null, 2));

  // Generate the zip file
  const content = await zip.generateAsync({ type: 'nodebuffer' });

  // Save the zip file
  const outputPath = path.join(__dirname, 'test-mesh.zip');
  fs.writeFileSync(outputPath, content);

  console.log(`Test mesh saved to ${outputPath}`);
  return outputPath;
}

// If this script is run directly, create the test mesh
if (require.main === module) {
  createTestMesh().catch(console.error);
}

// Export the createTestMesh function and mesh data for use in other scripts
module.exports = {
  createTestMesh,
  vertices,
  indices,
  normals,
  colors
};
