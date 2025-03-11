import * as THREE from 'three';
import { DecodedMesh, DecodeMeshOptions } from './types';

/**
 * Converts a decoded mesh to a THREE.js BufferGeometry
 * 
 * @param mesh The decoded mesh data
 * @param options Options for the conversion
 * @returns THREE.js BufferGeometry
 */
export function convertToBufferGeometry(
  mesh: DecodedMesh, 
  options: DecodeMeshOptions = {}
): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  
  // Set default options
  const opts = {
    normalize: false,
    computeNormals: true,
    ...options
  };
  
  // Add vertices
  const vertices = mesh.vertices;
  
  // If normalize is true, normalize the vertices to fit within a unit cube
  let normalizedVertices = vertices;
  if (opts.normalize) {
    normalizedVertices = normalizeVertices(vertices);
  }
  
  // Add the vertices to the geometry
  geometry.setAttribute('position', new THREE.BufferAttribute(normalizedVertices, 3));
  
  // Add indices if they exist
  if (mesh.indices) {
    geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
  }
  
  // Add normals if they exist, otherwise compute them if requested
  if (mesh.normals) {
    geometry.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
  } else if (opts.computeNormals) {
    geometry.computeVertexNormals();
  }
  
  // Add colors if they exist
  if (mesh.colors) {
    // Check if colors have 3 or 4 components (RGB or RGBA)
    const itemSize = mesh.colors.length / (vertices.length / 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(mesh.colors, itemSize));
  }
  
  // Add UVs if they exist
  if (mesh.uvs) {
    // Check if UVs have 2 or 3 components
    const itemSize = mesh.uvs.length / (vertices.length / 3);
    geometry.setAttribute('uv', new THREE.BufferAttribute(mesh.uvs, itemSize));
  }
  
  return geometry;
}

/**
 * Normalizes vertices to fit within a unit cube centered at the origin
 * 
 * @param vertices The vertices to normalize
 * @returns Normalized vertices
 */
function normalizeVertices(vertices: Float32Array): Float32Array {
  // Find the bounding box
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  
  for (let i = 0; i < vertices.length; i += 3) {
    const x = vertices[i];
    const y = vertices[i + 1];
    const z = vertices[i + 2];
    
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  
  // Calculate the center and size
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerZ = (minZ + maxZ) / 2;
  
  const sizeX = maxX - minX;
  const sizeY = maxY - minY;
  const sizeZ = maxZ - minZ;
  
  // Find the maximum dimension
  const maxSize = Math.max(sizeX, sizeY, sizeZ);
  
  // Create a new array for the normalized vertices
  const normalizedVertices = new Float32Array(vertices.length);
  
  // Normalize the vertices
  for (let i = 0; i < vertices.length; i += 3) {
    normalizedVertices[i] = (vertices[i] - centerX) / maxSize;
    normalizedVertices[i + 1] = (vertices[i + 1] - centerY) / maxSize;
    normalizedVertices[i + 2] = (vertices[i + 2] - centerZ) / maxSize;
  }
  
  return normalizedVertices;
}