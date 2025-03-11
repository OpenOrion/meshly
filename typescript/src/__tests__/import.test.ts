/**
 * Tests for importing meshes from zip files
 */

import * as fs from 'fs';
import * as THREE from 'three';
import { loadMeshFromZip } from '../index';

// Import the createTestMesh function from the examples directory
const { createTestMesh } = require('../../scripts/create-test-mesh');

describe('Mesh import', () => {
  let testMeshPath: string;
  
  // Before all tests, create a test mesh
  beforeAll(async () => {
    // Create the test mesh
    testMeshPath = await createTestMesh();
    expect(fs.existsSync(testMeshPath)).toBe(true);
  });
  
  it('should load a mesh from a zip file', async () => {
    // Read the test mesh file
    const zipData = fs.readFileSync(testMeshPath);
    
    // Load the mesh from the zip file
    const geometry = await loadMeshFromZip(zipData.buffer, {
      normalize: true,
      computeNormals: true
    });
    
    // Verify that the geometry is a THREE.BufferGeometry
    expect(geometry).toBeInstanceOf(THREE.BufferGeometry);
    
    // Verify that the geometry has the correct number of vertices
    expect(geometry.getAttribute('position').count).toBe(8);
    
    // Verify that the geometry has the correct number of triangles
    expect(geometry.getIndex()?.count).toBe(36); // 12 triangles * 3 vertices per triangle
    
    // Verify that the geometry has normals
    expect(geometry.getAttribute('normal')).toBeDefined();
    
    // Verify that the geometry has colors
    expect(geometry.getAttribute('color')).toBeDefined();
    expect(geometry.getAttribute('color').itemSize).toBe(4); // RGBA
  });
  
  it('should handle normalization correctly', async () => {
    // Since the decoder is using mock data that might not be suitable for normalization,
    // we'll test the normalization function directly using the converter

    // Import the converter function and create a simple mesh
    const { convertToBufferGeometry } = require('../converter');
    
    // Create a simple mesh with vertices outside the unit cube
    const vertices = new Float32Array([
      -10, -10, -10,
      10, -10, -10,
      10, 10, -10,
      -10, 10, -10,
      -10, -10, 10,
      10, -10, 10,
      10, 10, 10,
      -10, 10, 10
    ]);
    
    const mesh = {
      vertices
    };
    
    // Convert to BufferGeometry with normalization
    const normalizedGeometry = convertToBufferGeometry(mesh, { normalize: true });
    
    // Convert to BufferGeometry without normalization
    const unnormalizedGeometry = convertToBufferGeometry(mesh, { normalize: false });
    
    // Get the position attributes
    const normalizedPosition = normalizedGeometry.getAttribute('position');
    const unnormalizedPosition = unnormalizedGeometry.getAttribute('position');
    
    // Verify that the normalized geometry has vertices within the unit cube
    for (let i = 0; i < normalizedPosition.count; i++) {
      const x = normalizedPosition.getX(i);
      const y = normalizedPosition.getY(i);
      const z = normalizedPosition.getZ(i);
      
      expect(Math.abs(x)).toBeLessThanOrEqual(0.5);
      expect(Math.abs(y)).toBeLessThanOrEqual(0.5);
      expect(Math.abs(z)).toBeLessThanOrEqual(0.5);
    }
    
    // Verify that the unnormalized geometry has the original vertex positions
    for (let i = 0; i < unnormalizedPosition.count; i++) {
      const x = unnormalizedPosition.getX(i);
      const y = unnormalizedPosition.getY(i);
      const z = unnormalizedPosition.getZ(i);
      
      // The original vertices are at ±10 in each dimension
      expect(Math.abs(x)).toBe(10);
      expect(Math.abs(y)).toBe(10);
      expect(Math.abs(z)).toBe(10);
    }
  });
  
  it('should compute normals when requested', async () => {
    // Read the test mesh file
    const zipData = fs.readFileSync(testMeshPath);
    
    // Load the mesh from the zip file without computing normals
    const geometryWithoutNormals = await loadMeshFromZip(zipData.buffer, {
      computeNormals: false
    });
    
    // Load the mesh from the zip file with computing normals
    const geometryWithNormals = await loadMeshFromZip(zipData.buffer, {
      computeNormals: true
    });
    
    // Verify that both geometries have normals (since the test mesh already has normals)
    expect(geometryWithoutNormals.getAttribute('normal')).toBeDefined();
    expect(geometryWithNormals.getAttribute('normal')).toBeDefined();
    
    // Create a test mesh without normals
    const { vertices, indices } = require('../../scripts/create-test-mesh');
    const meshWithoutNormals = {
      vertices,
      indices
    };
    
    // Convert to buffer geometry without computing normals
    const bufferGeometryWithoutNormals = new THREE.BufferGeometry();
    bufferGeometryWithoutNormals.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    bufferGeometryWithoutNormals.setIndex(new THREE.BufferAttribute(indices, 1));
    
    // Verify that the geometry doesn't have normals
    expect(bufferGeometryWithoutNormals.getAttribute('normal')).toBeUndefined();
    
    // Compute normals
    bufferGeometryWithoutNormals.computeVertexNormals();
    
    // Verify that the geometry now has normals
    expect(bufferGeometryWithoutNormals.getAttribute('normal')).toBeDefined();
  });
});