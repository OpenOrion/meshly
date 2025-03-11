import * as THREE from 'three';
import { convertToBufferGeometry } from '../converter';
import { DecodedMesh } from '../types';

describe('Converter functions', () => {
  describe('convertToBufferGeometry', () => {
    it('should convert a basic mesh to a BufferGeometry', () => {
      // Create a simple cube mesh
      const vertices = new Float32Array([
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
      
      const mesh: DecodedMesh = {
        vertices,
        indices
      };
      
      // Convert to BufferGeometry
      const geometry = convertToBufferGeometry(mesh);
      
      // Check that the result is a BufferGeometry
      expect(geometry).toBeInstanceOf(THREE.BufferGeometry);
      
      // Check that the geometry has the correct attributes
      expect(geometry.getAttribute('position')).toBeDefined();
      expect(geometry.getAttribute('position').count).toBe(8);
      expect(geometry.getIndex()).toBeDefined();
      expect(geometry.getIndex()?.count).toBe(36);
      
      // Check that normals were computed
      expect(geometry.getAttribute('normal')).toBeDefined();
    });
    
    it('should normalize vertices when requested', () => {
      // Create a mesh with vertices outside the unit cube
      const vertices = new Float32Array([
        -10, -10, -10,
        10, -10, -10,
        10, 10, -10,
        -10, 10, -10
      ]);
      
      const mesh: DecodedMesh = {
        vertices
      };
      
      // Convert to BufferGeometry with normalization
      const geometry = convertToBufferGeometry(mesh, { normalize: true });
      
      // Get the position attribute
      const position = geometry.getAttribute('position');
      
      // Check that all vertices are within the unit cube
      for (let i = 0; i < position.count; i++) {
        const x = position.getX(i);
        const y = position.getY(i);
        const z = position.getZ(i);
        
        expect(Math.abs(x)).toBeLessThanOrEqual(0.5);
        expect(Math.abs(y)).toBeLessThanOrEqual(0.5);
        expect(Math.abs(z)).toBeLessThanOrEqual(0.5);
      }
    });
    
    it('should not compute normals when requested', () => {
      // Create a simple mesh
      const vertices = new Float32Array([
        -0.5, -0.5, 0,
        0.5, -0.5, 0,
        0, 0.5, 0
      ]);
      
      const mesh: DecodedMesh = {
        vertices
      };
      
      // Convert to BufferGeometry without computing normals
      const geometry = convertToBufferGeometry(mesh, { computeNormals: false });
      
      // Check that normals were not computed
      expect(geometry.getAttribute('normal')).toBeUndefined();
    });
    
    it('should use provided normals', () => {
      // Create a simple mesh with normals
      const vertices = new Float32Array([
        -0.5, -0.5, 0,
        0.5, -0.5, 0,
        0, 0.5, 0
      ]);
      
      const normals = new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1
      ]);
      
      const mesh: DecodedMesh = {
        vertices,
        normals
      };
      
      // Convert to BufferGeometry
      const geometry = convertToBufferGeometry(mesh);
      
      // Check that the provided normals were used
      expect(geometry.getAttribute('normal')).toBeDefined();
      expect(geometry.getAttribute('normal').array).toBe(normals);
    });
    
    it('should handle colors and UVs', () => {
      // Create a simple mesh with colors and UVs
      const vertices = new Float32Array([
        -0.5, -0.5, 0,
        0.5, -0.5, 0,
        0, 0.5, 0
      ]);
      
      const colors = new Float32Array([
        1, 0, 0, 1,
        0, 1, 0, 1,
        0, 0, 1, 1
      ]);
      
      const uvs = new Float32Array([
        0, 0,
        1, 0,
        0.5, 1
      ]);
      
      const mesh: DecodedMesh = {
        vertices,
        colors,
        uvs
      };
      
      // Convert to BufferGeometry
      const geometry = convertToBufferGeometry(mesh);
      
      // Check that colors and UVs were added
      expect(geometry.getAttribute('color')).toBeDefined();
      expect(geometry.getAttribute('color').itemSize).toBe(4);
      expect(geometry.getAttribute('uv')).toBeDefined();
      expect(geometry.getAttribute('uv').itemSize).toBe(2);
    });
  });
});