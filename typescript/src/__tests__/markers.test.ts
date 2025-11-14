import { describe, it, expect, beforeEach } from 'vitest'
import { Mesh, MeshUtils } from '../mesh'

describe('Mesh Markers', () => {
  let vertices: Float32Array
  let indices: Uint32Array
  let baseMesh: Mesh

  beforeEach(() => {
    // Create a simple 2D mesh (quad)
    vertices = new Float32Array([
      0.0, 0.0, 0.0,    // vertex 0
      1.0, 0.0, 0.0,    // vertex 1
      1.0, 1.0, 0.0,    // vertex 2
      0.0, 1.0, 0.0,    // vertex 3
    ])
    
    indices = new Uint32Array([0, 1, 2, 0, 2, 3])  // Two triangles
    
    baseMesh = {
      vertices,
      indices,
      dim: 2
    }
  })


  describe('mesh encoding/decoding with markers', () => {
    it('should preserve markers during encode/decode cycle', async () => {
      const meshWithMarkers: Mesh = {
        ...baseMesh,
        markerIndices: {
          "boundary": new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0])
        },
        markerOffsets: {
          "boundary": new Uint32Array([0, 2, 4, 6])
        },
        markerTypes: {
          "boundary": new Uint8Array([3, 3, 3, 3])
        }
      }

      // Encode and decode
      const encoded = MeshUtils.encode(meshWithMarkers)
      const decoded = MeshUtils.decode(encoded)

      // Check that marker data is preserved
      expect(decoded.markerIndices).toBeDefined()
      expect(decoded.markerOffsets).toBeDefined()
      expect(decoded.markerTypes).toBeDefined()

      expect(Array.from(decoded.markerIndices!.boundary)).toEqual([0, 1, 1, 2, 2, 3, 3, 0])
      expect(Array.from(decoded.markerOffsets!.boundary)).toEqual([0, 2, 4, 6])
      expect(Array.from(decoded.markerTypes!.boundary)).toEqual([3, 3, 3, 3])
    })

    it('should preserve markers during zip save/load cycle', async () => {
      const meshWithMarkers: Mesh = {
        ...baseMesh,
        markerIndices: {
          "edges": new Uint32Array([0, 1, 2, 3])
        },
        markerOffsets: {
          "edges": new Uint32Array([0, 2])
        },
        markerTypes: {
          "edges": new Uint8Array([3, 3])
        }
      }

      // Save and load
      const zipData = await MeshUtils.saveMeshToZip(meshWithMarkers)
      const loadedMesh = await MeshUtils.loadMeshFromZip(zipData.buffer as ArrayBuffer)

      // Check that marker data is preserved
      expect(loadedMesh.markerIndices).toBeDefined()
      expect(loadedMesh.markerOffsets).toBeDefined()
      expect(loadedMesh.markerTypes).toBeDefined()

      expect(Array.from(loadedMesh.markerIndices!.edges)).toEqual([0, 1, 2, 3])
      expect(Array.from(loadedMesh.markerOffsets!.edges)).toEqual([0, 2])
      expect(Array.from(loadedMesh.markerTypes!.edges)).toEqual([3, 3])
    })
  })

})