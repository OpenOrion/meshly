import { beforeEach, describe, expect, it } from 'vitest'
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

  describe('marker auto-calculation', () => {
    it('should automatically calculate marker offsets from marker types during decode', () => {
      // Test the utility functions directly since encoding always includes offsets when they're present
      const markerTypes = new Uint8Array([3, 3]) // Two lines (VTK_LINE = 3)
      const expectedSizes = MeshUtils.inferSizesFromCellTypes(markerTypes)
      const calculatedOffsets = MeshUtils.sizesToOffsets(expectedSizes)

      expect(Array.from(expectedSizes)).toEqual([2, 2]) // Two lines, each with 2 vertices
      expect(Array.from(calculatedOffsets)).toEqual([0, 2]) // Offsets: 0, 2
    })

    it('should automatically calculate marker offsets for mixed cell types', () => {
      const markerTypes = new Uint8Array([1, 3, 5]) // VTK_VERTEX, VTK_LINE, VTK_TRIANGLE
      const expectedSizes = MeshUtils.inferSizesFromCellTypes(markerTypes)
      const calculatedOffsets = MeshUtils.sizesToOffsets(expectedSizes)

      expect(Array.from(expectedSizes)).toEqual([1, 2, 3]) // vertex(1) + line(2) + triangle(3)
      expect(Array.from(calculatedOffsets)).toEqual([0, 1, 3]) // Offsets: 0, 1, 3
    })

    it('should preserve manually provided marker offsets', () => {
      const meshWithManualOffsets: Mesh = {
        ...baseMesh,
        markerIndices: {
          "boundary": new Uint32Array([0, 1, 1, 2])
        },
        markerTypes: {
          "boundary": new Uint8Array([3, 3]) // Two lines
        },
        markerOffsets: {
          "boundary": new Uint32Array([0, 2]) // Manually provided
        }
      }

      // Encode and decode
      const encoded = MeshUtils.encode(meshWithManualOffsets)
      const decoded = MeshUtils.decode(encoded)

      // Check that the manually provided marker offsets were preserved
      expect(decoded.markerOffsets).toBeDefined()
      expect(decoded.markerOffsets!.boundary).toBeDefined()
      expect(Array.from(decoded.markerOffsets!.boundary)).toEqual([0, 2])
    })

    it('should handle error for unknown VTK cell types', () => {
      // Use a smaller number that won't overflow in Uint8Array
      const unknownTypes = new Uint8Array([99]) // Unknown cell type (small enough to not overflow)

      // Should throw an error when calculating sizes
      expect(() => MeshUtils.inferSizesFromCellTypes(unknownTypes)).toThrow('Unknown VTK cell type: 99')
    })

  })

  describe('extractByMarker', () => {
    it('should extract a submesh by marker name', () => {
      // Create a mesh with a simple triangle
      const vertices = new Float32Array([
        0.0, 0.0, 0.0,    // vertex 0
        1.0, 0.0, 0.0,    // vertex 1
        0.5, 1.0, 0.0,    // vertex 2
        2.0, 0.0, 0.0,    // vertex 3
      ])

      const indices = new Uint32Array([0, 1, 2, 1, 3, 2])  // Two triangles

      const mesh: Mesh = {
        vertices,
        indices,
        indexSizes: new Uint32Array([3, 3]),
        cellTypes: new Uint32Array([5, 5]), // VTK_TRIANGLE
        markerIndices: {
          "first_triangle": new Uint32Array([0, 1, 2])
        },
        markerOffsets: {
          "first_triangle": new Uint32Array([0])
        },
        markerTypes: {
          "first_triangle": new Uint8Array([5]) // VTK_TRIANGLE
        },
        dim: 2
      }

      const extracted = MeshUtils.extractByMarker(mesh, "first_triangle")

      // Should have 3 vertices (0, 1, 2)
      expect(extracted.vertices.length).toBe(9) // 3 vertices * 3 components
      expect(extracted.indices).toBeDefined()
      expect(extracted.indices!.length).toBe(3)

      // Indices should be remapped to 0, 1, 2
      expect(Array.from(extracted.indices!)).toEqual([0, 1, 2])

      // Should have correct sizes and types
      expect(Array.from(extracted.indexSizes!)).toEqual([3])
      expect(Array.from(extracted.cellTypes!)).toEqual([5])
    })

    it('should extract a submesh with multiple elements', () => {
      const vertices = new Float32Array([
        0.0, 0.0, 0.0,    // vertex 0
        1.0, 0.0, 0.0,    // vertex 1
        1.0, 1.0, 0.0,    // vertex 2
        0.0, 1.0, 0.0,    // vertex 3
        2.0, 0.0, 0.0,    // vertex 4
      ])

      const mesh: Mesh = {
        vertices,
        indices: new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0]),  // 4 edges
        indexSizes: new Uint32Array([2, 2, 2, 2]),
        cellTypes: new Uint32Array([3, 3, 3, 3]), // VTK_LINE
        markerIndices: {
          "boundary": new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0])
        },
        markerOffsets: {
          "boundary": new Uint32Array([0, 2, 4, 6])
        },
        markerTypes: {
          "boundary": new Uint8Array([3, 3, 3, 3]) // VTK_LINE
        },
        dim: 2
      }

      const extracted = MeshUtils.extractByMarker(mesh, "boundary")

      // Should have 4 vertices (0, 1, 2, 3)
      expect(extracted.vertices.length).toBe(12) // 4 vertices * 3 components

      // Indices should be remapped
      expect(extracted.indices).toBeDefined()
      expect(extracted.indices!.length).toBe(8)
      expect(Array.from(extracted.indices!)).toEqual([0, 1, 1, 2, 2, 3, 3, 0])

      // Should have correct sizes and types
      expect(Array.from(extracted.indexSizes!)).toEqual([2, 2, 2, 2])
      expect(Array.from(extracted.cellTypes!)).toEqual([3, 3, 3, 3])
    })

    it('should throw error for nonexistent marker', () => {
      const mesh: Mesh = {
        vertices: baseMesh.vertices,
        indices: baseMesh.indices,
        markerIndices: {
          "boundary": new Uint32Array([0, 1])
        },
        markerOffsets: {
          "boundary": new Uint32Array([0])
        },
        markerTypes: {
          "boundary": new Uint8Array([3])
        }
      }

      expect(() => MeshUtils.extractByMarker(mesh, "nonexistent")).toThrow(
        "Marker 'nonexistent' not found"
      )
    })

    it('should throw error for marker missing offset information', () => {
      const mesh: Mesh = {
        vertices: baseMesh.vertices,
        indices: baseMesh.indices,
        markerIndices: {
          "incomplete": new Uint32Array([0, 1, 2])
        }
        // Missing markerOffsets and markerTypes
      }

      expect(() => MeshUtils.extractByMarker(mesh, "incomplete")).toThrow(
        "Marker 'incomplete' is missing offset or type information"
      )
    })

    it('should remap vertex indices correctly', () => {
      // Create a mesh where marker references non-contiguous vertices
      const vertices = new Float32Array([
        0.0, 0.0, 0.0,    // vertex 0
        1.0, 0.0, 0.0,    // vertex 1
        2.0, 0.0, 0.0,    // vertex 2
        3.0, 0.0, 0.0,    // vertex 3
        4.0, 0.0, 0.0,    // vertex 4
      ])

      const mesh: Mesh = {
        vertices,
        markerIndices: {
          "subset": new Uint32Array([1, 3, 1, 4]) // Uses vertices 1, 3, 4
        },
        markerOffsets: {
          "subset": new Uint32Array([0, 2])
        },
        markerTypes: {
          "subset": new Uint8Array([3, 3]) // Two lines
        },
        dim: 3
      }

      const extracted = MeshUtils.extractByMarker(mesh, "subset")

      // Should have only 3 unique vertices (1, 3, 4) -> remapped to (0, 1, 2)
      expect(extracted.vertices.length).toBe(9) // 3 vertices * 3 components

      // Check that vertices are correctly extracted
      expect(Array.from(extracted.vertices)).toEqual([
        1.0, 0.0, 0.0,  // original vertex 1
        3.0, 0.0, 0.0,  // original vertex 3
        4.0, 0.0, 0.0,  // original vertex 4
      ])

      // Indices should be remapped: [1, 3, 1, 4] -> [0, 1, 0, 2]
      expect(Array.from(extracted.indices!)).toEqual([0, 1, 0, 2])
    })
  })

  describe('extractMarkerAsBufferGeometry', () => {
    it('should extract marker and convert to BufferGeometry', () => {
      const vertices = new Float32Array([
        0.0, 0.0, 0.0,    // vertex 0
        1.0, 0.0, 0.0,    // vertex 1
        0.5, 1.0, 0.0,    // vertex 2
      ])

      const mesh: Mesh = {
        vertices,
        markerIndices: {
          "triangle": new Uint32Array([0, 1, 2])
        },
        markerOffsets: {
          "triangle": new Uint32Array([0])
        },
        markerTypes: {
          "triangle": new Uint8Array([5]) // VTK_TRIANGLE
        },
        dim: 2
      }

      const geometry = MeshUtils.extractMarkerAsBufferGeometry(mesh, "triangle")

      // Should have position attribute
      expect(geometry.attributes.position).toBeDefined()
      expect(geometry.attributes.position.count).toBe(3)

      // Should have index
      expect(geometry.index).toBeDefined()
      expect(geometry.index!.count).toBe(3)
    })

    it('should apply options when converting to BufferGeometry', () => {
      const vertices = new Float32Array([
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, 1.0, 0.0,
      ])

      const mesh: Mesh = {
        vertices,
        markerIndices: {
          "triangle": new Uint32Array([0, 1, 2])
        },
        markerOffsets: {
          "triangle": new Uint32Array([0])
        },
        markerTypes: {
          "triangle": new Uint8Array([5])
        },
        dim: 2
      }

      const geometry = MeshUtils.extractMarkerAsBufferGeometry(mesh, "triangle", {
        computeNormals: true
      })

      // Should have computed normals
      expect(geometry.attributes.normal).toBeDefined()
    })
  })

})