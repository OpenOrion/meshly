import { beforeEach, describe, expect, it } from 'vitest'
import { Mesh } from '../mesh'

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

    baseMesh = new Mesh({
      vertices,
      indices,
      dim: 2
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

      const mesh = new Mesh({
        vertices,
        indices,
        indexSizes: new Uint32Array([3, 3]),
        cellTypes: new Uint32Array([5, 5]), // VTK_TRIANGLE
        markers: {
          "first_triangle": new Uint32Array([0, 1, 2])
        },
        markerSizes: {
          "first_triangle": new Uint32Array([3])
        },
        markerCellTypes: {
          "first_triangle": new Uint32Array([5]) // VTK_TRIANGLE
        },
        dim: 2
      })

      const extracted = mesh.extractByMarker("first_triangle")

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

      const mesh = new Mesh({
        vertices,
        indices: new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0]),  // 4 edges
        indexSizes: new Uint32Array([2, 2, 2, 2]),
        cellTypes: new Uint32Array([3, 3, 3, 3]), // VTK_LINE
        markers: {
          "boundary": new Uint32Array([0, 1, 1, 2, 2, 3, 3, 0])
        },
        markerSizes: {
          "boundary": new Uint32Array([2, 2, 2, 2])
        },
        markerCellTypes: {
          "boundary": new Uint32Array([3, 3, 3, 3]) // VTK_LINE
        },
        dim: 2
      })

      const extracted = mesh.extractByMarker("boundary")

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
      const mesh = new Mesh({
        vertices: baseMesh.vertices,
        indices: baseMesh.indices,
        markers: {
          "boundary": new Uint32Array([0, 1])
        },
        markerSizes: {
          "boundary": new Uint32Array([2])
        },
        markerCellTypes: {
          "boundary": new Uint32Array([3])
        }
      })

      expect(() => mesh.extractByMarker("nonexistent")).toThrow(
        "Marker 'nonexistent' not found"
      )
    })

    it('should throw error for marker missing offset information', () => {
      const mesh = new Mesh({
        vertices: baseMesh.vertices,
        indices: baseMesh.indices,
        markers: {
          "incomplete": new Uint32Array([0, 1, 2])
        }
        // Missing markerSizes and markerCellTypes
      })

      expect(() => mesh.extractByMarker("incomplete")).toThrow(
        "Marker 'incomplete' is missing sizes or cell type information"
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

      const mesh = new Mesh({
        vertices,
        markers: {
          "subset": new Uint32Array([1, 3, 1, 4]) // Uses vertices 1, 3, 4
        },
        markerSizes: {
          "subset": new Uint32Array([2, 2])
        },
        markerCellTypes: {
          "subset": new Uint32Array([3, 3]) // Two lines
        },
        dim: 3
      })

      const extracted = mesh.extractByMarker("subset")

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

      const mesh = new Mesh({
        vertices,
        markers: {
          "triangle": new Uint32Array([0, 1, 2])
        },
        markerSizes: {
          "triangle": new Uint32Array([3])
        },
        markerCellTypes: {
          "triangle": new Uint32Array([5]) // VTK_TRIANGLE
        },
        dim: 2
      })

      const geometry = mesh.extractMarkerAsBufferGeometry("triangle")

      // Should have position attribute
      expect(geometry.attributes.position).toBeDefined()
      expect(geometry.attributes.position.count).toBe(3)

      // Should have index
      expect(geometry.index).toBeDefined()
      expect(geometry.index!.count).toBe(3)
    })
  })

})