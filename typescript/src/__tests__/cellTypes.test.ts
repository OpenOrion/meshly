import { beforeEach, describe, expect, it } from 'vitest'
import { Mesh, MeshUtils } from '../mesh'

describe('CellTypes', () => {
  let vertices: Float32Array

  beforeEach(() => {
    // Create vertices for various polygon tests
    vertices = new Float32Array([
      0.0, 0.0, 0.0,  // 0
      1.0, 0.0, 0.0,  // 1
      1.0, 1.0, 0.0,  // 2
      0.0, 1.0, 0.0,  // 3
      0.5, 0.5, 1.0,  // 4
      2.0, 0.0, 0.0,  // 5
      2.0, 1.0, 0.0,  // 6
      1.5, 0.5, 1.0   // 7
    ])
  })

  it('should infer cell types from polygon sizes', () => {
    const indexSizes = new Uint32Array([2, 3, 4, 5, 6, 8, 9])
    const cellTypes = MeshUtils.inferCellTypes(indexSizes)

    expect(cellTypes).toEqual(new Uint32Array([
      3,  // Line (2 vertices)
      5,  // Triangle (3 vertices) 
      9,  // Quad (4 vertices)
      14, // Pyramid (5 vertices)
      13, // Wedge (6 vertices)
      12, // Hexahedron (8 vertices)
      7   // Generic polygon (9 vertices)
    ]))
  })

  it('should get cell types from mesh when available', () => {
    const mesh: Mesh = {
      vertices,
      indices: new Uint32Array([0, 1, 2, 1, 2, 3, 4]),
      indexSizes: new Uint32Array([3, 4]),
      cellTypes: new Uint32Array([5, 9])  // Triangle, Quad
    }

    const cellTypes = MeshUtils.getCellTypes(mesh)
    expect(cellTypes).toEqual(new Uint32Array([5, 9]))
  })

  it('should infer cell types when not provided', () => {
    const mesh: Mesh = {
      vertices,
      indices: new Uint32Array([0, 1, 2, 1, 2, 3, 4]),
      indexSizes: new Uint32Array([3, 4])
    }

    const cellTypes = MeshUtils.getCellTypes(mesh)
    expect(cellTypes).toEqual(new Uint32Array([5, 9])) // Triangle, Quad
  })

  it('should convert mesh to BufferGeometry', () => {
    // Simple triangle mesh
    const mesh: Mesh = {
      vertices,
      indices: new Uint32Array([0, 1, 2]),
      indexSizes: new Uint32Array([3]),
      cellTypes: new Uint32Array([5]) // VTK_TRIANGLE
    }

    const geometry = MeshUtils.convertToBufferGeometry(mesh)

    expect(geometry.index!.count).toBe(3)
    expect(Array.from(geometry.index!.array)).toEqual([0, 1, 2])
  })

  it('should encode and decode cell types correctly', () => {
    const mesh: Mesh = {
      vertices,
      indices: new Uint32Array([0, 1, 2, 1, 5, 6, 2]),
      indexSizes: new Uint32Array([3, 4]),
      cellTypes: new Uint32Array([5, 9]) // Triangle, Quad
    }

    // Encode the mesh
    const encodedMesh = MeshUtils.encode(mesh)

    // Verify cellTypes is in the encoded arrays
    expect(encodedMesh.arrays).toBeDefined()
    expect(encodedMesh.arrays!['cellTypes']).toBeDefined()

    // Decode the mesh
    const decodedMesh = MeshUtils.decode(encodedMesh)

    // Check that cellTypes is preserved
    expect(decodedMesh.cellTypes).toEqual(new Uint32Array([5, 9]))
    expect(decodedMesh.indexSizes).toEqual(new Uint32Array([3, 4]))
  })

  it('should handle uniform polygons check', () => {
    const mesh: Mesh = {
      vertices,
      indices: new Uint32Array([0, 1, 2, 2, 3, 0]),
      indexSizes: new Uint32Array([3, 3]),
      cellTypes: new Uint32Array([5, 5])
    }

    expect(MeshUtils.isUniformPolygons(mesh)).toBe(true)
    expect(MeshUtils.getPolygonCount(mesh)).toBe(2)
  })

  it('should handle legacy format without indexSizes', () => {
    // Legacy triangular mesh without indexSizes
    const mesh: Mesh = {
      vertices,
      indices: new Uint32Array([0, 1, 2, 2, 3, 0])
    }

    expect(MeshUtils.getPolygonCount(mesh)).toBe(0)
    expect(MeshUtils.isUniformPolygons(mesh)).toBe(true)
    expect(MeshUtils.getCellTypes(mesh)).toBeUndefined()

    const geometry = MeshUtils.convertToBufferGeometry(mesh)

    // Indices should remain unchanged for legacy format
    expect(geometry.index!.count).toBe(6)
    expect(Array.from(geometry.index!.array)).toEqual([0, 1, 2, 2, 3, 0])
  })

  it('should handle complex VTK cell types', () => {
    const indexSizes = new Uint32Array([1, 2, 3, 4, 5, 6, 8])
    const expectedCellTypes = new Uint32Array([
      1,  // VTK_VERTEX
      3,  // VTK_LINE
      5,  // VTK_TRIANGLE
      9,  // VTK_QUAD
      14, // VTK_PYRAMID
      13, // VTK_WEDGE
      12  // VTK_HEXAHEDRON
    ])

    const inferredCellTypes = MeshUtils.inferCellTypes(indexSizes)
    expect(inferredCellTypes).toEqual(expectedCellTypes)
  })
})