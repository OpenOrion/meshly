import { beforeEach, describe, expect, it } from 'vitest'
import { Mesh } from '../mesh'

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

  it('should convert mesh to BufferGeometry', () => {
    // Simple triangle mesh
    const mesh = new Mesh({
      vertices,
      indices: new Uint32Array([0, 1, 2]),
      indexSizes: new Uint32Array([3]),
      cellTypes: new Uint32Array([5]) // VTK_TRIANGLE
    })

    const geometry = mesh.toBufferGeometry()

    expect(geometry.index!.count).toBe(3)
    expect(Array.from(geometry.index!.array)).toEqual([0, 1, 2])
  })

  it('should handle uniform polygons check', () => {
    const mesh = new Mesh({
      vertices,
      indices: new Uint32Array([0, 1, 2, 2, 3, 0]),
      indexSizes: new Uint32Array([3, 3]),
      cellTypes: new Uint32Array([5, 5])
    })

    expect(mesh.isUniformPolygons()).toBe(true)
    expect(mesh.getPolygonCount()).toBe(2)
  })

  it('should handle legacy format without indexSizes', () => {
    // Legacy triangular mesh without indexSizes
    const mesh = new Mesh({
      vertices,
      indices: new Uint32Array([0, 1, 2, 2, 3, 0])
    })

    expect(mesh.getPolygonCount()).toBe(0)
    expect(mesh.isUniformPolygons()).toBe(true)

    const geometry = mesh.toBufferGeometry()

    // Indices should remain unchanged for legacy format
    expect(geometry.index!.count).toBe(6)
    expect(Array.from(geometry.index!.array)).toEqual([0, 1, 2, 2, 3, 0])
  })
})