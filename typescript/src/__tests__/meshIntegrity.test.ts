import * as THREE from 'three'
import { beforeEach, describe, expect, it } from 'vitest'
import { Mesh } from '../mesh'

describe('MeshIntegrity', () => {
  let vertices: Float32Array
  let indices: Uint32Array
  let mesh: Mesh

  beforeEach(() => {
    // Create a simple mesh (a cube)
    vertices = new Float32Array([
      // positions
      -0.5, -0.5, -0.5,
      0.5, -0.5, -0.5,
      0.5, 0.5, -0.5,
      -0.5, 0.5, -0.5,
      -0.5, -0.5, 0.5,
      0.5, -0.5, 0.5,
      0.5, 0.5, 0.5,
      -0.5, 0.5, 0.5
    ])

    indices = new Uint32Array([
      0, 1, 2, 2, 3, 0,  // front
      1, 5, 6, 6, 2, 1,  // right
      5, 4, 7, 7, 6, 5,  // back
      4, 0, 3, 3, 7, 4,  // left
      3, 2, 6, 6, 7, 3,  // top
      4, 5, 1, 1, 0, 4   // bottom
    ])

    mesh = new Mesh({
      vertices,
      indices
    })
  })

  it('should convert mesh to THREE.js BufferGeometry correctly', () => {
    // Convert the mesh to a THREE.js BufferGeometry
    const geometry = mesh.toBufferGeometry()

    // Check that the geometry has the correct attributes
    expect(geometry).toBeInstanceOf(THREE.BufferGeometry)
    expect(geometry.attributes.position).toBeDefined()
    expect(geometry.attributes.position.count).toBe(vertices.length / 3)
    expect(geometry.index).toBeDefined()
    expect(geometry.index!.count).toBe(indices.length)

    // Check that the vertices match
    const positionAttribute = geometry.attributes.position
    for (let i = 0; i < vertices.length; i++) {
      expect(positionAttribute.array[i]).toBeCloseTo(vertices[i], 5)
    }

    // Check that the indices match
    const indexAttribute = geometry.index!
    for (let i = 0; i < indices.length; i++) {
      expect(indexAttribute.array[i]).toBe(indices[i])
    }
  })
})