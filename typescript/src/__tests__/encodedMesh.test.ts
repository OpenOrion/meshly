import JSZip from 'jszip'
import { MeshoptEncoder } from 'meshoptimizer'
import * as THREE from 'three'
import { beforeEach, describe, expect, it } from 'vitest'
import { EncodedMesh, Mesh, MeshUtils } from '../mesh'

describe('EncodedMesh', () => {
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

    mesh = {
      vertices,
      indices
    }
  })

  /**
   * Get a set of triangles from vertices and indices.
   * Each triangle is represented as a string of vertex coordinates.
   * This makes the comparison invariant to vertex order within triangles.
   */
  function getTrianglesSet(vertices: Float32Array, indices: Uint32Array): Set<string> {
    const triangles = new Set<string>()

    for (let i = 0; i < indices.length; i += 3) {
      // Get the three vertices of the triangle
      const v1 = [
        vertices[indices[i] * 3],
        vertices[indices[i] * 3 + 1],
        vertices[indices[i] * 3 + 2]
      ]

      const v2 = [
        vertices[indices[i + 1] * 3],
        vertices[indices[i + 1] * 3 + 1],
        vertices[indices[i + 1] * 3 + 2]
      ]

      const v3 = [
        vertices[indices[i + 2] * 3],
        vertices[indices[i + 2] * 3 + 1],
        vertices[indices[i + 2] * 3 + 2]
      ]

      // Sort vertices to make the comparison order-invariant
      const sortedVertices = [v1, v2, v3].sort((a, b) => {
        // Compare x, then y, then z
        if (a[0] !== b[0]) return a[0] - b[0]
        if (a[1] !== b[1]) return a[1] - b[1]
        return a[2] - b[2]
      })

      // Create a string representation of the triangle
      const triangleStr = JSON.stringify(sortedVertices)
      triangles.add(triangleStr)
    }

    return triangles
  }

  it('should encode and decode a mesh correctly', async () => {
    // Create an encoded mesh
    const vertexCount = mesh.vertices.length / 3
    const vertexSize = 3 * 4 // 3 floats * 4 bytes per float

    // Encode the vertex buffer
    const encodedVertices = MeshoptEncoder.encodeVertexBuffer(
      new Uint8Array(mesh.vertices.buffer),
      vertexCount,
      vertexSize
    )

    // Encode the index buffer
    const encodedIndices = MeshoptEncoder.encodeIndexBuffer(
      new Uint8Array(mesh.indices!.buffer),
      mesh.indices!.length,
      4 // 4 bytes per index (Uint32Array)
    )

    // Create the encoded mesh
    const encodedMesh: EncodedMesh = {
      vertices: encodedVertices,
      indices: encodedIndices,
      vertex_count: vertexCount,
      vertex_size: vertexSize,
      index_count: mesh.indices!.length,
      index_size: 4
    }

    // Decode the mesh
    const decodedVertices = MeshUtils.decodeVertexBuffer(
      encodedMesh.vertex_count,
      encodedMesh.vertex_size,
      encodedMesh.vertices
    )

    const decodedIndices = MeshUtils.decodeIndexBuffer(
      encodedMesh.index_count!,
      encodedMesh.index_size,
      encodedMesh.indices!
    )

    const decodedMesh: Mesh = {
      vertices: decodedVertices,
      indices: decodedIndices
    }

    // Check that the vertices match
    expect(decodedMesh.vertices.length).toBe(mesh.vertices.length)
    for (let i = 0; i < mesh.vertices.length; i++) {
      expect(decodedMesh.vertices[i]).toBeCloseTo(mesh.vertices[i], 4)
    }

    // Check that the indices match
    expect(decodedMesh.indices!.length).toBe(mesh.indices!.length)

    // Instead of checking each index individually, check that the triangles match
    const originalTriangles = getTrianglesSet(mesh.vertices, mesh.indices!)
    const decodedTriangles = getTrianglesSet(decodedMesh.vertices, decodedMesh.indices!)

    expect(decodedTriangles.size).toBe(originalTriangles.size)

    for (const triangle of originalTriangles) {
      expect(decodedTriangles.has(triangle)).toBe(true)
    }
  })

  it('should convert mesh to THREE.js BufferGeometry correctly', () => {
    // Convert the mesh to a THREE.js BufferGeometry
    const geometry = MeshUtils.convertToBufferGeometry(mesh)

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

  it('should load mesh from zip correctly', async () => {
    // Create a zip file with the mesh data
    const zip = new JSZip()

    // Create mesh size metadata
    const MeshSize = {
      vertex_count: mesh.vertices.length / 3,
      vertex_size: 3 * 4, // 3 floats * 4 bytes per float
      index_count: mesh.indices!.length,
      index_size: 4 // 4 bytes per index (Uint32Array)
    }

    // Add file metadata with mesh size
    const fileMetadata = {
      class_name: 'Mesh',
      module_name: 'meshly',
      mesh_size: MeshSize
    }

    zip.file('metadata.json', JSON.stringify(fileMetadata))

    // Encode and add vertex data
    const encodedVertices = MeshoptEncoder.encodeVertexBuffer(
      new Uint8Array(mesh.vertices.buffer),
      MeshSize.vertex_count,
      MeshSize.vertex_size
    )

    zip.file('mesh/vertices.bin', encodedVertices)

    // Encode and add index data
    const encodedIndices = MeshoptEncoder.encodeIndexBuffer(
      new Uint8Array(mesh.indices!.buffer),
      MeshSize.index_count,
      MeshSize.index_size
    )

    zip.file('mesh/indices.bin', encodedIndices)

    // Generate the zip file
    const zipData = await zip.generateAsync({ type: 'arraybuffer' })

    // Load the mesh from the zip file
    const loadedMesh = await MeshUtils.loadMeshFromZip(zipData)

    // Check that the vertices match
    expect(loadedMesh.vertices.length).toBe(mesh.vertices.length)
    for (let i = 0; i < mesh.vertices.length; i++) {
      expect(loadedMesh.vertices[i]).toBeCloseTo(mesh.vertices[i], 4)
    }

    // Check that the indices match
    expect(loadedMesh.indices!.length).toBe(mesh.indices!.length)

    // Instead of checking each index individually, check that the triangles match

    // Check that the triangles match
    const originalTriangles = getTrianglesSet(mesh.vertices, mesh.indices!)
    const loadedTriangles = getTrianglesSet(loadedMesh.vertices, loadedMesh.indices!)

    expect(loadedTriangles.size).toBe(originalTriangles.size)

    for (const triangle of originalTriangles) {
      expect(loadedTriangles.has(triangle)).toBe(true)
    }
  })
})