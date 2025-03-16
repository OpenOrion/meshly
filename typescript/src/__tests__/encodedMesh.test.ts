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

    // Create the encoded mesh
    const encodedMesh = MeshUtils.encode(mesh)
    // Decode the mesh
    const decodedMesh = MeshUtils.decode(encodedMesh)

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

  it('should use the new encode and decode functions correctly', () => {
    // Add normals to the mesh
    const normals = new Float32Array([
      0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
      1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0
    ])
    const meshWithNormals: Mesh = {
      ...mesh,
      normals
    }

    // Encode the mesh using the new encode function
    const encodedMesh = MeshUtils.encode(meshWithNormals)

    // Check that the encoded mesh has the expected properties
    expect(encodedMesh.vertex_count).toBe(meshWithNormals.vertices.length / 3)
    expect(encodedMesh.vertex_size).toBe(12) // 3 floats * 4 bytes
    expect(encodedMesh.index_count).toBe(meshWithNormals.indices!.length)
    expect(encodedMesh.index_size).toBe(4)
    expect(encodedMesh.vertices).toBeDefined()
    expect(encodedMesh.indices).toBeDefined()
    expect(encodedMesh.arrays).toBeDefined()
    expect(encodedMesh.arrays!.normals).toBeDefined()

    // Decode the mesh using the new decode function
    // In TypeScript, we can't use the Mesh type as a value, so we pass null
    const decodedMesh = MeshUtils.decode(encodedMesh)

    // Check that the decoded mesh has the expected properties
    expect(decodedMesh.vertices.length).toBe(meshWithNormals.vertices.length)
    expect(decodedMesh.indices!.length).toBe(meshWithNormals.indices!.length)
    expect(decodedMesh.normals).toBeDefined()
    expect(decodedMesh.normals!.length).toBe(normals.length)

    // Check that the vertices match
    for (let i = 0; i < meshWithNormals.vertices.length; i++) {
      expect(decodedMesh.vertices[i]).toBeCloseTo(meshWithNormals.vertices[i], 4)
    }

    // Instead of checking each index individually, check that the triangles match
    // This is more reliable than comparing indices directly
    const originalTriangles = getTrianglesSet(meshWithNormals.vertices, meshWithNormals.indices!)
    const decodedTriangles = getTrianglesSet(decodedMesh.vertices, decodedMesh.indices!)
    
    expect(decodedTriangles.size).toBe(originalTriangles.size)
    
    for (const triangle of originalTriangles) {
      expect(decodedTriangles.has(triangle)).toBe(true)
    }

    // Check that the normals match
    for (let i = 0; i < normals.length; i++) {
      expect(decodedMesh.normals![i]).toBeCloseTo(normals[i], 4)
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

    const encondedMesh = MeshUtils.encode(mesh)

    zip.file('mesh/vertices.bin', encondedMesh.vertices)
    if (encondedMesh.indices){
      zip.file('mesh/indices.bin', encondedMesh.indices)
    }

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