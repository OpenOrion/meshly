import JSZip from 'jszip'
import { MeshoptDecoder, MeshoptEncoder } from "meshoptimizer"
import * as THREE from 'three'
import { ArrayMetadata, ArrayUtils, EncodedArray } from './array'


/**
 * Types for the mesh decoder library
 */

/**
 * Size metadata for a mesh
 */
export interface MeshSize {
  vertex_count: number
  vertex_size: number
  index_count: number | null
  index_size: number
}

/**
 * File metadata for a mesh
 */
export interface MeshMetadata {
  class_name: string
  module_name: string
  field_data?: any
  mesh_size: MeshSize
}

/**
 * Options for decoding a mesh
 */
export interface DecodeMeshOptions {
  /**
   * Whether to normalize the mesh to fit within a unit cube
   * @default false
   */
  normalize?: boolean

  /**
   * Whether to compute normals if they don't exist
   * @default true
   */
  computeNormals?: boolean
}

/**
 * Mesh interface representing a 3D mesh
 */
export interface Mesh {
  /**
   * Vertex data as a Float32Array
   */
  vertices: Float32Array

  /**
   * Index data as a Uint32Array (optional)
   */
  indices?: Uint32Array

  /**
   * Additional properties that can be any type of array
   */
  [key: string]: Float32Array | Uint32Array | undefined | any
}

/**
 * Encoded mesh data
 */
export interface EncodedMesh {
  /**
   * Encoded vertex buffer
   */
  vertices: Uint8Array

  /**
   * Encoded index buffer (optional)
   */
  indices?: Uint8Array

  /**
   * Number of vertices
   */
  vertex_count: number

  /**
   * Size of each vertex in bytes
   */
  vertex_size: number

  /**
   * Number of indices (optional)
   */
  index_count?: number | null

  /**
   * Size of each index in bytes
   */
  index_size: number

  /**
   * Dictionary of additional encoded arrays
   */
  arrays?: Record<string, EncodedArray>
}

/**
 * Utility class for mesh operations
 */
export class MeshUtils {

  /**
   * Encode a mesh for efficient transmission
   *
   * @param mesh The mesh to encode
   * @returns EncodedMesh object with encoded vertices, indices, and arrays
   */
  static encode(mesh: Mesh): EncodedMesh {

    if (!(mesh.vertices instanceof Float32Array)) {
      throw new Error('Array must be a Float32Array')
    }
    const vertexCount = mesh.vertices.length / 3
    const vertexSize = 4 * 3 // 4 bytes per float, 3 floats per vertex
    const encodedVertices = MeshoptEncoder.encodeVertexBuffer(
      new Uint8Array(mesh.vertices.buffer),
      vertexCount,
      vertexSize
    )

    if ((!mesh.indices || mesh.indices instanceof Int32Array)) {
      throw new Error('Array must be a Float32Array')
    }

    // Encode index buffer if present
    let encodedIndices: Uint8Array | undefined
    let indexCount: number | null = null
    const indexSize = 4 // 4 bytes per index (Uint32Array)
    if (mesh.indices) {
      indexCount = mesh.indices.length
      encodedIndices = MeshoptEncoder.encodeIndexSequence(
        new Uint8Array(mesh.indices.buffer),
        indexCount,
        indexSize
      )

    }

    // Encode additional arrays
    const encodedArrays: Record<string, EncodedArray> = {}

    // Encode all additional Float32Array properties
    for (const key in mesh) {
      if (key !== 'vertices' && key !== 'indices' && mesh[key] instanceof Float32Array) {
        encodedArrays[key] = ArrayUtils.encodeArray(mesh[key])
      }
    }

    // Create and return the encoded mesh
    return {
      vertices: encodedVertices,
      indices: encodedIndices,
      vertex_count: vertexCount,
      vertex_size: vertexSize,
      index_count: indexCount,
      index_size: indexSize,
      arrays: encodedArrays
    }
  }

  /**
   * Decode an encoded mesh
   *
   * @param MeshClass The mesh class constructor (not used in TypeScript version, included for API compatibility)
   * @param encodedMesh EncodedMesh object to decode
   * @returns Decoded Mesh object
   */
  static decode<MeshType extends Mesh>(encodedMesh: EncodedMesh): MeshType {

    // Create the output buffer
    const verticesUint8 = new Uint8Array(encodedMesh.vertex_count * encodedMesh.vertex_size)
    MeshoptDecoder.decodeVertexBuffer(
      verticesUint8,
      encodedMesh.vertex_count,
      encodedMesh.vertex_size,
      encodedMesh.vertices
    )
    const vertices = new Float32Array(verticesUint8.buffer)



    // Decode index buffer if present
    let indices: Uint32Array | undefined
    if (encodedMesh.indices && encodedMesh.index_count) {
      const indicesUint8 = new Uint8Array(encodedMesh.index_count * encodedMesh.index_size)
      MeshoptDecoder.decodeIndexSequence(
        indicesUint8,
        encodedMesh.index_count,
        encodedMesh.index_size,
        encodedMesh.indices
      )
      indices = new Uint32Array(indicesUint8.buffer)
    }


    // Create the result object
    const result = {
      vertices,
      indices
    } as MeshType

    // Decode additional arrays if present
    if (encodedMesh.arrays) {
      for (const [name, encodedArray] of Object.entries(encodedMesh.arrays)) {
        const decodedArray = ArrayUtils.decodeArray(
          encodedArray.data,
          {
            shape: encodedArray.shape,
            dtype: encodedArray.dtype,
            itemsize: encodedArray.itemsize
          }
        )

        // Add the decoded array to the result
        Object.assign(result, { [name]: decodedArray })
      }
    }

    return result
  }

  /**
   * Extracts and decodes a mesh from a zip file
   * 
   * @param zipData Zip file data as an ArrayBuffer
   * @returns Promise that resolves to the decoded mesh
   */
  static async loadMeshFromZip<MeshType extends Mesh>(zipData: ArrayBuffer): Promise<MeshType> {
    // Load the zip file
    const zip = await JSZip.loadAsync(zipData)

    // Extract the file metadata
    const metadataJson = await zip.file('metadata.json')?.async('string')
    if (!metadataJson) {
      throw new Error('File metadata not found in zip file')
    }
    const metadata: MeshMetadata = JSON.parse(metadataJson)

    // Get mesh size metadata
    const meshSize: MeshSize = metadata.mesh_size

    // Extract the vertex data
    const vertexData = await zip.file('mesh/vertices.bin')?.async('uint8array')
    if (!vertexData) {
      throw new Error('Vertex data not found in zip file')
    }

    // Extract the index data if present
    let indexData: Uint8Array | undefined
    if (meshSize.index_count !== null) {
      indexData = await zip.file('mesh/indices.bin')?.async('uint8array') || undefined
    }


    // Extract additional arrays
    const arrays: Record<string, EncodedArray> = {}
    const arrayFiles = zip.files ? Object.keys(zip.files)
      .filter(name => name.startsWith('arrays/') && name.endsWith('.bin'))
      .map(name => name.split('/')[1].split('.')[0]) : []

    for (const arrayName of arrayFiles) {
      // Extract the array metadata
      const arrayMetadataJson = await zip.file(`arrays/${arrayName}_metadata.json`)?.async('string')
      if (!arrayMetadataJson) {
        continue
      }

      const arrayMetadata: ArrayMetadata = JSON.parse(arrayMetadataJson)

      // Extract the array data
      const arrayData = await zip.file(`arrays/${arrayName}.bin`)?.async('uint8array')
      if (!arrayData) {
        continue
      }

      // Add the encoded array
      arrays[arrayName] = {
        data: arrayData,
        shape: arrayMetadata.shape,
        dtype: arrayMetadata.dtype,
        itemsize: arrayMetadata.itemsize
      }
    }

    // Create the EncodedMesh object with all collected data
    const encodedMesh: EncodedMesh = {
      vertices: vertexData,
      vertex_count: meshSize.vertex_count,
      vertex_size: meshSize.vertex_size,
      index_count: meshSize.index_count,
      index_size: meshSize.index_size,
      indices: indexData,
      arrays: Object.keys(arrays).length > 0 ? arrays : undefined
    }

    // Decode the mesh using our decode function
    const result = MeshUtils.decode<MeshType>(encodedMesh)

    // Add any additional field data from the metadata
    if (metadata.field_data) {
      Object.assign(result, metadata.field_data)
    }

    return result
  }

  /**
   * Converts a mesh to a THREE.js BufferGeometry
   * 
   * @param mesh The mesh to convert
   * @param options Options for the conversion
   * @returns THREE.js BufferGeometry
   */
  static convertToBufferGeometry(
    mesh: Mesh,
    options: DecodeMeshOptions = {}
  ): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry()

    // Set default options
    const opts = {
      normalize: false,
      computeNormals: false,
      ...options
    }

    // Add vertices
    const vertices = mesh.vertices

    // If normalize is true, normalize the vertices to fit within a unit cube
    let normalizedVertices = vertices
    if (opts.normalize) {
      normalizedVertices = MeshUtils.normalizeVertices(vertices)
    }

    // Add the vertices to the geometry
    geometry.setAttribute('position', new THREE.BufferAttribute(normalizedVertices, 3))

    // Add indices if they exist
    if (mesh.indices) {
      geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1))
    }

    // Add normals if they exist, otherwise compute them if requested
    if (mesh.normals) {
      geometry.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3))
    } else if (opts.computeNormals) {
      geometry.computeVertexNormals()
    }

    // Add colors if they exist
    if (mesh.colors) {
      // Check if colors have 3 or 4 components (RGB or RGBA)
      const itemSize = mesh.colors.length / (vertices.length / 3)
      geometry.setAttribute('color', new THREE.BufferAttribute(mesh.colors, itemSize))
    }

    // Add UVs if they exist
    if (mesh.uvs) {
      // Check if UVs have 2 or 3 components
      const itemSize = mesh.uvs.length / (vertices.length / 3)
      geometry.setAttribute('uv', new THREE.BufferAttribute(mesh.uvs, itemSize))
    }

    return geometry
  }

  /**
   * Normalizes vertices to fit within a unit cube centered at the origin
   * 
   * @param vertices The vertices to normalize
   * @returns Normalized vertices
   */
  static normalizeVertices(vertices: Float32Array): Float32Array {
    // Find the bounding box
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i]
      const y = vertices[i + 1]
      const z = vertices[i + 2]

      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      minZ = Math.min(minZ, z)

      maxX = Math.max(maxX, x)
      maxY = Math.max(maxY, y)
      maxZ = Math.max(maxZ, z)
    }

    // Calculate the center and size
    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    const centerZ = (minZ + maxZ) / 2

    const sizeX = maxX - minX
    const sizeY = maxY - minY
    const sizeZ = maxZ - minZ

    // Find the maximum dimension
    const maxSize = Math.max(sizeX, sizeY, sizeZ)

    // Create a new array for the normalized vertices
    const normalizedVertices = new Float32Array(vertices.length)

    // Normalize the vertices
    for (let i = 0; i < vertices.length; i += 3) {
      normalizedVertices[i] = (vertices[i] - centerX) / maxSize
      normalizedVertices[i + 1] = (vertices[i + 1] - centerY) / maxSize
      normalizedVertices[i + 2] = (vertices[i + 2] - centerZ) / maxSize
    }

    return normalizedVertices
  }


  /**
   * Main function to load a mesh from a zip file and convert it to a THREE.js BufferGeometry
   *
   * @param zipData Zip file data as an ArrayBuffer
   * @param options Options for the conversion
   * @returns Promise that resolves to a THREE.js BufferGeometry
   */
  static async loadZipAsBufferGeometry(
    zipData: ArrayBuffer,
    options?: DecodeMeshOptions
  ): Promise<THREE.BufferGeometry> {
    const mesh = await MeshUtils.loadMeshFromZip(zipData)
    return MeshUtils.convertToBufferGeometry(mesh, options)
  }
}