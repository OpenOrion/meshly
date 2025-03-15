import JSZip from 'jszip'
import { MeshoptDecoder } from "meshoptimizer"
import * as THREE from 'three'
import { ArrayMetadata, ArrayUtils } from './array'


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
}

/**
 * Utility class for mesh operations
 */
export class MeshUtils {
  /**
   * Decodes a vertex buffer using the meshoptimizer algorithm
   * 
   * @param vertexCount Number of vertices
   * @param vertexSize Size of each vertex in bytes
   * @param data Encoded vertex buffer
   * @returns Decoded vertex buffer as a Float32Array
   */
  static decodeVertexBuffer(vertexCount: number, vertexSize: number, data: Uint8Array): Float32Array {
    // Create the output buffer
    const destUint8Array = new Uint8Array(vertexCount * vertexSize)

    MeshoptDecoder.decodeVertexBuffer(
      destUint8Array,
      vertexCount,
      vertexSize,
      data
    )

    return new Float32Array(destUint8Array.buffer)
  }

  /**
   * Decodes an index buffer using the meshoptimizer algorithm
   * 
   * @param indexCount Number of indices
   * @param indexSize Size of each index in bytes
   * @param data Encoded index buffer
   * @returns Decoded index buffer as a Uint32Array
   */
  static decodeIndexBuffer(indexCount: number, indexSize: number, data: Uint8Array): Uint32Array {
    // Create the output buffer
    const destUint8Array = new Uint8Array(indexCount * indexSize)

    MeshoptDecoder.decodeIndexBuffer(
      destUint8Array,
      indexCount,
      indexSize,
      data
    )

    return new Uint32Array(destUint8Array.buffer)
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

    // Decode the vertex data
    const vertices = MeshUtils.decodeVertexBuffer(
      meshSize.vertex_count,
      meshSize.vertex_size,
      vertexData
    )

    // Extract and decode the index data if it exists
    let indices: Uint32Array | undefined
    if (meshSize.index_count !== null) {
      const indexData = await zip.file('mesh/indices.bin')?.async('uint8array')
      if (indexData) {
        indices = MeshUtils.decodeIndexBuffer(
          meshSize.index_count,
          meshSize.index_size,
          indexData
        )
      }
    }

    // Create the result object
    const result: MeshType = {
      vertices,
      indices,
      ...(metadata.field_data || {})
    }

    // Extract and decode additional arrays
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

      // Decode the array
      const decodedArray = ArrayUtils.decodeArray(arrayData, arrayMetadata)

      // Create a temporary object with the new property
      const tempObj = { [arrayName]: decodedArray }
      
      // Merge the temporary object with the result using Object.assign
      Object.assign(result, tempObj)
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
      computeNormals: true,
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