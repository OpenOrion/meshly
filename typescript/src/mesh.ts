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
   * Index data as a Uint32Array (flattened 1D array)
   */
  indices?: Uint32Array

  /**
   * Size of each polygon (number of vertices per polygon)
   */
  indexSizes?: Uint32Array

  /**
   * Cell type identifier for each polygon, corresponding to indexSizes.
   * Common VTK cell types include:
   * - 1: Vertex, 3: Line, 5: Triangle, 9: Quad, 10: Tetra, 12: Hexahedron, 13: Wedge, 14: Pyramid
   */
  cellTypes?: Uint32Array

  /**
   * Mesh dimension (2D or 3D)
   */
  dim?: number

  /**
   * Flattened marker node indices
   */
  markerIndices?: Record<string, Uint32Array>

  /**
   * Offset indices to reconstruct individual marker elements
   */
  markerOffsets?: Record<string, Uint32Array>

  /**
   * VTK cell types for each marker element
   */
  markerTypes?: Record<string, Uint8Array>

  /**
   * Optional markers as list of lists, auto-converted to flattened structure
   */
  markers?: Record<string, number[][]>

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
   * Encoded polygon sizes (optional, for backwards compatibility)
   */
  indexSizes?: Uint8Array

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
   * Get polygon count from a mesh
   */
  static getPolygonCount(mesh: Mesh): number {
    return mesh.indexSizes ? mesh.indexSizes.length : 0
  }

  /**
   * Check if all polygons have the same number of vertices
   */
  static isUniformPolygons(mesh: Mesh): boolean {
    if (!mesh.indexSizes) {
      return true // No polygon info means uniform (legacy)
    }
    const firstSize = mesh.indexSizes[0]
    return mesh.indexSizes.every(size => size === firstSize)
  }

  /**
   * Infer VTK cell types from polygon sizes
   * @param indexSizes Array of polygon sizes
   * @returns Array of VTK cell type identifiers
   */
  static inferCellTypes(indexSizes: Uint32Array): Uint32Array {
    const cellTypes = new Uint32Array(indexSizes.length)
    
    for (let i = 0; i < indexSizes.length; i++) {
      const size = indexSizes[i]
      switch (size) {
        case 1:
          cellTypes[i] = 1 // VTK_VERTEX
          break
        case 2:
          cellTypes[i] = 3 // VTK_LINE
          break
        case 3:
          cellTypes[i] = 5 // VTK_TRIANGLE
          break
        case 4:
          cellTypes[i] = 9 // VTK_QUAD
          break
        case 5:
          cellTypes[i] = 14 // VTK_PYRAMID
          break
        case 6:
          cellTypes[i] = 13 // VTK_WEDGE
          break
        case 8:
          cellTypes[i] = 12 // VTK_HEXAHEDRON
          break
        default:
          cellTypes[i] = 7 // VTK_POLYGON (generic polygon)
          break
      }
    }
    
    return cellTypes
  }

  /**
   * Get or infer cell types for a mesh
   * @param mesh The mesh to get cell types for
   * @returns Array of VTK cell type identifiers
   */
  static getCellTypes(mesh: Mesh): Uint32Array | undefined {
    if (mesh.cellTypes) {
      return mesh.cellTypes
    }
    
    if (mesh.indexSizes) {
      return MeshUtils.inferCellTypes(mesh.indexSizes)
    }
    
    return undefined
  }

  /**
   * Get indices in their original polygon structure
   */
  static getPolygonIndices(mesh: Mesh): Uint32Array[] | Uint32Array {
    if (!mesh.indices) {
      return []
    }

    if (!mesh.indexSizes) {
      // Legacy format - assume triangles
      if (mesh.indices.length % 3 === 0) {
        const result = new Uint32Array(mesh.indices.length)
        result.set(mesh.indices)
        return result
      } else {
        throw new Error('Cannot determine polygon structure without indexSizes')
      }
    }

    if (MeshUtils.isUniformPolygons(mesh)) {
      // Uniform case: return as 2D-like structure (but we'll return the flat array for now)
      return mesh.indices
    } else {
      // Mixed case: return as array of arrays
      const result: Uint32Array[] = []
      let offset = 0
      for (const size of mesh.indexSizes) {
        const polygon = new Uint32Array(size)
        polygon.set(mesh.indices.subarray(offset, offset + size))
        result.push(polygon)
        offset += size
      }
      return result
    }
  }

  /**
   * Recursively extract arrays from nested dictionary structures
   *
   * @param obj The object to extract arrays from
   * @param prefix The current path prefix for nested keys
   * @returns Record mapping dotted paths to typed arrays
   */
  private static extractNestedArrays(obj: any, prefix: string = ''): Record<string, Float32Array | Uint32Array> {
    const arrays: Record<string, Float32Array | Uint32Array> = {}
    
    if (typeof obj === 'object' && obj !== null && !ArrayBuffer.isView(obj)) {
      for (const [key, value] of Object.entries(obj)) {
        const nestedKey = prefix ? `${prefix}.${key}` : key
        
        if (value instanceof Float32Array || value instanceof Uint32Array) {
          arrays[nestedKey] = value
        } else if (typeof value === 'object' && value !== null && !ArrayBuffer.isView(value)) {
          Object.assign(arrays, MeshUtils.extractNestedArrays(value, nestedKey))
        }
      }
    }
    
    return arrays
  }

  /**
   * Reconstruct nested dictionary structure from dotted keys
   *
   * @param arrays Record of arrays with dotted keys
   * @returns Reconstructed nested object structure
   */
  private static reconstructDictionaries(arrays: Record<string, Float32Array | Uint32Array>): Record<string, any> {
    const result: Record<string, any> = {}
    const directArrays: Record<string, Float32Array | Uint32Array> = {}
    
    for (const [key, array] of Object.entries(arrays)) {
      if (key.includes('.')) {
        // This is a nested array
        const parts = key.split('.')
        let current = result
        
        // Navigate/create the nested structure
        for (let i = 0; i < parts.length - 1; i++) {
          const part = parts[i]
          if (!(part in current)) {
            current[part] = {}
          }
          current = current[part]
        }
        
        // Set the final array
        current[parts[parts.length - 1]] = array
      } else {
        // This is a direct array field
        directArrays[key] = array
      }
    }
    
    return { ...result, ...directArrays }
  }

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

    // Encode additional arrays, including nested arrays from dictionaries
    const encodedArrays: Record<string, EncodedArray> = {}

    // Extract all arrays (including nested ones)
    const allArrays = MeshUtils.extractNestedArrays(mesh)
    
    for (const [key, array] of Object.entries(allArrays)) {
      if (key !== 'vertices' && key !== 'indices') {
        if (array instanceof Float32Array) {
          encodedArrays[key] = ArrayUtils.encodeArray(array)
        } else if (array instanceof Uint32Array) {
          encodedArrays[key] = ArrayUtils.encodeArray(array)
        }
      }
    }

    // Handle indexSizes as a special case - store in arrays if present
    if (mesh.indexSizes) {
      encodedArrays['indexSizes'] = ArrayUtils.encodeArray(mesh.indexSizes)
    }

    // Handle cellTypes as a special case - store in arrays if present
    if (mesh.cellTypes) {
      encodedArrays['cellTypes'] = ArrayUtils.encodeArray(mesh.cellTypes)
    }

    // Handle marker arrays as special cases - store in arrays if present
    if (mesh.markerIndices) {
      for (const [name, array] of Object.entries(mesh.markerIndices)) {
        encodedArrays[`markerIndices.${name}`] = ArrayUtils.encodeArray(array)
      }
    }

    if (mesh.markerOffsets) {
      for (const [name, array] of Object.entries(mesh.markerOffsets)) {
        encodedArrays[`markerOffsets.${name}`] = ArrayUtils.encodeArray(array)
      }
    }

    if (mesh.markerTypes) {
      for (const [name, array] of Object.entries(mesh.markerTypes)) {
        // Convert Uint8Array to Uint32Array for encoding
        const uint32Array = new Uint32Array(array.length)
        for (let i = 0; i < array.length; i++) {
          uint32Array[i] = array[i]
        }
        encodedArrays[`markerTypes.${name}`] = ArrayUtils.encodeArray(uint32Array)
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
      indexSizes: undefined, // Not used anymore - stored in arrays
      arrays: encodedArrays
    }
  }

  /**
   * Decode an encoded mesh
   *
   * @param encodedMesh EncodedMesh object to decode
   * @returns Decoded Mesh object
   */
  static decode<MeshType extends Mesh>(
    encodedMesh: EncodedMesh
  ): MeshType {

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
    const decodedArrays: Record<string, Float32Array | Uint32Array> = {}
    let indexSizes: Uint32Array | undefined
    let cellTypes: Uint32Array | undefined
    const markerIndices: Record<string, Uint32Array> = {}
    const markerOffsets: Record<string, Uint32Array> = {}
    const markerTypes: Record<string, Uint8Array> = {}
    
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
        
        if (name === 'indexSizes') {
          // Special handling for indexSizes
          indexSizes = decodedArray as Uint32Array
        } else if (name === 'cellTypes') {
          // Special handling for cellTypes
          cellTypes = decodedArray as Uint32Array
        } else if (name.startsWith('markerIndices.')) {
          // Special handling for marker indices
          const markerName = name.substring('markerIndices.'.length)
          markerIndices[markerName] = decodedArray as Uint32Array
        } else if (name.startsWith('markerOffsets.')) {
          // Special handling for marker offsets
          const markerName = name.substring('markerOffsets.'.length)
          markerOffsets[markerName] = decodedArray as Uint32Array
        } else if (name.startsWith('markerTypes.')) {
          // Special handling for marker types - convert back from Uint32Array to Uint8Array
          const markerName = name.substring('markerTypes.'.length)
          const uint32Array = decodedArray as Uint32Array
          const uint8Array = new Uint8Array(uint32Array.length)
          for (let i = 0; i < uint32Array.length; i++) {
            uint8Array[i] = uint32Array[i]
          }
          markerTypes[markerName] = uint8Array
        } else {
          decodedArrays[name] = decodedArray
        }
      }
    }

    // Add indexSizes and cellTypes to result if present
    if (indexSizes) {
      result.indexSizes = indexSizes
    }
    if (cellTypes) {
      result.cellTypes = cellTypes
    }

    // Add marker arrays to result if present
    if (Object.keys(markerIndices).length > 0) {
      result.markerIndices = markerIndices
    }
    if (Object.keys(markerOffsets).length > 0) {
      result.markerOffsets = markerOffsets
    }
    if (Object.keys(markerTypes).length > 0) {
      result.markerTypes = markerTypes
    }

    // Reconstruct dictionary structure from decoded arrays
    const reconstructedStructure = MeshUtils.reconstructDictionaries(decodedArrays)
    
    // Merge reconstructed structure into result
    Object.assign(result, reconstructedStructure)

    return result
  }

  /**
   * Extract non-array values from nested structures for metadata storage
   * @param obj Object to extract from
   * @param prefix Current path prefix
   * @returns Object with non-array values only
   */
  private static extractNonArrays(obj: any, prefix: string = ''): any {
    if (typeof obj === 'object' && obj !== null && !ArrayBuffer.isView(obj)) {
      const result: any = {}
      for (const [key, value] of Object.entries(obj)) {
        const nestedKey = prefix ? `${prefix}.${key}` : key
        if (ArrayBuffer.isView(value)) {
          // Skip arrays - they're stored separately
          continue
        } else if (typeof value === 'object' && value !== null && !ArrayBuffer.isView(value)) {
          // Recursively process nested objects
          const nestedResult = MeshUtils.extractNonArrays(value, nestedKey)
          if (nestedResult && Object.keys(nestedResult).length > 0) {
            result[key] = nestedResult
          }
        } else {
          // Include non-array values
          result[key] = value
        }
      }
      return Object.keys(result).length > 0 ? result : null
    } else if (ArrayBuffer.isView(obj)) {
      // Skip arrays
      return null
    } else {
      // Include non-array values
      return obj
    }
  }

  /**
   * Save a mesh to a zip file (returns zip data as Uint8Array)
   *
   * @param mesh The mesh to save
   * @returns Promise that resolves to zip file data as Uint8Array
   */
  static async saveMeshToZip<MeshType extends Mesh>(mesh: MeshType): Promise<Uint8Array> {
    const zip = new JSZip()
    
    // Encode the mesh
    const encodedMesh = MeshUtils.encode(mesh)
    
    // Save mesh data
    zip.file('mesh/vertices.bin', encodedMesh.vertices)
    if (encodedMesh.indices) {
      zip.file('mesh/indices.bin', encodedMesh.indices)
    }
    
    // Extract non-array field data for metadata
    const fieldData: any = {}
    const allArrays = MeshUtils.extractNestedArrays(mesh)
    const arrayFieldNames = new Set([...Object.keys(allArrays), 'vertices', 'indices', 'indexSizes'])
    
    for (const [fieldName, fieldValue] of Object.entries(mesh)) {
      if (arrayFieldNames.has(fieldName)) {
        continue // Skip array fields
      }
      
      if (typeof fieldValue === 'object' && fieldValue !== null && !ArrayBuffer.isView(fieldValue)) {
        // Extract non-array values from dictionaries
        const nonArrayContent = MeshUtils.extractNonArrays(fieldValue, fieldName)
        if (nonArrayContent) {
          fieldData[fieldName] = nonArrayContent
        }
      } else {
        // Include scalar fields directly
        fieldData[fieldName] = fieldValue
      }
    }
    
    // Create metadata
    const meshSize: MeshSize = {
      vertex_count: encodedMesh.vertex_count,
      vertex_size: encodedMesh.vertex_size,
      index_count: encodedMesh.index_count ?? null,
      index_size: encodedMesh.index_size
    }
    
    const metadata: MeshMetadata = {
      class_name: 'FlexibleMesh',  // Use FlexibleMesh class name for Python compatibility
      module_name: '__main__',
      field_data: Object.keys(fieldData).length > 0 ? fieldData : undefined,
      mesh_size: meshSize
    }
    
    // Save array data
    if (encodedMesh.arrays) {
      for (const [name, encodedArray] of Object.entries(encodedMesh.arrays)) {
        // Convert dots to nested directory structure, each array gets its own directory
        const arrayPath = name.replace(/\./g, '/')
        
        zip.file(`arrays/${arrayPath}/array.bin`, encodedArray.data)
        
        // Save array metadata
        const arrayMetadata: ArrayMetadata = {
          shape: encodedArray.shape,
          dtype: encodedArray.dtype,
          itemsize: encodedArray.itemsize
        }
        zip.file(`arrays/${arrayPath}/metadata.json`, JSON.stringify(arrayMetadata, null, 2))
      }
    }
    
    // Save general metadata
    zip.file('metadata.json', JSON.stringify(metadata, null, 2))
    
    // Generate and return zip data
    const zipData = await zip.generateAsync({ type: 'uint8array' })
    return zipData
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
      .filter(name => name.startsWith('arrays/') && name.endsWith('/array.bin')) : []

    for (const arrayFileName of arrayFiles) {
      // Extract the array path by removing "arrays/" prefix and "/array.bin" suffix
      const arrayPath = arrayFileName.slice(7, -10) // Remove "arrays/" and "/array.bin"

      // Extract the array metadata
      const arrayMetadataJson = await zip.file(`arrays/${arrayPath}/metadata.json`)?.async('string')
      if (!arrayMetadataJson) {
        continue
      }

      const arrayMetadata: ArrayMetadata = JSON.parse(arrayMetadataJson)

      // Extract the array data
      const arrayData = await zip.file(arrayFileName)?.async('uint8array')
      if (!arrayData) {
        continue
      }

      // Convert directory path back to dotted name
      const originalName = arrayPath.replace(/\//g, '.')

      // Add the encoded array with original name
      arrays[originalName] = {
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

    // Add any additional field data from the metadata, merging with existing dict fields
    if (metadata.field_data) {
      for (const [fieldName, fieldValue] of Object.entries(metadata.field_data)) {
        const existingValue = (result as any)[fieldName]
        if (existingValue && typeof existingValue === 'object' && typeof fieldValue === 'object' &&
            !ArrayBuffer.isView(existingValue) && !ArrayBuffer.isView(fieldValue)) {
          // Merge non-array values into existing dictionary structure
          function mergeDicts(target: any, source: any): void {
            for (const [key, value] of Object.entries(source)) {
              if (key in target && typeof target[key] === 'object' && typeof value === 'object' &&
                  !ArrayBuffer.isView(target[key]) && !ArrayBuffer.isView(value)) {
                mergeDicts(target[key], value)
              } else {
                target[key] = value
              }
            }
          }
          
          mergeDicts(existingValue, fieldValue)
        } else {
          // Set scalar fields directly
          (result as any)[fieldName] = fieldValue
        }
      }
    }

    return result
  }

  /**
   * Triangulates a polygon using fan triangulation
   * @param indices The polygon indices
   * @returns Array of triangulated indices
   */
  private static triangulatePoly(indices: Uint32Array): Uint32Array {
    if (indices.length < 3) {
      throw new Error('Polygon must have at least 3 vertices')
    }
    
    if (indices.length === 3) {
      // Already a triangle
      return indices
    }
    
    if (indices.length === 4) {
      // Quad: split into 2 triangles
      return new Uint32Array([
        indices[0], indices[1], indices[2],
        indices[0], indices[2], indices[3]
      ])
    }
    
    // General polygon: fan triangulation from first vertex
    const triangulated: number[] = []
    for (let i = 1; i < indices.length - 1; i++) {
      triangulated.push(indices[0], indices[i], indices[i + 1])
    }
    
    return new Uint32Array(triangulated)
  }

  /**
   * Converts mesh indices to triangulated indices for THREE.js compatibility
   * @param mesh The mesh to triangulate
   * @returns Triangulated indices
   */
  private static triangulateIndices(mesh: Mesh): Uint32Array {
    if (!mesh.indices) {
      return new Uint32Array()
    }

    // If no indexSizes, assume triangles (legacy format)
    if (!mesh.indexSizes) {
      return mesh.indices
    }

    const triangulatedIndices: number[] = []
    let offset = 0

    for (let i = 0; i < mesh.indexSizes.length; i++) {
      const polygonSize = mesh.indexSizes[i]
      const cellType = mesh.cellTypes ? mesh.cellTypes[i] : undefined
      
      // Extract polygon indices
      const polygonIndices = mesh.indices.slice(offset, offset + polygonSize)
      
      // Skip degenerate polygons
      if (polygonSize < 3) {
        offset += polygonSize
        continue
      }

      // Triangulate based on polygon size and cell type
      let triangulated: Uint32Array
      
      if (polygonSize === 3) {
        // Triangle - no triangulation needed
        triangulated = polygonIndices
      } else if (polygonSize === 4 && (!cellType || cellType === 9)) {
        // Quad (VTK_QUAD = 9) - split into 2 triangles
        triangulated = MeshUtils.triangulatePoly(polygonIndices)
      } else {
        // General polygon - fan triangulation
        triangulated = MeshUtils.triangulatePoly(polygonIndices)
      }
      
      triangulatedIndices.push(...Array.from(triangulated))
      offset += polygonSize
    }

    return new Uint32Array(triangulatedIndices)
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

    // Add indices if they exist, triangulating non-triangular polygons
    if (mesh.indices) {
      const triangulatedIndices = MeshUtils.triangulateIndices(mesh)
      if (triangulatedIndices.length > 0) {
        geometry.setIndex(new THREE.BufferAttribute(triangulatedIndices, 1))
      }
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