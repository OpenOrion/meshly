import JSZip from 'jszip'
import { MeshoptDecoder } from "meshoptimizer"
import * as THREE from 'three'
import { ArrayType } from './array'
import { CustomFieldConfig, Packable, PackableMetadata } from './packable'


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
 * Mesh-specific metadata extending PackableMetadata
 */
export interface MeshMetadata extends PackableMetadata {
  mesh_size: MeshSize
  /** Array backend type for vertices/indices (for Python compatibility) */
  array_type?: ArrayType
}

/**
 * Interface representing mesh data that can be passed to the Mesh constructor
 */
export interface MeshData {
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
}

/**
 * Mesh class for loading and working with meshoptimizer-compressed meshes.
 * 
 * Extends Packable to provide mesh-specific decoding for vertices and indices
 * using meshoptimizer's vertex and index buffer decompression.
 * 
 * @example
 * ```typescript
 * // Decode from zip data
 * const mesh = await Mesh.decode(zipData)
 * 
 * // Convert to Three.js BufferGeometry
 * const geometry = mesh.toBufferGeometry()
 * ```
 */
export class Mesh<TData extends MeshData = MeshData> extends Packable<TData> {
  // Declare mesh-specific fields for type safety (assigned via Object.assign in parent)
  declare vertices: Float32Array
  declare indices?: Uint32Array
  declare indexSizes?: Uint32Array
  declare cellTypes?: Uint32Array
  declare dim?: number
  declare markerIndices?: Record<string, Uint32Array>
  declare markerOffsets?: Record<string, Uint32Array>
  declare markerTypes?: Record<string, Uint8Array>
  declare markers?: Record<string, number[][]>

  // ============================================================
  // Custom field decoders for meshoptimizer
  // ============================================================

  /**
   * Decode vertices using meshoptimizer
   */
  private static _decodeVertices(data: Uint8Array, metadata: MeshMetadata): Float32Array {
    const meshSize = metadata.mesh_size
    const verticesUint8 = new Uint8Array(meshSize.vertex_count * meshSize.vertex_size)
    MeshoptDecoder.decodeVertexBuffer(
      verticesUint8,
      meshSize.vertex_count,
      meshSize.vertex_size,
      data
    )
    return new Float32Array(verticesUint8.buffer)
  }

  /**
   * Decode indices using meshoptimizer
   */
  private static _decodeIndices(data: Uint8Array, metadata: MeshMetadata): Uint32Array {
    const meshSize = metadata.mesh_size
    const indicesUint8 = new Uint8Array(meshSize.index_count! * meshSize.index_size)
    MeshoptDecoder.decodeIndexSequence(
      indicesUint8,
      meshSize.index_count!,
      meshSize.index_size,
      data
    )
    return new Uint32Array(indicesUint8.buffer)
  }

  /**
   * Custom field configurations for mesh-specific decoding
   */
  protected static override getCustomFields(): Record<string, CustomFieldConfig<unknown, MeshMetadata>> {
    return {
      vertices: {
        fileName: 'vertices',
        decode: (data, metadata) => Mesh._decodeVertices(data, metadata),
        optional: false
      },
      indices: {
        fileName: 'indices',
        decode: (data, metadata) => Mesh._decodeIndices(data, metadata),
        optional: true
      }
    }
  }

  // ============================================================
  // Override decode to use MeshMetadata
  // ============================================================

  /**
   * Decode a Mesh from zip data
   */
  static override async decode(zipData: ArrayBuffer | Uint8Array): Promise<Mesh> {
    const zip = await JSZip.loadAsync(zipData)
    const metadata = await Packable.loadMetadata<MeshMetadata>(zip)
    const customFieldNames = Mesh.getCustomFieldNames()

    const data: Record<string, unknown> = {}

    // Decode custom fields (vertices, indices)
    await Mesh.decodeCustomFields(zip, metadata, data)

    // Load standard arrays
    await Mesh.loadStandardArrays(zip, data, customFieldNames)

    // Merge non-array fields from metadata
    if (metadata.field_data) {
      Mesh._mergeFieldData(data, metadata.field_data)
    }

    return new Mesh(data as unknown as MeshData)
  }

  // ============================================================
  // Mesh-specific utility methods
  // ============================================================

  /**
   * Get polygon count from this mesh
   */
  getPolygonCount(): number {
    return this.indexSizes ? this.indexSizes.length : 0
  }

  /**
   * Check if all polygons have the same number of vertices
   */
  isUniformPolygons(): boolean {
    if (!this.indexSizes) {
      return true // No polygon info means uniform (legacy)
    }
    const firstSize = this.indexSizes[0]
    return this.indexSizes.every(size => size === firstSize)
  }

  /**
   * Get indices in their original polygon structure
   */
  getPolygonIndices(): Uint32Array[] | Uint32Array {
    if (!this.indices) {
      return []
    }

    if (!this.indexSizes) {
      // Legacy format - assume triangles
      if (this.indices.length % 3 === 0) {
        const result = new Uint32Array(this.indices.length)
        result.set(this.indices)
        return result
      } else {
        throw new Error('Cannot determine polygon structure without indexSizes')
      }
    }

    if (this.isUniformPolygons()) {
      return this.indices
    } else {
      // Mixed case: return as array of arrays
      const result: Uint32Array[] = []
      let offset = 0
      for (const size of this.indexSizes) {
        const polygon = new Uint32Array(size)
        polygon.set(this.indices.subarray(offset, offset + size))
        result.push(polygon)
        offset += size
      }
      return result
    }
  }


  // ============================================================
  // Marker extraction
  // ============================================================

  /**
   * Extract a submesh containing only elements referenced by a specific marker.
   */
  extractByMarker(markerName: string): Mesh {
    if (!this.markerIndices || !(markerName in this.markerIndices)) {
      const availableMarkers = this.markerIndices ? Object.keys(this.markerIndices).join(', ') : 'none'
      throw new Error(`Marker '${markerName}' not found. Available markers: ${availableMarkers}`)
    }

    const markerIndices = this.markerIndices[markerName]
    const markerOffsets = this.markerOffsets?.[markerName]
    const markerTypes = this.markerTypes?.[markerName]

    if (!markerOffsets || !markerTypes) {
      throw new Error(`Marker '${markerName}' is missing offset or type information`)
    }

    // Convert offsets to sizes
    const markerSizes = new Uint32Array(markerOffsets.length)
    for (let i = 0; i < markerOffsets.length - 1; i++) {
      markerSizes[i] = markerOffsets[i + 1] - markerOffsets[i]
    }
    if (markerOffsets.length > 0) {
      markerSizes[markerOffsets.length - 1] = markerIndices.length - markerOffsets[markerOffsets.length - 1]
    }

    // Find all unique vertex indices
    const uniqueVerticesSet = new Set<number>()
    for (let i = 0; i < markerIndices.length; i++) {
      uniqueVerticesSet.add(markerIndices[i])
    }
    const uniqueVertices = new Uint32Array(Array.from(uniqueVerticesSet).sort((a, b) => a - b))

    // Extract vertices
    const extractedVertices = new Float32Array(uniqueVertices.length * 3)
    for (let i = 0; i < uniqueVertices.length; i++) {
      const vertexIndex = uniqueVertices[i]
      extractedVertices[i * 3] = this.vertices[vertexIndex * 3]
      extractedVertices[i * 3 + 1] = this.vertices[vertexIndex * 3 + 1]
      extractedVertices[i * 3 + 2] = this.vertices[vertexIndex * 3 + 2]
    }

    // Remap indices using binary search
    const remappedIndices = new Uint32Array(markerIndices.length)
    for (let i = 0; i < markerIndices.length; i++) {
      let left = 0
      let right = uniqueVertices.length
      const target = markerIndices[i]

      while (left < right) {
        const mid = Math.floor((left + right) / 2)
        if (uniqueVertices[mid] < target) {
          left = mid + 1
        } else {
          right = mid
        }
      }
      remappedIndices[i] = left
    }

    // Convert markerTypes from Uint8Array to Uint32Array
    const cellTypes = new Uint32Array(markerTypes.length)
    for (let i = 0; i < markerTypes.length; i++) {
      cellTypes[i] = markerTypes[i]
    }

    return new Mesh({
      vertices: extractedVertices,
      indices: remappedIndices,
      indexSizes: markerSizes,
      cellTypes: cellTypes,
      dim: this.dim
    })
  }

  // ============================================================
  // THREE.js integration
  // ============================================================

  /**
   * Convert this mesh to a THREE.js BufferGeometry
   */
  toBufferGeometry(): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry()

    geometry.setAttribute('position', new THREE.BufferAttribute(this.vertices, 3))

    if (this.indices && this.indices.length > 0) {
      geometry.setIndex(new THREE.BufferAttribute(this.indices, 1))
    }

    return geometry
  }

  /**
   * Extract a submesh by marker and convert to BufferGeometry
   */
  extractMarkerAsBufferGeometry(markerName: string): THREE.BufferGeometry {
    const extractedMesh = this.extractByMarker(markerName)
    return extractedMesh.toBufferGeometry()
  }
}
