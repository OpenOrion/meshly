import JSZip from 'jszip'
import * as THREE from 'three'
import { ArrayUtils, TypedArray } from './array'
import { ExportConstants } from './constants'
import { JsonSchema, JsonSchemaUtils } from './json-schema'
import { isArrayRef, isRefObject, Packable } from './packable'


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
  markers?: Record<string, Uint32Array>

  /**
   * Sizes of each marker element
   */
  markerSizes?: Record<string, Uint32Array>

  /**
   * VTK cell types for each marker element
   */
  markerCellTypes?: Record<string, Uint32Array>
}

/**
 * Mesh class for loading and working with meshoptimizer-compressed meshes.
 * 
 * Extends Packable to provide mesh-specific decoding for vertices and indices
 * using meshoptimizer's vertex and index buffer decompression.
 * 
 * Uses the new format: metadata/data.json + metadata/schema.json + assets/{checksum}.bin
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
  declare markers?: Record<string, Uint32Array>
  declare markerSizes?: Record<string, Uint32Array>
  declare markerCellTypes?: Record<string, Uint32Array>

  // ============================================================
  // Decode from zip
  // ============================================================

  /**
   * Decode a Mesh from zip data.
   * 
   * Expects format: metadata/data.json + metadata/schema.json + assets/{checksum}.bin
   */
  static override async decode(zipData: ArrayBuffer | Uint8Array): Promise<Mesh> {
    const zip = await JSZip.loadAsync(zipData)
    // Read data.json
    const dataFile = zip.file(ExportConstants.DATA_FILE)
    if (!dataFile) {
      throw new Error(`${ExportConstants.DATA_FILE} not found in zip file`)
    }
    const dataText = await dataFile.async("text")
    const data: Record<string, unknown> = JSON.parse(dataText)

    // Read schema.json (optional but recommended)
    let schema: JsonSchema | undefined
    const schemaFile = zip.file(ExportConstants.SCHEMA_FILE)
    if (schemaFile) {
      const schemaText = await schemaFile.async("text")
      schema = JSON.parse(schemaText)
    }

    // Build assets dict from files in assets/ directory
    const assets: Record<string, Uint8Array> = {}
    for (const filePath of Object.keys(zip.files)) {
      if (filePath.startsWith(ExportConstants.ASSETS_DIR + "/") &&
        filePath.endsWith(ExportConstants.ASSET_EXT)) {
        const checksum = ExportConstants.checksumFromPath(filePath)
        const file = zip.file(filePath)
        if (file) {
          assets[checksum] = await file.async("uint8array")
        }
      }
    }

    // Reconstruct using schema-aware method
    const result: Record<string, unknown> = {}

    for (const [key, value] of Object.entries(data)) {
      if (key.startsWith("$")) continue

      // Get encoding from schema
      const encoding = schema ? JsonSchemaUtils.getEncoding(schema, key) : "array"
      const schemaProp = schema ? JsonSchemaUtils.getResolvedProperty(schema, key) : undefined

      result[key] = await Mesh._resolveField(value, assets, encoding, schema, schemaProp)
    }

    // Map Python field names to TypeScript names for known mesh fields
    const meshData: MeshData = {
      vertices: result.vertices as Float32Array,
      indices: result.indices as Uint32Array | undefined,
      indexSizes: (result.index_sizes || result.indexSizes) as Uint32Array | undefined,
      cellTypes: (result.cell_types || result.cellTypes) as Uint32Array | undefined,
      dim: result.dim as number | undefined,
      markers: Mesh._convertMarkers(result.markers as Record<string, unknown> | undefined),
      markerSizes: Mesh._convertMarkers(result.marker_sizes as Record<string, unknown> | undefined),
      markerCellTypes: Mesh._convertMarkers(result.marker_cell_types as Record<string, unknown> | undefined),
    }

    // Pass through additional fields from subclasses (e.g., TexturedMesh)
    const knownMeshFields = new Set([
      'vertices', 'indices', 'index_sizes', 'indexSizes', 'cell_types', 'cellTypes',
      'dim', 'markers', 'marker_sizes', 'markerSizes', 'marker_cell_types', 'markerCellTypes'
    ])
    for (const [key, value] of Object.entries(result)) {
      if (!knownMeshFields.has(key) && !key.startsWith('$')) {
        (meshData as unknown as Record<string, unknown>)[key] = value
      }
    }

    return new Mesh(meshData)
  }

  /**
   * Resolve a field value, handling $ref references
   */
  private static async _resolveField(
    value: unknown,
    assets: Record<string, Uint8Array>,
    encoding: string,
    schema?: JsonSchema,
    schemaProp?: unknown
  ): Promise<unknown> {
    if (value === null || value === undefined) {
      return value
    }

    // Handle $ref references
    if (isRefObject(value)) {
      const bytes = assets[value.$ref]
      if (!bytes) {
        throw new Error(`Asset not found: ${value.$ref}`)
      }

      // Check if it's an array ref (has shape and dtype)
      if (isArrayRef(value)) {
        return ArrayUtils.reconstruct({ data: bytes, info: value, encoding: encoding as "array" | "vertex_buffer" | "index_sequence" })
      }

      // Otherwise it's a nested Packable (stored as a zip file)
      // Check if bytes start with PK (zip magic)
      if (bytes.length >= 2 && bytes[0] === 0x50 && bytes[1] === 0x4B) {
        // Recursively decode the nested Packable
        return Packable.decode(bytes)
      }

      // Unknown ref type - return raw bytes
      return bytes
    }

    // Handle nested objects (like markers dict)
    if (typeof value === "object" && !Array.isArray(value)) {
      const result: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        if (k.startsWith("$")) continue
        result[k] = await Mesh._resolveField(v, assets, "array", schema, undefined)
      }
      return result
    }

    // Handle arrays
    if (Array.isArray(value)) {
      return Promise.all(value.map(v => Mesh._resolveField(v, assets, "array", schema, undefined)))
    }

    return value
  }

  /**
   * Convert markers from Record<string, unknown> to Record<string, Uint32Array>
   */
  private static _convertMarkers(markers: Record<string, unknown> | undefined): Record<string, Uint32Array> | undefined {
    if (!markers) return undefined
    const result: Record<string, Uint32Array> = {}
    for (const [key, value] of Object.entries(markers)) {
      if (value instanceof Uint32Array) {
        result[key] = value
      } else if (ArrayBuffer.isView(value)) {
        result[key] = new Uint32Array((value as TypedArray).buffer)
      }
    }
    return Object.keys(result).length > 0 ? result : undefined
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
    if (!this.markers || !(markerName in this.markers)) {
      const availableMarkers = this.markers ? Object.keys(this.markers).join(', ') : 'none'
      throw new Error(`Marker '${markerName}' not found. Available markers: ${availableMarkers}`)
    }

    const markerIndices = this.markers[markerName]
    const markerSizes = this.markerSizes?.[markerName]
    const markerCellTypes = this.markerCellTypes?.[markerName]

    if (!markerSizes || !markerCellTypes) {
      throw new Error(`Marker '${markerName}' is missing sizes or cell type information`)
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

    return new Mesh({
      vertices: extractedVertices,
      indices: remappedIndices,
      indexSizes: markerSizes,
      cellTypes: markerCellTypes,
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
