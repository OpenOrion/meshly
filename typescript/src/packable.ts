/**
 * Base packable class for encoded array storage.
 *
 * This module provides a base class that handles automatic typed array
 * detection and deserialization from zip files. Classes like Mesh inherit from
 * this base to get automatic array decoding support.
 */

import JSZip from "jszip"
import { ArrayMetadata, ArrayUtils, EncodedArray, TypedArray } from "./array"
import { AssetProvider, CachedAssetLoader, getAsset } from "./data-handler"


/**
 * Base metadata interface for Packable zip files.
 * Uses snake_case to match Python serialization format.
 */
export interface PackableMetadata {
  /** Non-array field values */
  field_data?: Record<string, unknown>
}

/**
 * Custom decoder function type.
 * Takes encoded bytes and metadata, returns decoded value.
 */
export type CustomDecoder<T = unknown, M extends PackableMetadata = PackableMetadata> = (
  data: Uint8Array,
  metadata: M
) => T

/**
 * Custom field configuration for decoding
 */
export interface CustomFieldConfig<T = unknown, M extends PackableMetadata = PackableMetadata> {
  /** File name in zip (without .bin extension) */
  fileName: string
  /** Custom decoder function */
  decode: CustomDecoder<T, M>
  /** Whether the field is optional (won't throw if missing) */
  optional?: boolean
}


/**
 * Base class for packable data with automatic array deserialization.
 *
 * Subclasses can define typed array attributes which will be automatically
 * detected and loaded from zip files. Non-array fields are preserved
 * in metadata.
 * 
 * Custom field decoding is supported via `getCustomFields()` override.
 */
export class Packable<TData> {
  /**
   * Create a new Packable from data
   */
  constructor(data: TData) {
    Object.assign(this, data)
  }

  /**
   * Load and parse metadata.json from a zip file.
   */
  static async loadMetadata<T extends PackableMetadata = PackableMetadata>(zip: JSZip): Promise<T> {
    const metadataFile = zip.file("metadata.json")
    if (!metadataFile) {
      throw new Error("metadata.json not found in zip file")
    }
    const metadataText = await metadataFile.async("text")
    return JSON.parse(metadataText) as T
  }

  // ============================================================
  // Custom field handling
  // ============================================================

  /**
   * Get custom field configurations for this class.
   * Subclasses override this to define custom decoders.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  protected static getCustomFields(): Record<string, CustomFieldConfig<unknown, any>> {
    return {}
  }

  /**
   * Get the set of custom field names
   */
  protected static getCustomFieldNames(): Set<string> {
    return new Set(Object.keys(this.getCustomFields()))
  }

  /**
   * Decode custom fields from the zip
   */
  protected static async decodeCustomFields<M extends PackableMetadata>(
    zip: JSZip,
    metadata: M,
    data: Record<string, unknown>
  ): Promise<void> {
    const customFields = this.getCustomFields()

    for (const [fieldName, config] of Object.entries(customFields)) {
      const file = zip.file(`${config.fileName}.bin`)
      if (file) {
        const encoded = await file.async('uint8array')
        data[fieldName] = config.decode(encoded, metadata)
      } else if (!config.optional) {
        throw new Error(`Required custom field '${fieldName}' (${config.fileName}.bin) not found in zip`)
      }
    }
  }

  // ============================================================
  // Standard array loading
  // ============================================================

  /**
   * Load standard arrays from arrays/ folder
   */
  protected static async loadStandardArrays(
    zip: JSZip,
    data: Record<string, unknown>,
    skipFields: Set<string>
  ): Promise<void> {
    const arraysFolder = zip.folder("arrays")
    if (!arraysFolder) return

    // Find all array directories
    const arrayPaths = new Set<string>()
    arraysFolder.forEach((relativePath) => {
      const parts = relativePath.split("/")
      if (parts.length >= 2) {
        const dirPath = parts.slice(0, -1).join("/")
        if (dirPath) {
          arrayPaths.add(dirPath)
        }
      }
    })

    // Load and decode each array
    for (const arrayPath of arrayPaths) {
      const name = arrayPath.replace(/\//g, ".")

      // Skip custom fields
      const baseFieldName = name.split(".")[0]
      if (skipFields.has(baseFieldName)) continue

      const decoded = await ArrayUtils.loadArray(zip, name)

      if (name.includes(".")) {
        // Nested array - build nested structure
        const parts = name.split(".")
        let current = data
        for (let i = 0; i < parts.length - 1; i++) {
          const part = parts[i]
          if (!current[part]) {
            current[part] = {}
          }
          current = current[part] as Record<string, unknown>
        }
        current[parts[parts.length - 1]] = decoded
      } else {
        // Flat array
        data[name] = decoded
      }
    }
  }

  /**
   * Decode a Packable from zip data.
   * 
   * @param zipData - Zip file bytes
   * 
   * Subclasses can override this to handle custom field decoding.
   */
  static async decode<TData>(
    zipData: ArrayBuffer | Uint8Array
  ): Promise<Packable<TData>> {
    const zip = await JSZip.loadAsync(zipData)
    const metadata = await Packable.loadMetadata(zip)
    const customFieldNames = this.getCustomFieldNames()

    const data: Record<string, unknown> = {}

    // Decode custom fields first
    await this.decodeCustomFields(zip, metadata, data)

    // Load standard arrays
    await this.loadStandardArrays(zip, data, customFieldNames)

    // Merge non-array fields from metadata
    if (metadata.field_data) {
      Packable._mergeFieldData(data, metadata.field_data)
    }

    return new Packable<TData>(data as TData)
  }

  // ============================================================
  // Private static helper methods
  // ============================================================

  /**
   * Merge non-array field values into data object (in place).
   */
  protected static _mergeFieldData(
    data: Record<string, unknown>,
    fieldData: Record<string, unknown>
  ): void {
    for (const [key, value] of Object.entries(fieldData)) {
      const existing = data[key]

      if (
        existing &&
        typeof existing === "object" &&
        typeof value === "object" &&
        !ArrayBuffer.isView(existing) &&
        !ArrayBuffer.isView(value)
      ) {
        // Both are objects - merge recursively
        Packable._mergeFieldData(
          existing as Record<string, unknown>,
          value as Record<string, unknown>
        )
      } else {
        data[key] = value
      }
    }
  }

  /**
   * Load a single array from a zip file without loading the entire object.
   */
  static async loadArray(
    zipData: ArrayBuffer | Uint8Array,
    name: string
  ): Promise<TypedArray> {
    const zip = await JSZip.loadAsync(zipData)
    return ArrayUtils.loadArray(zip, name)
  }

  // ============================================================
  // Extract / Reconstruct for content-addressable storage
  // ============================================================

  /**
   * Decode a packed array asset (metadata + data bytes) to a TypedArray.
   * 
   * Format: [4 bytes metadata length][metadata json][array data]
   */
  static _decodePackedArray(packed: Uint8Array | ArrayBuffer): TypedArray {
    const bytes = packed instanceof Uint8Array ? packed : new Uint8Array(packed)

    // Read metadata length (4 bytes little-endian)
    const metadataLen = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)

    // Parse metadata JSON
    const metadataJson = new TextDecoder().decode(bytes.slice(4, 4 + metadataLen))
    const metadata: ArrayMetadata = JSON.parse(metadataJson)

    // Get array data
    const arrayData = bytes.slice(4 + metadataLen)

    const encoded: EncodedArray = { data: arrayData, metadata }
    return ArrayUtils.decodeArray(encoded)
  }

  /**
   * Reconstruct a data object from extracted data and assets.
   * 
   * Since TypeScript doesn't have runtime type information like Python's Pydantic,
   * this provides a simpler approach:
   * - Resolves $ref references to arrays or nested Packables
   * - Uses the optional `schema` to determine which refs are Packables vs arrays
   * 
   * @param data - The data dict from extract(), with $ref references
   * @param assets - Asset provider (dict, function, or CachedAssetLoader)
   * @param schema - Optional schema defining which fields are Packables
   * @returns Reconstructed data object with resolved references
   * 
   * @example
   * ```ts
   * // Simple case - all $refs are arrays
   * const rebuilt = await Packable.reconstruct(data, assets)
   * 
   * // With nested Packables - define schema
   * const schema: ReconstructSchema = {
   *   mesh: { type: 'packable', decode: Mesh.decode },
   *   snapshots: { 
   *     type: 'array', 
   *     element: { 
   *       mesh: { type: 'packable', decode: Mesh.decode } 
   *     } 
   *   }
   * }
   * const rebuilt = await Packable.reconstruct(data, assets, schema)
   * ```
   */
  static async reconstruct<T = Record<string, unknown>>(
    data: Record<string, unknown>,
    assets: AssetProvider | CachedAssetLoader,
    schema?: ReconstructSchema
  ): Promise<T> {
    const result: Record<string, unknown> = {}

    for (const [key, value] of Object.entries(data)) {
      const fieldSchema = schema?.[key]
      result[key] = await Packable._resolveValue(value, assets, fieldSchema)
    }

    return result as T
  }

  /**
   * Resolve a single value, handling $ref, nested objects, and arrays.
   */
  private static async _resolveValue(
    value: unknown,
    assets: AssetProvider | CachedAssetLoader,
    schema?: FieldSchema
  ): Promise<unknown> {
    if (value === null || value === undefined) {
      return value
    }

    // Handle $ref references
    if (isRefObject(value)) {
      const checksum = value.$ref
      const assetBytes = await getAsset(assets, checksum)
      const bytes = assetBytes instanceof Uint8Array ? assetBytes : new Uint8Array(assetBytes)

      // Use schema to determine type, default to array
      if (schema?.type === 'packable' && schema.decode) {
        return schema.decode(bytes)
      }

      // Default: decode as array
      return Packable._decodePackedArray(bytes)
    }

    // Handle arrays (JS arrays, not TypedArrays)
    if (Array.isArray(value)) {
      const elementSchema = schema?.type === 'array' ? schema.element : undefined
      return Promise.all(
        value.map(v => Packable._resolveValue(v, assets, elementSchema))
      )
    }

    // Handle nested objects
    if (typeof value === 'object' && !ArrayBuffer.isView(value)) {
      const obj = value as Record<string, unknown>
      const result: Record<string, unknown> = {}

      for (const [k, v] of Object.entries(obj)) {
        // Skip Python model metadata
        if (k === '__model_class__' || k === '__model_module__') continue

        // Get nested schema if this is a dict schema
        const nestedSchema = schema?.type === 'dict' ? schema.value :
          schema?.type === 'object' ? schema.fields?.[k] : undefined
        result[k] = await Packable._resolveValue(v, assets, nestedSchema)
      }

      return result
    }

    // Primitive - return as-is
    return value
  }
}


// ============================================================
// Reconstruct Schema Types
// ============================================================

/**
 * Reference object with $ref checksum
 */
interface RefObject {
  $ref: string
}

function isRefObject(value: unknown): value is RefObject {
  return typeof value === 'object' && value !== null && '$ref' in value
}

/**
 * Decoder function for Packable types
 */
export type PackableDecoder<T> = (data: Uint8Array | ArrayBuffer) => Promise<T> | T

/**
 * Schema for a single field in reconstruct
 */
export type FieldSchema =
  | { type: 'array'; element?: FieldSchema }  // TypedArray or Array of items
  | { type: 'packable'; decode: PackableDecoder<unknown> }  // Nested Packable
  | { type: 'dict'; value?: FieldSchema }  // Dict with uniform value type
  | { type: 'object'; fields?: ReconstructSchema }  // Object with known field types

/**
 * Schema mapping field names to their types for reconstruction.
 * 
 * Without runtime type information, TypeScript needs hints to know
 * which $ref values are Packables vs arrays.
 */
export type ReconstructSchema = Record<string, FieldSchema>

/**
 * Result of extracting a Packable for serialization.
 * 
 * Contains the serializable data dict with checksum references,
 * plus the encoded assets (arrays as bytes).
 */
export interface SerializedPackableData {
  /** Serializable dict with primitive fields and checksum refs for arrays */
  data: Record<string, unknown>
  /** Map of checksum -> encoded bytes for all arrays */
  assets: Record<string, Uint8Array>
}
