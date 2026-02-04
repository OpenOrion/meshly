/**
 * Base packable class for encoded array storage.
 *
 * This module provides a base class that handles automatic typed array
 * detection and deserialization from zip files. Classes like Mesh inherit from
 * this base to get automatic array decoding support.
 * 
 * The new format stores data in:
 * - metadata/data.json: Instance data with $ref references to assets
 * - metadata/schema.json: JSON Schema with encoding info
 * - assets/{checksum}.bin: Binary assets (arrays, packables, resources)
 */

import JSZip from "jszip"
import { AssetProvider } from "./common"
import { ExportConstants } from "./constants"
import { JsonSchema, JsonSchemaUtils } from "./json-schema"
import { LazyModel, LazyModelProps } from "./lazy-model"
import {
  isRefObject,
  ReconstructSchema,
  SchemaUtils
} from "./schema-utils"

// Re-export for convenience
export { DynamicModel, DynamicModelBuilder, InstantiateOptions } from "./dynamic-model"
export { ArrayEncoding, JsonSchema, JsonSchemaProperty } from "./json-schema"
export { LazyModel, LazyModelProps } from "./lazy-model"
export { FieldSchema, isArrayRef, isRefObject, PackableDecoder, ReconstructSchema, RefObject } from "./schema-utils"

/**
 * Metadata containing the serializable data and JSON schema.
 * This is the JSON-serializable portion of extracted Packable data.
 */
export interface PackableMetadata {
  /** Serializable dict with primitive fields and checksum refs for arrays */
  data: Record<string, unknown>
  /** JSON Schema with encoding info */
  json_schema?: JsonSchema
}

/**
 * Result of extracting a Packable for serialization.
 */
export interface ExtractedPackable {
  /** JSON-serializable metadata (data + schema) */
  metadata: PackableMetadata
  /** Map of checksum -> encoded bytes for all arrays */
  assets: Record<string, Uint8Array>
}

/**
 * Base class for packable data with automatic array deserialization.
 *
 * Subclasses can define typed array attributes which will be automatically
 * detected and loaded from zip files. Non-array fields are preserved
 * in metadata.
 */
export class Packable<TData = Record<string, unknown>> {
  /**
   * Create a new Packable from data
   */
  constructor(data: TData) {
    Object.assign(this, data)
  }

  // ============================================================
  // Decode from zip (new format)
  // ============================================================

  /**
   * Decode a Packable from zip data (new format).
   * 
   * The new format has:
   * - metadata/data.json: Instance data with $ref references
   * - metadata/schema.json: JSON Schema with encoding info
   * - assets/{checksum}.bin: Binary assets
   *
   * @param zipData - Zip file bytes
   * @returns Reconstructed Packable instance
   */
  static async decode<TData extends Record<string, unknown> = Record<string, unknown>>(
    zipData: ArrayBuffer | Uint8Array
  ): Promise<Packable<TData>> {
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

    // Reconstruct using schema
    const reconstructed = await Packable.reconstruct<TData>(data, assets, schema)
    return new Packable<TData>(reconstructed)
  }

  // ============================================================
  // Reconstruct from data and assets
  // ============================================================

  /**
   * Reconstruct a data object from extracted data and assets.
   * 
   * Resolves $ref references to arrays or nested Packables using the schema
   * to determine encoding types.
   * 
   * @param data - The data dict from extract(), with $ref references
   * @param assets - Asset provider (dict or fetch function)
   * @param schema - JSON Schema for determining field types and encodings
   * @param fieldSchemas - Optional explicit field schemas (override JSON schema)
   * @param isLazy - If true, return a lazy proxy that defers asset loading
   * @returns Reconstructed data object with resolved references, or LazyModel if isLazy
   * 
   * @example
   * ```ts
   * // With JSON schema (recommended)
   * const rebuilt = await Packable.reconstruct(data, assets, jsonSchema)
   * 
   * // With explicit field schemas for nested Packables
   * const fieldSchemas: ReconstructSchema = {
   *   mesh: { type: 'packable', decode: Mesh.decode }
   * }
   * const rebuilt = await Packable.reconstruct(data, assets, jsonSchema, fieldSchemas)
   * 
   * // With lazy loading (defers asset loading until field access)
   * const lazy = await Packable.reconstruct(data, fetcher, jsonSchema, undefined, true)
   * const vertices = await lazy.vertices  // Asset fetched NOW
   * ```
   */
  static async reconstruct<T extends Record<string, unknown> = Record<string, unknown>>(
    data: Record<string, unknown>,
    assets: AssetProvider,
    schema?: JsonSchema,
    fieldSchemas?: ReconstructSchema,
    isLazy?: false
  ): Promise<T>
  static async reconstruct<T extends Record<string, unknown> = Record<string, unknown>>(
    data: Record<string, unknown>,
    assets: AssetProvider,
    schema?: JsonSchema,
    fieldSchemas?: ReconstructSchema,
    isLazy?: true
  ): Promise<LazyModelProps<T> & T>
  static async reconstruct<T extends Record<string, unknown> = Record<string, unknown>>(
    data: Record<string, unknown>,
    assets: AssetProvider,
    schema?: JsonSchema,
    fieldSchemas?: ReconstructSchema,
    isLazy: boolean = false
  ): Promise<T | (LazyModelProps<T> & T)> {
    // Lazy loading - return a LazyModel proxy
    if (isLazy) {
      return LazyModel.create<T>(data, assets, schema, fieldSchemas)
    }

    // Eager loading - resolve all fields now
    const result: Record<string, unknown> = {}

    for (const [key, value] of Object.entries(data)) {
      // Skip metadata fields
      if (key.startsWith("$")) continue

      // Get encoding from schema
      const encoding = schema ? JsonSchemaUtils.getEncoding(schema, key) : "array"
      const fieldSchema = fieldSchemas?.[key]
      const schemaProp = schema ? JsonSchemaUtils.getResolvedProperty(schema, key) : undefined

      result[key] = await SchemaUtils._resolveWithProp(value, schemaProp, schema!, assets, fieldSchema)
    }

    return result as T
  }

  /**
   * Extract all checksums from a data dict.
   */
  static extractChecksums(data: Record<string, unknown>): string[] {
    const checksums = new Set<string>()

    function extract(obj: unknown): void {
      if (isRefObject(obj)) {
        checksums.add(obj.$ref)
      } else if (Array.isArray(obj)) {
        obj.forEach(extract)
      } else if (obj && typeof obj === "object") {
        Object.values(obj).forEach(extract)
      }
    }

    extract(data)
    return Array.from(checksums)
  }
}

// Legacy type alias for backwards compatibility
export type SerializedPackableData = ExtractedPackable
