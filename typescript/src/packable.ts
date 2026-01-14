/**
 * Base packable class for encoded array storage.
 *
 * This module provides a base class that handles automatic typed array
 * detection and deserialization from zip files. Classes like Mesh inherit from
 * this base to get automatic array decoding support.
 */

import JSZip from "jszip"
import { ArrayUtils, TypedArray } from "./array"


/**
 * Base metadata interface for Packable zip files.
 * Uses snake_case to match Python serialization format.
 */
export interface PackableMetadata {
  /** Name of the class that created this data */
  class_name: string
  /** Module where the class is defined */
  module_name: string
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
  protected static getCustomFields(): Record<string, CustomFieldConfig> {
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
      // Skip Python BaseModel reconstruction metadata
      if (key === "__model_class__" || key === "__model_module__") {
        continue
      }

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
      } else if (typeof value === "object" && value !== null && !ArrayBuffer.isView(value)) {
        // Value is an object that might contain Python metadata - clean it
        data[key] = Packable._stripModelMetadata(value as Record<string, unknown>)
      } else {
        data[key] = value
      }
    }
  }

  /**
   * Recursively strip Python BaseModel metadata keys from an object.
   */
  private static _stripModelMetadata(obj: Record<string, unknown>): Record<string, unknown> {
    const result: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(obj)) {
      if (key === "__model_class__" || key === "__model_module__") {
        continue
      }
      if (typeof value === "object" && value !== null && !ArrayBuffer.isView(value)) {
        result[key] = Packable._stripModelMetadata(value as Record<string, unknown>)
      } else {
        result[key] = value
      }
    }
    return result
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
}
