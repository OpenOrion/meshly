/**
 * Base packable class for encoded array storage.
 *
 * This module provides a base class that handles automatic typed array
 * detection and deserialization from zip files. Classes like Mesh inherit from
 * this base to get automatic array decoding support.
 */

import JSZip from "jszip"
import { ZipUtils } from "./utils/zipUtils"

/**
 * TypedArray union for decoded array data
 */
export type TypedArray = Float32Array | Float64Array | Int8Array | Int16Array | Int32Array | Uint8Array | Uint16Array | Uint32Array

/**
 * Recursive type for decoded array data from zip files.
 * Values are typed arrays or nested objects containing arrays.
 */
export type ArrayData = Record<string, TypedArray | ArrayData>

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
 * Base class for packable data with automatic array deserialization.
 *
 * Subclasses can define typed array attributes which will be automatically
 * detected and loaded from zip files. Non-array fields are preserved
 * in metadata.
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

  /**
   * Merge non-array field values into data object (in place).
   *
   * Values like `dim: 2` from metadata.fieldData get merged in.
   * Existing object structures are merged recursively.
   *
   * @param data - Target object to merge into (modified in place)
   * @param fieldData - Field values from metadata
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
   * Load a Packable from a zip file
   */
  static async loadFromZip<TData>(
    zipData: ArrayBuffer | Uint8Array
  ): Promise<Packable<TData>> {
    const zip = await JSZip.loadAsync(zipData)

    const metadata = await Packable.loadMetadata(zip)

    // Load and decode all arrays (handles both flat and nested)
    const data = await ZipUtils.loadArrays(zip)

    // Merge non-array fields from metadata
    if (metadata.field_data) {
      Packable._mergeFieldData(data as Record<string, unknown>, metadata.field_data)
    }

    return new Packable<TData>(data as TData)
  }

  /**
   * Load a single array from a zip file without loading the entire object.
   *
   * Useful for large files where you only need one array.
   *
   * @param zipData - Zip file as ArrayBuffer or Uint8Array
   * @param name - Array name (e.g., "normals" or "markers.inlet")
   * @returns Decoded typed array
   * @throws Error if array not found in zip
   *
   * @example
   * const normals = await Mesh.loadArray(zipData, "normals")
   */
  static async loadArray(
    zipData: ArrayBuffer | Uint8Array,
    name: string
  ): Promise<Float32Array | Float64Array | Int32Array | Uint32Array | Uint8Array> {
    const zip = await JSZip.loadAsync(zipData)
    return ZipUtils.loadArray(zip, name)
  }
}
