/**
 * Base packable class for encoded array storage.
 *
 * This module provides a base class that handles automatic typed array
 * detection and deserialization from zip files. Classes like Mesh inherit from
 * this base to get automatic array decoding support.
 */

import JSZip from "jszip"
import { TypedArray } from "./array"
import { PackableUtils } from "./utils/packableUtils"
import { ZipUtils } from "./utils/zipUtils"


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
      PackableUtils.mergeFieldData(data as Record<string, unknown>, metadata.field_data)
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
  ): Promise<TypedArray> {
    const zip = await JSZip.loadAsync(zipData)
    return ZipUtils.loadArray(zip, name)
  }
}
