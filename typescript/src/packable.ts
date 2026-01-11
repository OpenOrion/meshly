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

    // Merge non-array field values from metadata
    ZipUtils.mergeFieldData(data, metadata.field_data)

    return new Packable<TData>(data as TData)
  }
}
