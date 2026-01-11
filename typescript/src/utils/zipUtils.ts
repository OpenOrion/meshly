/**
 * Utility class for zip file operations used by Packable and Mesh.
 */

import JSZip from "jszip"
import { ArrayMetadata, ArrayUtils } from "../array"

/**
 * Static utility methods for zip file operations.
 */
export class ZipUtils {
  /**
   * Load and decode all arrays from a zip file's arrays/ folder.
   * 
   * Handles both flat arrays (e.g., "vertices") and nested arrays 
   * (e.g., "markerIndices.boundary" becomes { markerIndices: { boundary: ... } })
   * 
   * @param zip - JSZip instance to read from
   * @returns Decoded arrays organized into proper structure
   */
  static async loadArrays(zip: JSZip): Promise<Record<string, unknown>> {
    const result: Record<string, unknown> = {}
    const arraysFolder = zip.folder("arrays")

    if (!arraysFolder) {
      return result
    }

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
      const metadataFile = arraysFolder.file(`${arrayPath}/metadata.json`)
      const arrayFile = arraysFolder.file(`${arrayPath}/array.bin`)

      if (!metadataFile || !arrayFile) continue

      const metadataText = await metadataFile.async("text")
      const metadata: ArrayMetadata = JSON.parse(metadataText)
      const data = await arrayFile.async("uint8array")
      const decoded = ArrayUtils.decodeArray(data, metadata)

      // Convert path to key (e.g., "markerIndices/boundary" -> "markerIndices.boundary")
      const key = arrayPath.replace(/\//g, ".")

      if (key.includes(".")) {
        // Nested array - build nested structure
        const parts = key.split(".")
        let current = result

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
        result[key] = decoded
      }
    }

    return result
  }

  /**
   * Merge non-array field values into data object.
   * 
   * Values like `dim: 2` from metadata.fieldData get merged in.
   * Existing dict structures are merged recursively.
   * 
   * @param data - Target object to merge into (modified in place)
   * @param fieldData - Field values from metadata.json
   */
  static mergeFieldData(
    data: Record<string, unknown>,
    fieldData?: Record<string, unknown>
  ): void {
    if (!fieldData) return

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
        ZipUtils.mergeFieldData(
          existing as Record<string, unknown>,
          value as Record<string, unknown>
        )
      } else {
        data[key] = value
      }
    }
  }
}
