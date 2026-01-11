/**
 * Utility class for zip file operations used by Packable and Mesh.
 */

import JSZip from "jszip"
import { ArrayMetadata, ArrayUtils, TypedArray } from "../array"

/**
 * Static utility methods for zip file operations.
 */
export class ZipUtils {
  /**
   * Load and decode a single array from a zip file.
   *
   * @param zip - JSZip instance to read from
   * @param name - Array name (e.g., "normals" or "markerIndices.boundary")
   * @returns Decoded typed array
   * @throws Error if array not found in zip
   */
  static async loadArray(
    zip: JSZip,
    name: string
  ): Promise<TypedArray> {
    // Convert dotted name to path
    const arrayPath = name.replace(/\./g, "/")
    const arraysFolder = zip.folder("arrays")

    if (!arraysFolder) {
      throw new Error(`Array '${name}' not found in zip file`)
    }

    const metadataFile = arraysFolder.file(`${arrayPath}/metadata.json`)
    const arrayFile = arraysFolder.file(`${arrayPath}/array.bin`)

    if (!metadataFile || !arrayFile) {
      throw new Error(`Array '${name}' not found in zip file`)
    }

    const metadataText = await metadataFile.async("text")
    const metadata: ArrayMetadata = JSON.parse(metadataText)
    const data = await arrayFile.async("uint8array")

    return ArrayUtils.decodeArray(data, metadata)
  }

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
      // Convert path to dotted name (e.g., "markerIndices/boundary" -> "markerIndices.boundary")
      const name = arrayPath.replace(/\//g, ".")

      const decoded = await ZipUtils.loadArray(zip, name)

      if (name.includes(".")) {
        // Nested array - build nested structure
        const parts = name.split(".")
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
        result[name] = decoded
      }
    }

    return result
  }
}
