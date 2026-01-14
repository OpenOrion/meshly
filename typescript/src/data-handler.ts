/**
 * Data handler interface for reading and writing files from various sources.
 * Provides a unified interface for file I/O operations.
 */

/**
 * Data handler interface for loading and saving data from various sources.
 * Provides a minimal interface for reading/writing zip files.
 */
export interface DataHandler {
  /**
   * Read binary content from a file.
   * @param path - File path (e.g., "hash.zip")
   * @returns File content as ArrayBuffer or Uint8Array, or undefined if not found
   */
  readBinary(path: string): Promise<ArrayBuffer | Uint8Array | undefined>

  /**
   * Check if a file exists.
   * @param path - File path
   * @returns true if file exists
   */
  exists?(path: string): Promise<boolean>
}

/**
 * Create a DataHandler from a simple hash loader function.
 * Provides backward compatibility for function-based loaders.
 */
export function createDataHandler(
  loader: (hash: string) => Promise<ArrayBuffer | Uint8Array | undefined>
): DataHandler {
  return {
    readBinary: async (path: string) => {
      // Extract hash from path (e.g., "abc123.zip" -> "abc123")
      const hash = path.replace(/\.zip$/, '')
      return loader(hash)
    }
  }
}
