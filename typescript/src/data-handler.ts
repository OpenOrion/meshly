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
   * Write binary content to a file.
   * @param path - File path
   * @param content - Content to write
   */
  writeBinary?(path: string, content: Uint8Array | ArrayBuffer): Promise<void>

  /**
   * Check if a file exists.
   * @param path - File path
   * @returns true if file exists
   */
  exists?(path: string): Promise<boolean>
}

/**
 * Asset fetch function type - takes a checksum and returns asset bytes
 */
export type AssetFetcher = (checksum: string) => Promise<Uint8Array | ArrayBuffer>

/**
 * Asset provider: either a dict of assets or a fetcher function
 */
export type AssetProvider = Record<string, Uint8Array | ArrayBuffer> | AssetFetcher

/**
 * Asset loader with optional disk cache for persistence.
 * 
 * Wraps a fetch function with a DataHandler for caching.
 * Fetched assets are stored as 'assets/{checksum}.bin' and read
 * from cache on subsequent access.
 * 
 * @example
 * ```ts
 * const loader = new CachedAssetLoader(
 *   async (checksum) => await fetch(`/api/assets/${checksum}`).then(r => r.arrayBuffer()),
 *   myDataHandler
 * )
 * const model = await Packable.reconstruct(data, loader)
 * ```
 */
export class CachedAssetLoader {
  constructor(
    /** Function that fetches asset bytes by checksum */
    public readonly fetch: AssetFetcher,
    /** DataHandler for caching fetched assets */
    public readonly cache: DataHandler
  ) { }

  /**
   * Get asset bytes, checking cache first then fetching if needed.
   */
  async getAsset(checksum: string): Promise<Uint8Array | ArrayBuffer> {
    const cachePath = `assets/${checksum}.bin`

    // Try cache first
    if (this.cache.exists) {
      const exists = await this.cache.exists(cachePath)
      if (exists) {
        const cached = await this.cache.readBinary(cachePath)
        if (cached) return cached
      }
    } else {
      // No exists method, try read directly
      const cached = await this.cache.readBinary(cachePath)
      if (cached) return cached
    }

    // Fetch from source
    const fetched = await this.fetch(checksum)

    // Cache for next time
    if (this.cache.writeBinary) {
      const data = fetched instanceof Uint8Array ? fetched : new Uint8Array(fetched)
      await this.cache.writeBinary(cachePath, data)
    }

    return fetched
  }
}


/**
 * Helper to get asset bytes from an AssetProvider
 */
export async function getAsset(
  assets: AssetProvider | CachedAssetLoader,
  checksum: string
): Promise<Uint8Array | ArrayBuffer> {
  if (assets instanceof CachedAssetLoader) {
    return assets.getAsset(checksum)
  }

  if (typeof assets === 'function') {
    return assets(checksum)
  }

  // Dict lookup
  const asset = assets[checksum]
  if (!asset) {
    throw new Error(`Missing asset with checksum '${checksum}'`)
  }
  return asset
}
