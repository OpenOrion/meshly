/**
 * Asset caching utilities for meshly.
 * 
 * Provides an LRU cache backed by IndexedDB for efficient asset caching
 * in browser environments.
 */

import { AssetFetcher, AssetProvider } from './common'

/**
 * Configuration for the AssetCache
 */
export interface AssetCacheConfig {
  /**
   * Database name for IndexedDB storage
   * @default 'meshly-asset-cache'
   */
  dbName?: string

  /**
   * Maximum number of assets to cache
   * @default 500
   */
  maxItems?: number

  /**
   * Maximum total size in bytes for the cache
   * @default 500 * 1024 * 1024 (500MB)
   */
  maxSize?: number
}

interface CachedAsset {
  checksum: string
  data: Uint8Array
  size: number
  lastAccess: number
}

interface CacheMetadata {
  totalSize: number
  itemCount: number
}

const DEFAULT_CONFIG: Required<AssetCacheConfig> = {
  dbName: 'meshly-asset-cache',
  maxItems: 500,
  maxSize: 500 * 1024 * 1024, // 500MB
}

/**
 * IndexedDB-backed LRU cache for binary assets.
 * 
 * Provides efficient caching of mesh assets with automatic eviction
 * based on LRU (Least Recently Used) policy when cache limits are reached.
 * 
 * @example
 * ```typescript
 * const cache = new AssetCache({ maxItems: 100 })
 * await cache.initialize()
 * 
 * // Use as an AssetProvider
 * const mesh = await Mesh.decode(zipData) // or use cache.createProvider(fetcher)
 * 
 * // Create a cached provider that wraps a fetcher
 * const cachedProvider = cache.createProvider(async (checksum) => {
 *   const response = await fetch(`/assets/${checksum}.bin`)
 *   return new Uint8Array(await response.arrayBuffer())
 * })
 * 
 * const data = await Packable.reconstruct(data, cachedProvider, schema)
 * ```
 */
export class AssetCache {
  private config: Required<AssetCacheConfig>
  private db: IDBDatabase | null = null
  private memoryCache = new Map<string, Uint8Array>()
  private accessOrder: string[] = []
  private initialized = false
  private initPromise: Promise<void> | null = null

  constructor(config: AssetCacheConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config }
  }

  /**
   * Initialize the cache, opening the IndexedDB database.
   * Safe to call multiple times - subsequent calls will return immediately.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return
    if (this.initPromise) return this.initPromise

    this.initPromise = this._initialize()
    return this.initPromise
  }

  private async _initialize(): Promise<void> {
    if (typeof indexedDB === 'undefined') {
      // Running in a non-browser environment - use memory-only cache
      this.initialized = true
      return
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.config.dbName, 1)

      request.onerror = () => {
        console.warn('IndexedDB not available, using memory-only cache')
        this.initialized = true
        resolve()
      }

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result

        // Create assets object store
        if (!db.objectStoreNames.contains('assets')) {
          const store = db.createObjectStore('assets', { keyPath: 'checksum' })
          store.createIndex('lastAccess', 'lastAccess', { unique: false })
        }

        // Create metadata object store
        if (!db.objectStoreNames.contains('metadata')) {
          db.createObjectStore('metadata', { keyPath: 'key' })
        }
      }

      request.onsuccess = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result
        this.initialized = true
        resolve()
      }
    })
  }

  /**
   * Get an asset from the cache.
   * Returns undefined if not found.
   */
  async get(checksum: string): Promise<Uint8Array | undefined> {
    await this.initialize()

    // Check memory cache first
    if (this.memoryCache.has(checksum)) {
      // Move to end of access order (most recently used)
      const idx = this.accessOrder.indexOf(checksum)
      if (idx > -1) {
        this.accessOrder.splice(idx, 1)
        this.accessOrder.push(checksum)
      }
      return this.memoryCache.get(checksum)
    }

    // Try IndexedDB
    if (!this.db) return undefined

    return new Promise((resolve) => {
      const transaction = this.db!.transaction(['assets'], 'readwrite')
      const store = transaction.objectStore('assets')
      const request = store.get(checksum)

      request.onsuccess = () => {
        const result = request.result as CachedAsset | undefined
        if (result) {
          // Update last access time
          result.lastAccess = Date.now()
          store.put(result)

          // Add to memory cache
          this._addToMemoryCache(checksum, result.data)
          resolve(result.data)
        } else {
          resolve(undefined)
        }
      }

      request.onerror = () => resolve(undefined)
    })
  }

  /**
   * Store an asset in the cache.
   */
  async put(checksum: string, data: Uint8Array): Promise<void> {
    await this.initialize()

    // Add to memory cache
    this._addToMemoryCache(checksum, data)

    // Store in IndexedDB
    if (!this.db) return

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['assets', 'metadata'], 'readwrite')
      const assetStore = transaction.objectStore('assets')

      const asset: CachedAsset = {
        checksum,
        data,
        size: data.byteLength,
        lastAccess: Date.now(),
      }

      const request = assetStore.put(asset)
      request.onsuccess = () => {
        // Trigger eviction check asynchronously
        this._evictIfNeeded().catch(console.error)
        resolve()
      }
      request.onerror = () => reject(request.error)
    })
  }

  /**
   * Store multiple assets in the cache at once.
   */
  async bulkPut(assets: Record<string, Uint8Array>): Promise<void> {
    await this.initialize()

    const entries = Object.entries(assets)
    if (entries.length === 0) return

    // Add all to memory cache
    for (const [checksum, data] of entries) {
      this._addToMemoryCache(checksum, data)
    }

    // Store in IndexedDB
    if (!this.db) return

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['assets'], 'readwrite')
      const store = transaction.objectStore('assets')
      const now = Date.now()

      for (const [checksum, data] of entries) {
        store.put({
          checksum,
          data,
          size: data.byteLength,
          lastAccess: now,
        })
      }

      transaction.oncomplete = () => {
        this._evictIfNeeded().catch(console.error)
        resolve()
      }
      transaction.onerror = () => reject(transaction.error)
    })
  }

  /**
   * Check if an asset exists in the cache.
   */
  async has(checksum: string): Promise<boolean> {
    await this.initialize()

    if (this.memoryCache.has(checksum)) return true

    if (!this.db) return false

    return new Promise((resolve) => {
      const transaction = this.db!.transaction(['assets'], 'readonly')
      const store = transaction.objectStore('assets')
      const request = store.count(IDBKeyRange.only(checksum))
      request.onsuccess = () => resolve(request.result > 0)
      request.onerror = () => resolve(false)
    })
  }

  /**
   * Remove an asset from the cache.
   */
  async delete(checksum: string): Promise<void> {
    await this.initialize()

    // Remove from memory cache
    this.memoryCache.delete(checksum)
    const idx = this.accessOrder.indexOf(checksum)
    if (idx > -1) this.accessOrder.splice(idx, 1)

    // Remove from IndexedDB
    if (!this.db) return

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['assets'], 'readwrite')
      const store = transaction.objectStore('assets')
      const request = store.delete(checksum)
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }

  /**
   * Clear all cached assets.
   */
  async clear(): Promise<void> {
    await this.initialize()

    this.memoryCache.clear()
    this.accessOrder = []

    if (!this.db) return

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['assets', 'metadata'], 'readwrite')
      transaction.objectStore('assets').clear()
      transaction.objectStore('metadata').clear()
      transaction.oncomplete = () => resolve()
      transaction.onerror = () => reject(transaction.error)
    })
  }

  /**
   * Create an AssetProvider that wraps a fetcher with caching.
   * 
   * When an asset is requested:
   * 1. Check cache first
   * 2. If not found, call fetcher
   * 3. Store result in cache
   * 4. Return result
   * 
   * @param fetcher - Function to fetch assets when not in cache
   * @returns AssetProvider suitable for Packable.reconstruct
   * 
   * @example
   * ```typescript
   * const cache = new AssetCache()
   * await cache.initialize()
   * 
   * const provider = cache.createProvider(async (checksum) => {
   *   const response = await fetch(`/api/assets/${checksum}`)
   *   return new Uint8Array(await response.arrayBuffer())
   * })
   * 
   * const data = await Packable.reconstruct(data, provider, schema)
   * ```
   */
  createProvider(fetcher: AssetFetcher): AssetFetcher {
    return async (checksum: string): Promise<Uint8Array | ArrayBuffer> => {
      // Try cache first
      const cached = await this.get(checksum)
      if (cached) return cached

      // Fetch and cache
      const data = await fetcher(checksum)
      const uint8 = data instanceof ArrayBuffer ? new Uint8Array(data) : data
      await this.put(checksum, uint8 instanceof Uint8Array ? uint8 : new Uint8Array(uint8))
      return data
    }
  }

  /**
   * Get cache statistics.
   */
  async getStats(): Promise<{ memoryItems: number; dbItems: number; dbSize: number }> {
    await this.initialize()

    const stats = {
      memoryItems: this.memoryCache.size,
      dbItems: 0,
      dbSize: 0,
    }

    if (!this.db) return stats

    return new Promise((resolve) => {
      const transaction = this.db!.transaction(['assets'], 'readonly')
      const store = transaction.objectStore('assets')
      const request = store.openCursor()
      
      request.onsuccess = () => {
        const cursor = request.result
        if (cursor) {
          const asset = cursor.value as CachedAsset
          stats.dbItems++
          stats.dbSize += asset.size
          cursor.continue()
        } else {
          resolve(stats)
        }
      }

      request.onerror = () => resolve(stats)
    })
  }

  /**
   * Close the database connection.
   */
  close(): void {
    if (this.db) {
      this.db.close()
      this.db = null
    }
    this.initialized = false
    this.initPromise = null
  }

  // ============================================================
  // Private methods
  // ============================================================

  private _addToMemoryCache(checksum: string, data: Uint8Array): void {
    // Remove from current position if exists
    const idx = this.accessOrder.indexOf(checksum)
    if (idx > -1) this.accessOrder.splice(idx, 1)

    // Add to memory cache and access order
    this.memoryCache.set(checksum, data)
    this.accessOrder.push(checksum)

    // Evict from memory cache if too many items (keep memory cache small)
    const maxMemoryItems = Math.min(50, this.config.maxItems)
    while (this.memoryCache.size > maxMemoryItems && this.accessOrder.length > 0) {
      const oldest = this.accessOrder.shift()
      if (oldest) this.memoryCache.delete(oldest)
    }
  }

  private async _evictIfNeeded(): Promise<void> {
    if (!this.db) return

    // Get current stats
    const stats = await this.getStats()

    // Check if eviction is needed
    if (stats.dbItems <= this.config.maxItems && stats.dbSize <= this.config.maxSize) {
      return
    }

    // Calculate how many items to remove
    const itemsToRemove = Math.max(
      stats.dbItems - Math.floor(this.config.maxItems * 0.9),
      Math.ceil(stats.dbItems * 0.1) // Remove at least 10%
    )

    if (itemsToRemove <= 0) return

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['assets'], 'readwrite')
      const store = transaction.objectStore('assets')
      const index = store.index('lastAccess')

      // Get oldest items by lastAccess
      const request = index.openCursor()
      let removed = 0

      request.onsuccess = () => {
        const cursor = request.result
        if (cursor && removed < itemsToRemove) {
          const checksum = (cursor.value as CachedAsset).checksum
          store.delete(checksum)
          this.memoryCache.delete(checksum)
          removed++
          cursor.continue()
        } else {
          resolve()
        }
      }

      request.onerror = () => reject(request.error)
    })
  }
}

// Default shared cache instance
let defaultCache: AssetCache | null = null

/**
 * Get the default shared AssetCache instance.
 * Creates one if it doesn't exist.
 */
export function getDefaultAssetCache(): AssetCache {
  if (!defaultCache) {
    defaultCache = new AssetCache()
  }
  return defaultCache
}

/**
 * Create a cached AssetProvider from a fetcher using the default cache.
 * Convenience function for common use case.
 * 
 * @param fetcher - Function to fetch assets
 * @returns Cached AssetProvider
 */
export async function createCachedProvider(fetcher: AssetFetcher): Promise<AssetFetcher> {
  const cache = getDefaultAssetCache()
  await cache.initialize()
  return cache.createProvider(fetcher)
}
