/**
 * PackableStore for file-based Packable asset storage.
 * 
 * Mirrors Python's PackableStore API for cross-platform compatibility.
 * Works in both Node.js (using node:fs) and browser (using lightning-fs with IndexedDB).
 * 
 * Assets (binary blobs) are stored by their SHA256 checksum, enabling deduplication.
 * Extracted packable data is stored at user-specified keys as JSON files.
 * 
 * Directory structure:
 *     root_dir/
 *         assets/           (ASSETS_DIR)
 *             <checksum1>.bin
 *             <checksum2>.bin
 *         runs/             (extractedDir)
 *             <key>.json
 * 
 * @example
 * ```ts
 * // Node.js (auto-detects)
 * const store = new PackableStore({ rootDir: '/data/my_package' })
 * 
 * // Browser (must call createBrowserStore)
 * const store = await createBrowserStore({ rootDir: '/my_package', dbName: 'myapp' })
 * 
 * await store.saveExtracted('experiment/result', extracted)
 * const loaded = await store.loadExtracted('experiment/result')
 * ```
 */

import { AssetFetcher } from './common'
import { ExportConstants } from './constants'
import { ExtractedPackable } from './packable'

/**
 * Minimal filesystem interface that both Node.js fs/promises and lightning-fs satisfy.
 */
export interface FileSystemAdapter {
    mkdir(path: string, options?: { recursive?: boolean }): Promise<void>
    writeFile(path: string, data: Uint8Array | string, options?: { encoding?: string } | string): Promise<void>
    readFile(path: string, options?: { encoding?: string } | string): Promise<Uint8Array | string>
    access?(path: string): Promise<void>
    stat?(path: string): Promise<{ isFile(): boolean; isDirectory(): boolean }>
    unlink?(path: string): Promise<void>
}

export interface PackableStoreConfig {
    /** Root directory for all storage */
    rootDir: string
    /** Subdirectory for extracted JSON files (default: "runs") */
    extractedDir?: string
    /** Filesystem adapter (defaults to node:fs/promises in Node.js) */
    fs?: FileSystemAdapter
}

export interface BrowserStoreConfig {
    /** Root directory for all storage (virtual path in IndexedDB) */
    rootDir: string
    /** Subdirectory for extracted JSON files (default: "runs") */
    extractedDir?: string
    /** IndexedDB database name (default: "meshly-store") */
    dbName?: string
}

/**
 * Join path segments, handling both / and \ separators.
 * Works in both Node.js and browser without requiring 'path' module.
 */
function joinPath(...segments: string[]): string {
    return segments
        .map((s, i) => (i === 0 ? s : s.replace(/^[/\\]+/, '')))
        .join('/')
        .replace(/[/\\]+/g, '/')
}

/**
 * Get parent directory of a path.
 */
function dirname(p: string): string {
    const parts = p.replace(/[/\\]+/g, '/').split('/')
    parts.pop()
    return parts.join('/') || '/'
}

export class PackableStore {
    readonly rootDir: string
    readonly extractedDir: string
    private readonly _fs: FileSystemAdapter

    constructor(config: PackableStoreConfig) {
        this.rootDir = config.rootDir
        this.extractedDir = config.extractedDir ?? 'runs'

        if (config.fs) {
            this._fs = config.fs
        } else {
            // Try to use Node.js fs/promises (will fail in browser)
            // eslint-disable-next-line @typescript-eslint/no-require-imports
            this._fs = require('fs/promises') as FileSystemAdapter
        }
    }

    // ---------------------------------------------------------------------------
    // Path helpers
    // ---------------------------------------------------------------------------

    /** Directory for binary assets: {rootDir}/assets */
    get assetsPath(): string {
        return joinPath(this.rootDir, ExportConstants.ASSETS_DIR)
    }

    /** Directory for extracted JSON files: {rootDir}/{extractedDir} */
    get extractedPath(): string {
        return joinPath(this.rootDir, this.extractedDir)
    }

    /** Get the filesystem path for an extracted packable's JSON file */
    getExtractedPath(key: string): string {
        return joinPath(this.extractedPath, `${key}.json`)
    }

    /** Get the filesystem path for a binary asset */
    assetFile(checksum: string): string {
        return joinPath(this.rootDir, ExportConstants.getRelativeAssetPath(checksum))
    }

    // ---------------------------------------------------------------------------
    // Extracted (result/settings) operations
    // ---------------------------------------------------------------------------

    /**
     * Save extracted packable JSON to storage.
     * 
     * @param key - Identifier for the packable (e.g., "{runnerId}/result")
     * @param extracted - ExtractedPackable to save (assets should be saved separately)
     */
    async saveExtracted(key: string, extracted: Omit<ExtractedPackable, 'assets'>): Promise<void> {
        const filePath = this.getExtractedPath(key)
        await this._fs.mkdir(dirname(filePath), { recursive: true })
        const json = JSON.stringify({ data: extracted.data, json_schema: extracted.json_schema ?? null })
        await this._fs.writeFile(filePath, json, 'utf8')
    }

    /**
     * Load extracted packable from storage.
     * 
     * @param key - Identifier for the packable
     * @returns ExtractedPackable with data and json_schema (assets loaded via loadAsset)
     * @throws Error if file doesn't exist
     */
    async loadExtracted(key: string): Promise<Omit<ExtractedPackable, 'assets'>> {
        const filePath = this.getExtractedPath(key)
        const content = await this._fs.readFile(filePath, 'utf8')
        const json = typeof content === 'string' ? content : new TextDecoder().decode(content)
        const parsed = JSON.parse(json) as { data: Record<string, unknown>; json_schema?: Record<string, unknown> }
        return {
            data: parsed.data,
            json_schema: parsed.json_schema,
        }
    }

    /**
     * Check if an extracted packable exists in storage.
     */
    async extractedExists(key: string): Promise<boolean> {
        try {
            const filePath = this.getExtractedPath(key)
            if (this._fs.access) {
                await this._fs.access(filePath)
            } else if (this._fs.stat) {
                await this._fs.stat(filePath)
            } else {
                // Fallback: try to read
                await this._fs.readFile(filePath)
            }
            return true
        } catch {
            return false
        }
    }

    // ---------------------------------------------------------------------------
    // Asset operations
    // ---------------------------------------------------------------------------

    /**
     * Save binary asset data to storage.
     * 
     * @param data - Binary data to save
     * @param checksum - SHA256 checksum identifier for the asset
     */
    async saveAsset(data: Buffer | Uint8Array, checksum: string): Promise<void> {
        const assetPath = this.assetFile(checksum)
        await this._fs.mkdir(dirname(assetPath), { recursive: true })
        await this._fs.writeFile(assetPath, data instanceof Uint8Array ? data : new Uint8Array(data))
    }

    /**
     * Load binary asset data from storage.
     * 
     * @param checksum - SHA256 checksum identifier for the asset
     * @returns Binary data of the asset
     * @throws Error if asset doesn't exist
     */
    async loadAsset(checksum: string): Promise<Uint8Array> {
        const content = await this._fs.readFile(this.assetFile(checksum))
        if (content instanceof Uint8Array) return content
        if (typeof content === 'string') return new TextEncoder().encode(content)
        // Handle Buffer (Node.js)
        return new Uint8Array(content as unknown as ArrayBuffer)
    }

    /**
     * Check if an asset exists in storage.
     */
    async assetExists(checksum: string): Promise<boolean> {
        try {
            const assetPath = this.assetFile(checksum)
            if (this._fs.access) {
                await this._fs.access(assetPath)
            } else if (this._fs.stat) {
                await this._fs.stat(assetPath)
            } else {
                // Fallback: try to read
                await this._fs.readFile(assetPath)
            }
            return true
        } catch {
            return false
        }
    }
}

// ---------------------------------------------------------------------------
// Browser support via lightning-fs
// ---------------------------------------------------------------------------

let _lightningFS: typeof import('@isomorphic-git/lightning-fs') | null = null

/**
 * Create a PackableStore for browser environments using IndexedDB storage.
 * 
 * @param config - Browser store configuration
 * @returns PackableStore configured for browser usage
 * 
 * @example
 * ```typescript
 * const store = await createBrowserStore({ 
 *   rootDir: '/my-project',
 *   dbName: 'my-app-storage'
 * })
 * await store.saveAsset(data, checksum)
 * ```
 */
export async function createBrowserStore(config: BrowserStoreConfig): Promise<PackableStore> {
    if (!_lightningFS) {
        // Dynamic import for browser environments
        const LightningFS = (await import('@isomorphic-git/lightning-fs')).default
        _lightningFS = LightningFS
    }

    const fs = new _lightningFS(config.dbName ?? 'meshly-store')

    return new PackableStore({
        rootDir: config.rootDir,
        extractedDir: config.extractedDir,
        fs: fs.promises as unknown as FileSystemAdapter,
    })
}

// ---------------------------------------------------------------------------
// PackableCache - Two-tier LRU cache (matches Python implementation)
// ---------------------------------------------------------------------------

export interface PackableCacheConfig {
    /** PackableStore for disk persistence. Omit for memory-only mode. */
    store?: PackableStore
    /** Key prefix for namespacing within the store's assets dir */
    prefix?: string
    /** Maximum entries in the in-memory LRU cache (default: 10000) */
    maxMemory?: number
}

/**
 * Two-tier LRU cache: in-memory + disk via PackableStore.
 * 
 * Lookup order: memory -> disk -> miss.
 * New entries are written to both tiers.
 * 
 * Mirrors Python's PackableCache API for cross-platform compatibility.
 * 
 * @example
 * ```typescript
 * const store = await createBrowserStore({ rootDir: '/cache' })
 * const cache = new PackableCache({ store, prefix: 'my_cache', maxMemory: 1000 })
 * 
 * // Single operations
 * await cache.put('key1', data1)
 * const result = await cache.get('key1')
 * 
 * // Batch operations
 * await cache.putMany({ key1: data1, key2: data2 })
 * const found = await cache.getMany(new Set(['key1', 'key2']))
 * ```
 */
export class PackableCache {
    private readonly _store: PackableStore | null
    private readonly _prefix: string
    private readonly _maxMemory: number
    private readonly _cache: Map<string, Uint8Array> = new Map()
    private readonly _accessOrder: string[] = []

    constructor(config: PackableCacheConfig = {}) {
        this._store = config.store ?? null
        this._prefix = config.prefix ?? ''
        this._maxMemory = config.maxMemory ?? 10_000
    }

    private _storeKey(key: string): string {
        return this._prefix ? `${this._prefix}/${key}` : key
    }

    // -- public API -----------------------------------------------------------

    /**
     * Get a single item (memory -> disk -> undefined).
     */
    async get(key: string): Promise<Uint8Array | undefined> {
        const result = await this.getMany(new Set([key]))
        return result.get(key)
    }

    /**
     * Put a single item into both tiers.
     */
    async put(key: string, value: Uint8Array): Promise<void> {
        await this.putMany({ [key]: value })
    }

    /**
     * Batch get. Returns only the keys that were found.
     */
    async getMany(keys: Set<string>): Promise<Map<string, Uint8Array>> {
        const found = new Map<string, Uint8Array>()

        // Tier 1: memory
        for (const k of keys) {
            if (this._cache.has(k)) {
                this._moveToEnd(k)
                found.set(k, this._cache.get(k)!)
            }
        }

        // Tier 2: disk (parallel)
        const missing = new Set([...keys].filter(k => !found.has(k)))
        if (missing.size === 0 || this._store === null) {
            return found
        }

        const diskHits = await this._loadManyDisk(missing)

        // Promote disk hits to memory
        if (diskHits.size > 0) {
            for (const [k, v] of diskHits) {
                found.set(k, v)
                this._cache.set(k, v)
                this._moveToEnd(k)
            }
            this._evict()
        }

        return found
    }

    /**
     * Batch put into memory + disk.
     */
    async putMany(items: Record<string, Uint8Array>): Promise<void> {
        // Memory
        for (const [k, v] of Object.entries(items)) {
            if (this._cache.has(k)) {
                this._moveToEnd(k)
            } else {
                this._cache.set(k, v)
                this._accessOrder.push(k)
            }
        }
        this._evict()

        // Disk (parallel)
        if (this._store !== null) {
            await this._saveManyDisk(items)
        }
    }

    /**
     * Clear in-memory cache (disk is not affected).
     */
    clear(): void {
        this._cache.clear()
        this._accessOrder.length = 0
    }

    /**
     * Get the number of items in the in-memory cache.
     */
    get size(): number {
        return this._cache.size
    }

    /**
     * Check if a key exists in the in-memory cache.
     */
    has(key: string): boolean {
        return this._cache.has(key)
    }

    /**
     * Create an AssetFetcher that uses this cache with an upstream fetcher.
     * 
     * When an asset is requested:
     * 1. Check cache first (memory -> disk)
     * 2. If not found, call upstream fetcher
     * 3. Store result in cache
     * 4. Return result
     */
    createProvider(upstream: AssetFetcher): AssetFetcher {
        return async (checksum: string): Promise<Uint8Array | ArrayBuffer> => {
            // Try cache first
            const cached = await this.get(checksum)
            if (cached) return cached

            // Fetch and cache
            const fetched = await upstream(checksum)
            if (fetched === null || fetched === undefined) {
                throw new Error(`Missing asset with checksum '${checksum}'`)
            }
            const uint8 = fetched instanceof ArrayBuffer ? new Uint8Array(fetched) : fetched
            await this.put(checksum, uint8)
            return fetched
        }
    }

    // -- disk I/O (parallelised) ----------------------------------------------

    private async _loadManyDisk(keys: Set<string>): Promise<Map<string, Uint8Array>> {
        const hits = new Map<string, Uint8Array>()

        // Load in parallel
        const results = await Promise.all(
            [...keys].map(async (key): Promise<[string, Uint8Array | null]> => {
                const storeKey = this._storeKey(key)
                try {
                    if (await this._store!.assetExists(storeKey)) {
                        const data = await this._store!.loadAsset(storeKey)
                        return [key, data]
                    }
                } catch {
                    // Ignore errors
                }
                return [key, null]
            })
        )

        for (const [key, data] of results) {
            if (data !== null) {
                hits.set(key, data)
            }
        }

        return hits
    }

    private async _saveManyDisk(items: Record<string, Uint8Array>): Promise<void> {
        // Save in parallel, skipping existing
        await Promise.all(
            Object.entries(items).map(async ([key, data]) => {
                const storeKey = this._storeKey(key)
                try {
                    if (!await this._store!.assetExists(storeKey)) {
                        await this._store!.saveAsset(data, storeKey)
                    }
                } catch {
                    // Ignore errors
                }
            })
        )
    }

    // -- internal -------------------------------------------------------------

    private _moveToEnd(key: string): void {
        const idx = this._accessOrder.indexOf(key)
        if (idx > -1) {
            this._accessOrder.splice(idx, 1)
        }
        this._accessOrder.push(key)
    }

    private _evict(): void {
        // Evict oldest entries to stay within maxMemory
        while (this._cache.size > this._maxMemory && this._accessOrder.length > 0) {
            const oldest = this._accessOrder.shift()
            if (oldest) {
                this._cache.delete(oldest)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Default cache instance helpers
// ---------------------------------------------------------------------------

let _defaultCache: PackableCache | null = null
let _defaultCachePromise: Promise<PackableCache> | null = null

/**
 * Get the default PackableCache instance (browser environment).
 * Creates one lazily with IndexedDB-backed storage.
 */
export async function getDefaultCache(): Promise<PackableCache> {
    if (_defaultCache) return _defaultCache
    if (_defaultCachePromise) return _defaultCachePromise

    _defaultCachePromise = (async () => {
        const store = await createBrowserStore({
            rootDir: '/meshly-cache',
            dbName: 'meshly-asset-cache'
        })
        return new PackableCache({ store, maxMemory: 500 })
    })()
    _defaultCache = await _defaultCachePromise
    return _defaultCache
}
