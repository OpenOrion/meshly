/**
 * Lazy proxy for Packable that defers asset loading until field access.
 *
 * Uses JavaScript Proxy to intercept property access and resolve $ref
 * references on-demand. Assets are only fetched when their corresponding
 * fields are accessed, and results are cached.
 *
 * This is the TypeScript equivalent of Python's LazyModel.
 */

import JSZip from "jszip"
import { AssetProvider } from "./common"
import { ExportConstants } from "./constants"
import { JsonSchema, JsonSchemaUtils } from "./json-schema"
import { ReconstructSchema, SchemaUtils } from "./schema-utils"

/**
 * Interface for the special $ properties exposed by LazyModel proxy.
 */
export interface LazyModelProps<T> {
  /** Array of field names that have been loaded */
  readonly $loaded: string[]
  /** Array of field names currently being loaded */
  readonly $pending: string[]
  /** All available field names (excluding $ prefixed) */
  readonly $fields: string[]
  /** Resolve all fields and return as plain object */
  $resolve(): Promise<T>
  /** Get a specific field by name */
  $get(name: string): Promise<unknown>
  /** Raw data dict with $ref references */
  readonly $data: Record<string, unknown>
  /** JSON schema if available */
  readonly $schema: JsonSchema | undefined
}

/**
 * Lazy proxy for Packable that defers asset loading until field access.
 *
 * @example
 * ```ts
 * // Create lazy packable from extracted data
 * const lazy = LazyModel.create(data, assets, schema)
 * // No assets loaded yet
 *
 * const vertices = await lazy.vertices  // NOW the vertices asset is fetched
 * const faces = await lazy.faces        // NOW the faces asset is fetched
 *
 * // Get all loaded field names
 * console.log(lazy.$loaded)  // ['vertices', 'faces']
 *
 * // Resolve all fields at once
 * const full = await lazy.$resolve()
 * ```
 */
export class LazyModel<T extends Record<string, unknown> = Record<string, unknown>> {
  private readonly _data: Record<string, unknown>
  private readonly _assets: AssetProvider
  private readonly _schema?: JsonSchema
  private readonly _fieldSchemas?: ReconstructSchema
  private readonly _cache: Map<string, unknown> = new Map()
  private readonly _pending: Map<string, Promise<unknown>> = new Map()

  private constructor(
    data: Record<string, unknown>,
    assets: AssetProvider,
    schema?: JsonSchema,
    fieldSchemas?: ReconstructSchema
  ) {
    this._data = data
    this._assets = assets
    this._schema = schema
    this._fieldSchemas = fieldSchemas
  }

  /**
   * Create a lazy packable proxy that resolves fields on access.
   *
   * @param data - Data dict with $ref references
   * @param assets - Asset provider (dict or async fetch function)
   * @param schema - Optional JSON Schema for encoding info
   * @param fieldSchemas - Optional explicit field schemas for nested Packables
   * @returns Proxy that lazily resolves fields
   */
  static create<T extends Record<string, unknown> = Record<string, unknown>>(
    data: Record<string, unknown>,
    assets: AssetProvider,
    schema?: JsonSchema,
    fieldSchemas?: ReconstructSchema
  ): LazyModelProps<T> & T {
    const lazy = new LazyModel<T>(data, assets, schema, fieldSchemas)
    return new Proxy(lazy, {
      get(target, prop: string | symbol) {
        // Handle symbol properties and internal methods
        if (typeof prop === "symbol") {
          return Reflect.get(target, prop)
        }

        // Expose internal properties with $ prefix
        if (prop === "$loaded") {
          return Array.from(target._cache.keys())
        }
        if (prop === "$pending") {
          return Array.from(target._pending.keys())
        }
        if (prop === "$fields") {
          return Object.keys(target._data).filter(k => !k.startsWith("$"))
        }
        if (prop === "$resolve") {
          return target._resolveAll.bind(target)
        }
        if (prop === "$get") {
          return target._getField.bind(target)
        }
        if (prop === "$data") {
          return target._data
        }
        if (prop === "$schema") {
          return target._schema
        }

        // Skip metadata fields
        if (prop.startsWith("$")) {
          return target._data[prop]
        }

        // Check if field exists
        if (!(prop in target._data)) {
          return undefined
        }

        // Return cached value if available
        if (target._cache.has(prop)) {
          return target._cache.get(prop)
        }

        // Return pending promise if already loading
        if (target._pending.has(prop)) {
          return target._pending.get(prop)
        }

        // Start async resolution
        const promise = target._resolveField(prop)
        target._pending.set(prop, promise)

        // Cache result when done and remove from pending
        promise
          .then(value => {
            target._cache.set(prop, value)
            target._pending.delete(prop)
          })
          .catch(() => {
            target._pending.delete(prop)
          })

        return promise
      },

      has(target, prop: string | symbol) {
        if (typeof prop === "symbol") return false
        if (prop.startsWith("$")) return true
        return prop in target._data
      },

      ownKeys(target) {
        return Object.keys(target._data).filter(k => !k.startsWith("$"))
      },

      getOwnPropertyDescriptor(target, prop: string | symbol) {
        if (typeof prop === "symbol") return undefined
        if (prop in target._data) {
          return { configurable: true, enumerable: true }
        }
        return undefined
      }
    }) as unknown as LazyModelProps<T> & T
  }

  /**
   * Resolve a single field.
   */
  private async _resolveField(name: string): Promise<unknown> {
    const value = this._data[name]
    if (value === undefined) {
      return undefined
    }

    const fieldSchema = this._fieldSchemas?.[name]
    const schemaProp = this._schema ? JsonSchemaUtils.getResolvedProperty(this._schema, name) : undefined

    return SchemaUtils._resolveWithProp(value, schemaProp, this._schema!, this._assets, fieldSchema)
  }

  /**
   * Get a specific field, resolving if necessary.
   */
  private async _getField(name: string): Promise<unknown> {
    if (this._cache.has(name)) {
      return this._cache.get(name)
    }
    const value = await this._resolveField(name)
    this._cache.set(name, value)
    return value
  }

  /**
   * Resolve all fields and return as a plain object.
   */
  private async _resolveAll(): Promise<T> {
    const result: Record<string, unknown> = {}
    const fields = Object.keys(this._data).filter(k => !k.startsWith("$"))

    await Promise.all(
      fields.map(async name => {
        if (this._cache.has(name)) {
          result[name] = this._cache.get(name)
        } else {
          const value = await this._resolveField(name)
          this._cache.set(name, value)
          result[name] = value
        }
      })
    )

    return result as T
  }

  /**
   * Create a lazy packable from zip data.
   */
  static async fromZip<T extends Record<string, unknown> = Record<string, unknown>>(
    zipData: ArrayBuffer | Uint8Array,
    fieldSchemas?: ReconstructSchema
  ): Promise<LazyModelProps<T> & T> {
    const zip = await JSZip.loadAsync(zipData)

    // Read data.json
    const dataFile = zip.file(ExportConstants.DATA_FILE)
    if (!dataFile) {
      throw new Error(`${ExportConstants.DATA_FILE} not found in zip file`)
    }
    const dataText = await dataFile.async("text")
    const data: Record<string, unknown> = JSON.parse(dataText)

    // Read schema.json (optional)
    let schema: JsonSchema | undefined
    const schemaFile = zip.file(ExportConstants.SCHEMA_FILE)
    if (schemaFile) {
      const schemaText = await schemaFile.async("text")
      schema = JSON.parse(schemaText)
    }

    // Create lazy asset loader that reads from zip on demand
    const assetCache: Record<string, Uint8Array> = {}
    const assets: AssetProvider = async (checksum: string) => {
      if (assetCache[checksum]) {
        return assetCache[checksum]
      }
      const assetPath = ExportConstants.assetPath(checksum)
      const file = zip.file(assetPath)
      if (!file) {
        throw new Error(`Asset '${checksum}' not found at ${assetPath}`)
      }
      const bytes = await file.async("uint8array")
      assetCache[checksum] = bytes
      return bytes
    }

    return LazyModel.create<T>(data, assets, schema, fieldSchemas)
  }

  /**
   * Get a string representation showing loaded and pending fields.
   */
  toString(): string {
    const title = this._schema?.title || "LazyModel"
    const loaded = Array.from(this._cache.keys())
    const pending = Object.keys(this._data).filter(k => !k.startsWith("$") && !this._cache.has(k))
    return `${title}(loaded=[${loaded.join(", ")}], pending=[${pending.join(", ")}])`
  }
}
