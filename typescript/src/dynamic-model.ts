/**
 * Dynamic model building from JSON schema.
 *
 * Provides utilities for:
 * - Building typed interfaces from JSON schemas at runtime
 * - Instantiating models with resolved data (eager or lazy)
 * - Type-safe access to dynamically shaped data
 *
 * This is the TypeScript equivalent of Python's dynamic_model.py.
 *
 * Note: TypeScript doesn't have runtime type creation like Python's Pydantic,
 * but we can provide type-safe wrappers around dynamic data.
 */

import JSZip from "jszip"
import { AssetProvider } from "./common"
import { ExportConstants } from "./constants"
import { JsonSchema, JsonSchemaProperty, JsonSchemaUtils } from "./json-schema"
import { LazyModel, LazyModelProps } from "./lazy-model"
import { ReconstructSchema, SchemaUtils } from "./schema-utils"

// ============================================================
// Types
// ============================================================

/**
 * A dynamically typed model instance.
 * Provides typed access to fields based on JSON schema.
 */
export interface DynamicModel<T = Record<string, unknown>> {
    /** The resolved data */
    readonly data: T
    /** The JSON schema */
    readonly schema: JsonSchema
    /** Get a field value with type checking */
    get<K extends keyof T>(field: K): T[K]
    /** Check if a field exists */
    has(field: string): boolean
    /** Get all field names */
    fields(): string[]
}

/**
 * Options for instantiating a dynamic model.
 */
export interface InstantiateOptions {
    /** Target array type (not applicable in JS, kept for API compatibility) */
    arrayType?: "numpy" | "jax"
    /** If true, return a lazy proxy that resolves fields on access */
    isLazy?: boolean
    /** Explicit field schemas for nested Packables */
    fieldSchemas?: ReconstructSchema
}

// ============================================================
// Dynamic Model Builder
// ============================================================

/**
 * Builds dynamic models from JSON schemas.
 *
 * Handles meshly's custom array types (array, vertex_buffer, index_sequence)
 * and provides both eager and lazy instantiation.
 *
 * @example
 * ```ts
 * // Instantiate with resolved data (eager)
 * const model = await DynamicModelBuilder.instantiate(schema, data, assets)
 * console.log(model.get('vertices'))
 *
 * // Instantiate with lazy loading (assets fetched on field access)
 * const lazy = await DynamicModelBuilder.instantiate(schema, data, assets, { isLazy: true })
 * const vertices = await lazy.vertices  // Asset fetched NOW
 *
 * // Load from zip file
 * const model = await DynamicModelBuilder.fromZip(zipData)
 * ```
 */
export class DynamicModelBuilder {
    /**
     * Instantiate a dynamic model from schema and data.
     *
     * @param schema - JSON Schema instance
     * @param data - Data dict with $ref references
     * @param assets - Asset provider (dict or callable)
     * @param options - Instantiation options
     * @returns Dynamic model instance or lazy proxy
     */
    static async instantiate<T = Record<string, unknown>>(
        schema: JsonSchema,
        data: Record<string, unknown>,
        assets: AssetProvider,
        options: InstantiateOptions = {}
    ): Promise<DynamicModel<T> | (LazyModelProps<T & Record<string, unknown>> & T)> {
        const { isLazy = false, fieldSchemas } = options

        if (isLazy) {
            return LazyModel.create<T & Record<string, unknown>>(data, assets, schema, fieldSchemas)
        }

        const resolved = await SchemaUtils.resolveFromSchema<T>(schema, data, assets, fieldSchemas)
        return DynamicModelBuilder._createModel<T>(resolved, schema)
    }

    /**
     * Load and instantiate a dynamic model from zip data.
     *
     * @param zipData - Zip file bytes
     * @param options - Instantiation options
     * @returns Dynamic model instance or lazy proxy
     */
    static async fromZip<T = Record<string, unknown>>(
        zipData: ArrayBuffer | Uint8Array,
        options: InstantiateOptions = {}
    ): Promise<DynamicModel<T> | (LazyModelProps<T & Record<string, unknown>> & T)> {
        const { isLazy = false, fieldSchemas } = options

        if (isLazy) {
            return LazyModel.fromZip<T & Record<string, unknown>>(zipData, fieldSchemas)
        }

        const zip = await JSZip.loadAsync(zipData)

        // Read data.json
        const dataFile = zip.file(ExportConstants.DATA_FILE)
        if (!dataFile) {
            throw new Error(`${ExportConstants.DATA_FILE} not found in zip file`)
        }
        const dataText = await dataFile.async("text")
        const data: Record<string, unknown> = JSON.parse(dataText)

        // Read schema.json
        const schemaFile = zip.file(ExportConstants.SCHEMA_FILE)
        if (!schemaFile) {
            throw new Error(`${ExportConstants.SCHEMA_FILE} not found in zip file`)
        }
        const schemaText = await schemaFile.async("text")
        const schema: JsonSchema = JSON.parse(schemaText)

        // Build assets dict
        const assets: Record<string, Uint8Array> = {}
        for (const filePath of Object.keys(zip.files)) {
            if (
                filePath.startsWith(ExportConstants.ASSETS_DIR + "/") &&
                filePath.endsWith(ExportConstants.ASSET_EXT)
            ) {
                const checksum = ExportConstants.checksumFromPath(filePath)
                const file = zip.file(filePath)
                if (file) {
                    assets[checksum] = await file.async("uint8array")
                }
            }
        }

        return DynamicModelBuilder.instantiate<T>(schema, data, assets, { fieldSchemas })
    }

    /**
     * Create a DynamicModel wrapper around resolved data.
     */
    private static _createModel<T>(data: T, schema: JsonSchema): DynamicModel<T> {
        return {
            data,
            schema,
            get<K extends keyof T>(field: K): T[K] {
                return data[field]
            },
            has(field: string): boolean {
                return field in (data as Record<string, unknown>)
            },
            fields(): string[] {
                return JsonSchemaUtils.fieldNames(schema)
            }
        }
    }

    /**
     * Get the Python type equivalent for a JSON schema property.
     * Useful for documentation and type inference.
     */
    static getTypeInfo(prop: JsonSchemaProperty, schema: JsonSchema): string {
        // Resolve $ref
        if (prop.$ref) {
            const resolved = JsonSchemaUtils.resolveRef(schema, prop.$ref)
            if (resolved) {
                prop = resolved
            }
        }

        // Handle anyOf (Optional types)
        if (prop.anyOf) {
            const inner = JsonSchemaUtils.getInnerType(prop)
            if (inner) {
                return `${DynamicModelBuilder.getTypeInfo(inner, schema)} | null`
            }
            return "unknown"
        }

        // Handle array types (meshly custom types)
        if (JsonSchemaUtils.isArrayType(prop)) {
            return "TypedArray"
        }

        // Handle resource types
        if (JsonSchemaUtils.isResourceType(prop)) {
            return "Resource"
        }

        // Handle standard JSON schema types
        switch (prop.type) {
            case "string":
                return "string"
            case "integer":
            case "number":
                return "number"
            case "boolean":
                return "boolean"
            case "null":
                return "null"
            case "array":
                if (prop.items) {
                    return `${DynamicModelBuilder.getTypeInfo(prop.items, schema)}[]`
                }
                return "unknown[]"
            case "object":
                if (prop.additionalProperties && typeof prop.additionalProperties === "object") {
                    return `Record<string, ${DynamicModelBuilder.getTypeInfo(prop.additionalProperties, schema)}>`
                }
                if (prop.properties) {
                    return "object"
                }
                return "Record<string, unknown>"
            default:
                return "unknown"
        }
    }

    /**
     * Get field information from a schema.
     * Returns a map of field names to their type info.
     */
    static getFieldInfo(schema: JsonSchema): Record<string, { type: string; required: boolean; description?: string }> {
        const result: Record<string, { type: string; required: boolean; description?: string }> = {}
        const required = new Set(schema.required ?? [])

        for (const name of JsonSchemaUtils.fieldNames(schema)) {
            const prop = JsonSchemaUtils.getResolvedProperty(schema, name)
            if (prop) {
                result[name] = {
                    type: DynamicModelBuilder.getTypeInfo(prop, schema),
                    required: required.has(name),
                    description: prop.description
                }
            }
        }

        return result
    }
}
