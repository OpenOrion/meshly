/**
 * Schema utilities for resolving $ref values during deserialization.
 *
 * Provides utilities for:
 * - Resolving $ref references to arrays, packables, and resources
 * - Type-driven resolution using JSON Schema
 *
 * This is the TypeScript equivalent of Python's schema_utils.py.
 */

import { ArrayRefInfo, ArrayUtils } from "./array"
import { AssetProvider } from "./common"
import { ArrayEncoding, JsonSchema, JsonSchemaProperty, JsonSchemaUtils } from "./json-schema"

// ============================================================
// Types
// ============================================================

/**
 * Decoder function for Packable types
 */
export type PackableDecoder<T> = (data: Uint8Array | ArrayBuffer) => Promise<T> | T

/**
 * Schema for a single field in reconstruct
 */
export type FieldSchema =
    | { type: "array"; element?: FieldSchema } // TypedArray or Array of items
    | { type: "packable"; decode: PackableDecoder<unknown> } // Nested Packable
    | { type: "dict"; value?: FieldSchema } // Dict with uniform value type
    | { type: "object"; fields?: ReconstructSchema } // Object with known field types

/**
 * Schema mapping field names to their types for reconstruction.
 */
export type ReconstructSchema = Record<string, FieldSchema>

/**
 * Reference object with $ref checksum
 */
export interface RefObject {
    $ref: string
    [key: string]: unknown
}

/**
 * Check if a value is a $ref object
 */
export function isRefObject(value: unknown): value is RefObject {
    return typeof value === "object" && value !== null && "$ref" in value
}

/**
 * Check if a $ref object is an array reference (has shape and dtype)
 */
export function isArrayRef(ref: RefObject): ref is RefObject & ArrayRefInfo {
    return "shape" in ref && "dtype" in ref
}

// ============================================================
// Schema Utils
// ============================================================

/**
 * Utilities for resolving $ref values during deserialization.
 */
export class SchemaUtils {
    // =========================================================================
    // Type helpers
    // =========================================================================

    /**
     * Check if a schema variant matches data via a Literal discriminator field.
     * 
     * A discriminator field is a property with a `const` value or single-value `enum`.
     * This is the TypeScript equivalent of Python's _matches_discriminator.
     */
    private static _matchesDiscriminator(
        variant: JsonSchemaProperty,
        data: Record<string, unknown>,
        schema: JsonSchema
    ): boolean {
        // Resolve $ref if present
        if (variant.$ref) {
            const resolved = JsonSchemaUtils.resolveRef(schema, variant.$ref)
            if (resolved) {
                variant = resolved
            }
        }

        const props = variant.properties
        if (!props) return false

        // Check each property for a discriminator (const or single-value enum)
        for (const [fieldName, fieldProp] of Object.entries(props)) {
            if (!(fieldName in data)) continue

            // Check const
            if (fieldProp.const !== undefined) {
                if (data[fieldName] === fieldProp.const) {
                    return true
                }
            }

            // Check single-value enum
            if (fieldProp.enum && fieldProp.enum.length === 1) {
                if (data[fieldName] === fieldProp.enum[0]) {
                    return true
                }
            }
        }

        return false
    }

    /**
     * Resolve a Union type (anyOf/oneOf) by matching discriminators.
     * Falls back to the first non-null variant if no discriminator matches.
     */
    private static _resolveUnionVariant(
        variants: JsonSchemaProperty[],
        data: Record<string, unknown>,
        schema: JsonSchema
    ): JsonSchemaProperty | undefined {
        // Filter out null type
        const nonNull = variants.filter(v => v.type !== "null")
        if (nonNull.length === 0) return undefined
        if (nonNull.length === 1) return nonNull[0]

        // Try discriminator matching first
        for (const variant of nonNull) {
            if (SchemaUtils._matchesDiscriminator(variant, data, schema)) {
                return variant
            }
        }

        // Fallback to first non-null variant
        return nonNull[0]
    }

    // =========================================================================
    // Public: Resolution entry points
    // =========================================================================

    /**
     * Resolve a data object from extracted data and assets using schema.
     *
     * @param schema - JSON Schema for determining field types and encodings
     * @param data - The data dict with $ref references
     * @param assets - Asset provider (dict or fetch function)
     * @param fieldSchemas - Optional explicit field schemas (override JSON schema)
     * @returns Reconstructed data object with resolved references
     */
    static async resolveFromSchema<T = Record<string, unknown>>(
        schema: JsonSchema,
        data: Record<string, unknown>,
        assets: AssetProvider,
        fieldSchemas?: ReconstructSchema
    ): Promise<T> {
        const result: Record<string, unknown> = {}

        for (const [key, value] of Object.entries(data)) {
            // Skip metadata fields
            if (key.startsWith("$")) continue

            const prop = JsonSchemaUtils.getResolvedProperty(schema, key)
            const fieldSchema = fieldSchemas?.[key]

            result[key] = await SchemaUtils._resolveWithProp(value, prop, schema, assets, fieldSchema)
        }

        return result as T
    }

    /**
     * Resolve a value using JSON schema property.
     */
    static async _resolveWithProp(
        value: unknown,
        prop: JsonSchemaProperty | undefined,
        schema: JsonSchema,
        assets: AssetProvider,
        fieldSchema?: FieldSchema
    ): Promise<unknown> {
        if (value === null || value === undefined) {
            return value
        }

        // Resolve schema $ref
        if (prop?.$ref) {
            prop = JsonSchemaUtils.resolveRef(schema, prop.$ref) || prop
        }

        // Handle anyOf/oneOf with discriminator matching for objects
        if (prop?.anyOf || prop?.oneOf) {
            const variants = prop.anyOf || prop.oneOf || []

            // For object values, try discriminator matching
            if (typeof value === "object" && !Array.isArray(value) && !ArrayBuffer.isView(value) && !isRefObject(value)) {
                const matchedVariant = SchemaUtils._resolveUnionVariant(
                    variants,
                    value as Record<string, unknown>,
                    schema
                )
                if (matchedVariant) {
                    prop = matchedVariant
                    // Resolve $ref if the matched variant is a reference
                    if (prop.$ref) {
                        prop = JsonSchemaUtils.resolveRef(schema, prop.$ref) || prop
                    }
                }
            } else {
                // For non-objects, just unwrap Optional
                prop = JsonSchemaUtils.getInnerType(prop) || prop
            }
        }

        // Handle $ref references in data
        if (isRefObject(value)) {
            return SchemaUtils._resolveRef(value, prop, schema, assets, fieldSchema)
        }

        // Handle arrays (JS arrays, not TypedArrays)
        if (Array.isArray(value)) {
            const elementSchema = fieldSchema?.type === "array" ? fieldSchema.element : undefined
            const itemProp = prop?.items
            return Promise.all(
                value.map(v => SchemaUtils._resolveWithProp(v, itemProp, schema, assets, elementSchema))
            )
        }

        // Handle nested objects
        if (typeof value === "object" && !ArrayBuffer.isView(value)) {
            return SchemaUtils._resolveObject(
                value as Record<string, unknown>,
                prop,
                schema,
                assets,
                fieldSchema
            )
        }

        // Primitive - return as-is
        return value
    }

    /**
     * Resolve a $ref reference to an array, packable, or resource.
     */
    private static async _resolveRef(
        ref: RefObject,
        prop: JsonSchemaProperty | undefined,
        schema: JsonSchema,
        assets: AssetProvider,
        fieldSchema?: FieldSchema
    ): Promise<unknown> {
        const checksum = ref.$ref

        // Get asset bytes
        let assetBytes: Uint8Array | ArrayBuffer
        if (typeof assets === "function") {
            assetBytes = await assets(checksum)
        } else {
            const asset = assets[checksum]
            if (!asset) {
                throw new Error(`Missing asset with checksum '${checksum}'`)
            }
            assetBytes = asset
        }
        const bytes = assetBytes instanceof Uint8Array ? assetBytes : new Uint8Array(assetBytes)

        // Use explicit field schema if provided
        if (fieldSchema?.type === "packable" && fieldSchema.decode) {
            return fieldSchema.decode(bytes)
        }

        // Check if it's an array reference (has shape, dtype)
        if (isArrayRef(ref)) {
            // Determine encoding from schema prop
            let encoding: ArrayEncoding = "array"
            if (prop) {
                const innerProp = JsonSchemaUtils.getInnerType(prop) || prop
                if (JsonSchemaUtils.isArrayType(innerProp) && innerProp.type) {
                    encoding = innerProp.type as ArrayEncoding
                }
            }
            return ArrayUtils.reconstruct({ data: bytes, info: ref, encoding })
        }

        // Check if it's a resource reference (has ext field)
        if ("ext" in ref) {
            // TODO: Handle resource decompression (gzip)
            // For now, return raw data with metadata
            return {
                data: bytes,
                ext: ref.ext as string,
                name: (ref.name as string) || ""
            }
        }

        // Assume it's a nested Packable - decode recursively
        // Import dynamically to avoid circular dependency
        const { Packable } = await import("./packable")
        return Packable.decode(bytes)
    }

    /**
     * Resolve a nested object, handling dicts and nested models.
     */
    private static async _resolveObject(
        obj: Record<string, unknown>,
        prop: JsonSchemaProperty | undefined,
        schema: JsonSchema,
        assets: AssetProvider,
        fieldSchema?: FieldSchema
    ): Promise<Record<string, unknown>> {
        const result: Record<string, unknown> = {}

        for (const [k, v] of Object.entries(obj)) {
            // Skip metadata fields
            if (k.startsWith("$")) continue

            // Get nested schema info
            let nestedProp: JsonSchemaProperty | undefined
            let nestedFieldSchema: FieldSchema | undefined

            if (fieldSchema?.type === "dict") {
                nestedFieldSchema = fieldSchema.value
            } else if (fieldSchema?.type === "object") {
                nestedFieldSchema = fieldSchema.fields?.[k]
            }

            // Get property from schema
            if (prop?.additionalProperties && typeof prop.additionalProperties === "object") {
                nestedProp = prop.additionalProperties
                if (nestedProp.$ref) {
                    nestedProp = JsonSchemaUtils.resolveRef(schema, nestedProp.$ref)
                }
            } else if (prop?.properties?.[k]) {
                nestedProp = prop.properties[k]
                if (nestedProp.$ref) {
                    nestedProp = JsonSchemaUtils.resolveRef(schema, nestedProp.$ref)
                }
            }

            result[k] = await SchemaUtils._resolveWithProp(v, nestedProp, schema, assets, nestedFieldSchema)
        }

        return result
    }
}
