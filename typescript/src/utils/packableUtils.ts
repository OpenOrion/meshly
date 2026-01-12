/**
 * Utilities for Packable serialization and field data handling.
 */

/**
 * Static utilities for Packable field data merging.
 */
export class PackableUtils {
    /**
     * Merge non-array field values into data object (in place).
     *
     * Values like `dim: 2` from metadata.fieldData get merged in.
     * Existing object structures are merged recursively.
     * Python BaseModel metadata keys (__model_class__, __model_module__) are stripped.
     *
     * @param data - Target object to merge into (modified in place)
     * @param fieldData - Field values from metadata
     */
    static mergeFieldData(
        data: Record<string, unknown>,
        fieldData: Record<string, unknown>
    ): void {
        for (const [key, value] of Object.entries(fieldData)) {
            // Skip Python BaseModel reconstruction metadata
            if (key === "__model_class__" || key === "__model_module__") {
                continue
            }

            const existing = data[key]

            if (
                existing &&
                typeof existing === "object" &&
                typeof value === "object" &&
                !ArrayBuffer.isView(existing) &&
                !ArrayBuffer.isView(value)
            ) {
                // Both are objects - merge recursively
                PackableUtils.mergeFieldData(
                    existing as Record<string, unknown>,
                    value as Record<string, unknown>
                )
            } else if (typeof value === "object" && value !== null && !ArrayBuffer.isView(value)) {
                // Value is an object that might contain Python metadata - clean it
                data[key] = PackableUtils.stripModelMetadata(value as Record<string, unknown>)
            } else {
                data[key] = value
            }
        }
    }

    /**
     * Recursively strip Python BaseModel metadata keys from an object.
     */
    static stripModelMetadata(obj: Record<string, unknown>): Record<string, unknown> {
        const result: Record<string, unknown> = {}
        for (const [key, value] of Object.entries(obj)) {
            if (key === "__model_class__" || key === "__model_module__") {
                continue
            }
            if (typeof value === "object" && value !== null && !ArrayBuffer.isView(value)) {
                result[key] = PackableUtils.stripModelMetadata(value as Record<string, unknown>)
            } else {
                result[key] = value
            }
        }
        return result
    }
}
