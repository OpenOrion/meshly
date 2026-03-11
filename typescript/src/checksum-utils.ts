/**
 * Checksum utilities for hashing data.
 * 
 * Provides SHA256 checksum computation for bytes and dictionaries,
 * compatible with Python meshly's ChecksumUtils.
 */

/**
 * Utility class for computing checksums.
 * Uses WebCrypto API for cross-platform compatibility (Node.js 18+ and browsers).
 */
export class ChecksumUtils {
    /**
     * Compute SHA256 checksum for bytes.
     * Returns first 16 characters of hex digest for compatibility with Python meshly.
     * 
     * @param data - Bytes to hash (Uint8Array or ArrayBuffer)
     * @returns 16-character hex string (truncated SHA256)
     */
    static async computeBytesChecksum(data: Uint8Array | ArrayBuffer): Promise<string> {
        const buffer = data instanceof ArrayBuffer ? data : data.buffer.slice(
            data.byteOffset,
            data.byteOffset + data.byteLength
        )
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer)
        const hashArray = Array.from(new Uint8Array(hashBuffer))
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 16)
    }

    /**
     * Compute SHA256 checksum for a dictionary/object.
     * 
     * Checksum Format:
     *   SHA256 of compact JSON with sorted keys (no whitespace).
     *   Returns first 16 characters of hex digest.
     * 
     * Why JSON-based:
     *   The data dict contains $ref entries pointing to asset checksums,
     *   so this checksum transitively covers all array/binary content.
     *   This format makes checksum recreation straightforward:
     * 
     *   ```typescript
     *   const compact = JSON.stringify(data, Object.keys(data).sort())
     *   const checksum = await ChecksumUtils.computeBytesChecksum(
     *     new TextEncoder().encode(compact)
     *   )
     *   ```
     * 
     * @param data - Object to hash (will be JSON-serialized with sorted keys)
     * @returns 16-character hex string (truncated SHA256)
     */
    static async computeDictChecksum(data: Record<string, unknown>): Promise<string> {
        const jsonString = ChecksumUtils.toSortedJson(data)
        const bytes = new TextEncoder().encode(jsonString)
        return ChecksumUtils.computeBytesChecksum(bytes)
    }

    /**
     * Convert an object to compact JSON with recursively sorted keys.
     * Produces deterministic output regardless of object key insertion order.
     * 
     * @param obj - Object to serialize
     * @returns Compact JSON string with sorted keys
     */
    static toSortedJson(obj: unknown): string {
        return JSON.stringify(obj, (_, value) => {
            if (value && typeof value === 'object' && !Array.isArray(value)) {
                // Sort object keys recursively
                const sorted: Record<string, unknown> = {}
                for (const key of Object.keys(value).sort()) {
                    sorted[key] = value[key]
                }
                return sorted
            }
            return value
        })
    }

    /**
     * Compute full SHA256 checksum for bytes (64-character hex string).
     * Use this when you need the full hash, not the truncated 16-char version.
     * 
     * @param data - Bytes to hash (Uint8Array or ArrayBuffer)
     * @returns Full 64-character hex string (SHA256)
     */
    static async computeFullChecksum(data: Uint8Array | ArrayBuffer): Promise<string> {
        const buffer = data instanceof ArrayBuffer ? data : data.buffer.slice(
            data.byteOffset,
            data.byteOffset + data.byteLength
        )
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer)
        const hashArray = Array.from(new Uint8Array(hashBuffer))
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
    }
}
