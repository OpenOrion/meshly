/**
 * Constants for Packable zip file format.
 * Matches Python's ExportConstants for cross-platform compatibility.
 */

export const ExportConstants = {
    /** Fixed date_time for deterministic zip output (not used in TS, but documented for reference) */
    EXPORT_TIME: [2020, 1, 1, 0, 0, 0] as const,

    /** ExtractedPackable with data and json_schema */
    EXTRACTED_FILE: "extracted.json",

    /** Directory containing binary assets */
    ASSETS_DIR: "assets",

    /** Extension for binary assets (arrays, packables, resources) */
    ASSET_EXT: ".bin",

    /**
     * Get the path for an asset in the zip archive.
     * @param checksum - Asset checksum
     * @returns Path like "assets/{checksum}.bin"
     */
    assetPath(checksum: string): string {
        return `${this.ASSETS_DIR}/${checksum}${this.ASSET_EXT}`
    },

    /**
     * Extract checksum from an asset path.
     * @param path - Asset path like "assets/{checksum}.bin"
     * @returns The checksum string
     */
    checksumFromPath(path: string): string {
        const prefixLen = this.ASSETS_DIR.length + 1 // +1 for the /
        const suffixLen = this.ASSET_EXT.length
        return path.slice(prefixLen, -suffixLen)
    }
}
