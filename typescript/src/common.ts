/**
 * Asset fetch function type - takes a checksum and returns asset bytes.
 * Matches Python: Callable[[str], Union[bytes, None, Awaitable[Optional[bytes]]]]
 * 
 * Can return:
 * - Uint8Array/ArrayBuffer: the asset data
 * - null/undefined: asset not found (for cache miss patterns)
 * - Promise: async fetch
 */
export type AssetFetcher = (checksum: string) =>
    | Uint8Array | ArrayBuffer | null | undefined
    | Promise<Uint8Array | ArrayBuffer | null | undefined>

/**
 * Asset provider: either a dict of assets or a fetcher function.
 * Matches Python: Union[Dict[str, bytes], AssetFetcher]
 */
export type AssetProvider = Record<string, Uint8Array | ArrayBuffer> | AssetFetcher

/**
 * Get asset bytes from a provider (dict or callable).
 * Supports both sync and async callables.
 * 
 * Matches Python's SerializationUtils.get_asset()
 * 
 * @param assets - Asset provider (dict or callable)
 * @param checksum - Asset checksum to fetch
 * @returns Asset bytes
 * @throws Error if asset not found
 */
export async function getAsset(assets: AssetProvider, checksum: string): Promise<Uint8Array> {
    let result: Uint8Array | ArrayBuffer | null | undefined

    if (typeof assets === "function") {
        const fetched = assets(checksum)
        // Handle both sync and async returns
        result = fetched instanceof Promise ? await fetched : fetched
    } else {
        result = assets[checksum]
    }

    if (result === null || result === undefined) {
        throw new Error(`Missing asset with checksum '${checksum}'`)
    }

    return result instanceof Uint8Array ? result : new Uint8Array(result)
}
