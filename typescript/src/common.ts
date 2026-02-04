/**
 * Asset fetch function type - takes a checksum and returns asset bytes
 */
export type AssetFetcher = (checksum: string) => Promise<Uint8Array | ArrayBuffer>

/**
 * Asset provider: either a dict of assets or a fetcher function
 */
export type AssetProvider = Record<string, Uint8Array | ArrayBuffer> | AssetFetcher
