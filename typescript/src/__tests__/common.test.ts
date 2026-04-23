import { describe, expect, it } from 'vitest'
import { AssetFetcher, AssetProvider, getAsset } from '../common'

describe('getAsset', () => {
    describe('dict provider', () => {
        it('should get asset from dict', async () => {
            const data = new Uint8Array([1, 2, 3, 4])
            const assets: AssetProvider = { 'abc123': data }

            const result = await getAsset(assets, 'abc123')

            expect(result).toEqual(data)
        })

        it('should throw on missing asset in dict', async () => {
            const assets: AssetProvider = {}

            await expect(getAsset(assets, 'missing'))
                .rejects.toThrow("Missing asset with checksum 'missing'")
        })

        it('should convert ArrayBuffer to Uint8Array', async () => {
            const buffer = new ArrayBuffer(4)
            new Uint8Array(buffer).set([5, 6, 7, 8])
            const assets: AssetProvider = { 'xyz789': buffer }

            const result = await getAsset(assets, 'xyz789')

            expect(result).toBeInstanceOf(Uint8Array)
            expect(Array.from(result)).toEqual([5, 6, 7, 8])
        })
    })

    describe('sync function provider', () => {
        it('should get asset from sync function', async () => {
            const data = new Uint8Array([10, 20, 30])
            const fetcher: AssetFetcher = (checksum) => {
                if (checksum === 'sync123') return data
                return null
            }

            const result = await getAsset(fetcher, 'sync123')

            expect(result).toEqual(data)
        })

        it('should throw when sync function returns null', async () => {
            const fetcher: AssetFetcher = () => null

            await expect(getAsset(fetcher, 'any'))
                .rejects.toThrow("Missing asset with checksum 'any'")
        })

        it('should throw when sync function returns undefined', async () => {
            const fetcher: AssetFetcher = () => undefined

            await expect(getAsset(fetcher, 'any'))
                .rejects.toThrow("Missing asset with checksum 'any'")
        })
    })

    describe('async function provider', () => {
        it('should get asset from async function', async () => {
            const data = new Uint8Array([100, 200])
            const fetcher: AssetFetcher = async (checksum) => {
                // Simulate async fetch
                await new Promise(resolve => setTimeout(resolve, 1))
                if (checksum === 'async456') return data
                return null
            }

            const result = await getAsset(fetcher, 'async456')

            expect(result).toEqual(data)
        })

        it('should throw when async function returns null', async () => {
            const fetcher: AssetFetcher = async () => {
                await new Promise(resolve => setTimeout(resolve, 1))
                return null
            }

            await expect(getAsset(fetcher, 'any'))
                .rejects.toThrow("Missing asset with checksum 'any'")
        })

        it('should convert ArrayBuffer from async function', async () => {
            const fetcher: AssetFetcher = async () => {
                const buffer = new ArrayBuffer(3)
                new Uint8Array(buffer).set([11, 22, 33])
                return buffer
            }

            const result = await getAsset(fetcher, 'any')

            expect(result).toBeInstanceOf(Uint8Array)
            expect(Array.from(result)).toEqual([11, 22, 33])
        })
    })

    describe('matches Python behavior', () => {
        it('should behave like SerializationUtils.get_asset with dict', async () => {
            // Python: SerializationUtils.get_asset({'abc': b'\x01\x02'}, 'abc')
            const assets: AssetProvider = { 'abc': new Uint8Array([1, 2]) }
            const result = await getAsset(assets, 'abc')
            expect(Array.from(result)).toEqual([1, 2])
        })

        it('should behave like SerializationUtils.get_asset with callable', async () => {
            // Python: SerializationUtils.get_asset(lambda cs: b'\x03\x04', 'xyz')
            const fetcher: AssetFetcher = () => new Uint8Array([3, 4])
            const result = await getAsset(fetcher, 'xyz')
            expect(Array.from(result)).toEqual([3, 4])
        })

        it('should raise KeyError equivalent on missing asset', async () => {
            // Python: raises KeyError("Missing asset with checksum 'missing'")
            const assets: AssetProvider = {}
            await expect(getAsset(assets, 'missing'))
                .rejects.toThrow(/Missing asset with checksum/)
        })
    })
})
