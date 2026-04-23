import { beforeEach, describe, expect, it } from 'vitest'
import { AssetFetcher } from '../common'
import { PackableCache } from '../store'

describe('PackableCache', () => {
    let cache: PackableCache

    beforeEach(() => {
        // Memory-only cache for testing (no disk store)
        cache = new PackableCache({ maxMemory: 100 })
    })

    describe('basic operations', () => {
        it('should store and retrieve values', async () => {
            const data = new Uint8Array([1, 2, 3])
            await cache.put('key1', data)

            const result = await cache.get('key1')
            expect(result).toEqual(data)
        })

        it('should return undefined for missing keys', async () => {
            const result = await cache.get('nonexistent')
            expect(result).toBeUndefined()
        })

        it('should track size', async () => {
            expect(cache.size).toBe(0)

            await cache.put('a', new Uint8Array([1]))
            expect(cache.size).toBe(1)

            await cache.put('b', new Uint8Array([2]))
            expect(cache.size).toBe(2)
        })

        it('should check has() correctly', async () => {
            expect(cache.has('key')).toBe(false)

            await cache.put('key', new Uint8Array([1]))
            expect(cache.has('key')).toBe(true)
        })

        it('should clear memory cache', async () => {
            await cache.put('a', new Uint8Array([1]))
            await cache.put('b', new Uint8Array([2]))
            expect(cache.size).toBe(2)

            cache.clear()
            expect(cache.size).toBe(0)
            expect(await cache.get('a')).toBeUndefined()
        })
    })

    describe('batch operations', () => {
        it('should putMany and getMany', async () => {
            await cache.putMany({
                'x': new Uint8Array([10]),
                'y': new Uint8Array([20]),
                'z': new Uint8Array([30])
            })

            const results = await cache.getMany(new Set(['x', 'z', 'missing']))

            expect(results.size).toBe(2)
            expect(results.get('x')).toEqual(new Uint8Array([10]))
            expect(results.get('z')).toEqual(new Uint8Array([30]))
            expect(results.has('missing')).toBe(false)
        })
    })

    describe('LRU eviction', () => {
        it('should evict oldest entries when maxMemory exceeded', async () => {
            const smallCache = new PackableCache({ maxMemory: 3 })

            await smallCache.put('a', new Uint8Array([1]))
            await smallCache.put('b', new Uint8Array([2]))
            await smallCache.put('c', new Uint8Array([3]))
            expect(smallCache.size).toBe(3)

            // Adding 4th should evict 'a'
            await smallCache.put('d', new Uint8Array([4]))
            expect(smallCache.size).toBe(3)
            expect(smallCache.has('a')).toBe(false)
            expect(smallCache.has('b')).toBe(true)
            expect(smallCache.has('c')).toBe(true)
            expect(smallCache.has('d')).toBe(true)
        })

        it('should update access order on get', async () => {
            const smallCache = new PackableCache({ maxMemory: 3 })

            await smallCache.put('a', new Uint8Array([1]))
            await smallCache.put('b', new Uint8Array([2]))
            await smallCache.put('c', new Uint8Array([3]))

            // Access 'a' to make it recently used
            await smallCache.get('a')

            // Adding 'd' should now evict 'b' (oldest)
            await smallCache.put('d', new Uint8Array([4]))
            expect(smallCache.has('a')).toBe(true)  // recently accessed
            expect(smallCache.has('b')).toBe(false) // evicted
            expect(smallCache.has('c')).toBe(true)
            expect(smallCache.has('d')).toBe(true)
        })
    })

    describe('createProvider', () => {
        it('should return cached value without calling upstream', async () => {
            const data = new Uint8Array([42, 43, 44])
            await cache.put('cached-key', data)

            let upstreamCalled = false
            const upstream: AssetFetcher = () => {
                upstreamCalled = true
                return new Uint8Array([99])
            }

            const provider = cache.createProvider(upstream)
            const result = await provider('cached-key')

            expect(result).toEqual(data)
            expect(upstreamCalled).toBe(false)
        })

        it('should call upstream for cache miss and store result', async () => {
            const upstreamData = new Uint8Array([50, 60])
            const upstream: AssetFetcher = (checksum) => {
                if (checksum === 'fetch-me') return upstreamData
                return null
            }

            const provider = cache.createProvider(upstream)
            const result = await provider('fetch-me')

            expect(result).toEqual(upstreamData)
            // Should now be cached
            expect(await cache.get('fetch-me')).toEqual(upstreamData)
        })

        it('should work with async upstream fetcher', async () => {
            const upstreamData = new Uint8Array([70, 80])
            const upstream: AssetFetcher = async (checksum) => {
                await new Promise(r => setTimeout(r, 1))
                if (checksum === 'async-fetch') return upstreamData
                return null
            }

            const provider = cache.createProvider(upstream)
            const result = await provider('async-fetch')

            expect(result).toEqual(upstreamData)
            expect(await cache.get('async-fetch')).toEqual(upstreamData)
        })

        it('should throw when upstream returns null', async () => {
            const upstream: AssetFetcher = () => null

            const provider = cache.createProvider(upstream)

            await expect(provider('missing'))
                .rejects.toThrow("Missing asset with checksum 'missing'")
        })

        it('should throw when upstream returns undefined', async () => {
            const upstream: AssetFetcher = () => undefined

            const provider = cache.createProvider(upstream)

            await expect(provider('missing'))
                .rejects.toThrow("Missing asset with checksum 'missing'")
        })

        it('should convert ArrayBuffer from upstream to Uint8Array', async () => {
            const upstream: AssetFetcher = () => {
                const buf = new ArrayBuffer(3)
                new Uint8Array(buf).set([1, 2, 3])
                return buf
            }

            const provider = cache.createProvider(upstream)
            const result = await provider('buffer-test')

            // Cached as Uint8Array
            const cached = await cache.get('buffer-test')
            expect(cached).toBeInstanceOf(Uint8Array)
            expect(Array.from(cached!)).toEqual([1, 2, 3])
        })
    })
})
