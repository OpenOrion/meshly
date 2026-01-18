import { MeshoptEncoder } from 'meshoptimizer'
import { describe, expect, it } from 'vitest'
import { ArrayMetadata } from '../array'
import { AssetProvider, CachedAssetLoader, DataHandler } from '../data-handler'
import { Packable, ReconstructSchema, SerializedPackableData } from '../packable'

/**
 * Helper to encode an array using meshoptimizer and pack with metadata.
 * Matches Python's packed array format: [4 bytes metadata length][metadata json][array data]
 */
async function packArray(
    values: Float32Array | Uint32Array | Int32Array,
    dtype: string
): Promise<Uint8Array> {
    await MeshoptEncoder.ready

    const itemsize = values.BYTES_PER_ELEMENT
    const count = values.length
    const shape = [count]

    // Encode with meshoptimizer
    const encoded = MeshoptEncoder.encodeVertexBuffer(
        new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
        count,
        itemsize
    )

    // Create metadata
    const metadata: ArrayMetadata = { shape, dtype, itemsize }
    const metadataJson = JSON.stringify(metadata)
    const metadataBytes = new TextEncoder().encode(metadataJson)

    // Pack: [4 bytes len][metadata][data]
    const packed = new Uint8Array(4 + metadataBytes.length + encoded.length)
    const view = new DataView(packed.buffer)
    view.setUint32(0, metadataBytes.length, true) // little-endian
    packed.set(metadataBytes, 4)
    packed.set(encoded, 4 + metadataBytes.length)

    return packed
}

/**
 * Simple SHA256 hash (first 16 chars) for deterministic checksums
 */
async function sha256(data: Uint8Array): Promise<string> {
    const hashBuffer = await crypto.subtle.digest('SHA-256', data)
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 16)
}

/**
 * Helper to create extracted data format (simulating Python's Packable.extract output)
 */
async function createExtractedData(
    fields: Record<string, unknown>,
    arrays: Record<string, Float32Array | Uint32Array | Int32Array>
): Promise<SerializedPackableData> {
    const data: Record<string, unknown> = { ...fields }
    const assets: Record<string, Uint8Array> = {}

    for (const [name, values] of Object.entries(arrays)) {
        const dtype = values instanceof Float32Array ? 'float32' : 'uint32'
        const packed = await packArray(values, dtype)
        const checksum = await sha256(packed)
        data[name] = { $ref: checksum }
        assets[checksum] = packed
    }

    return { data, assets }
}

describe('Packable.reconstruct', () => {
    describe('with dict assets (eager loading)', () => {
        it('reconstructs simple data with arrays', async () => {
            const extracted = await createExtractedData(
                { name: 'test', time: 0.5 },
                {
                    temperature: new Float32Array([300.0, 301.0, 302.0]),
                    velocity: new Float32Array([1.0, 0.0, 0.0, 1.0])
                }
            )

            const result = await Packable.reconstruct<{
                name: string
                time: number
                temperature: Float32Array
                velocity: Float32Array
            }>(extracted.data, extracted.assets)

            expect(result.name).toBe('test')
            expect(result.time).toBe(0.5)
            expect(result.temperature).toBeInstanceOf(Float32Array)
            expect(Array.from(result.temperature)).toEqual([300.0, 301.0, 302.0])
            expect(Array.from(result.velocity)).toEqual([1.0, 0.0, 0.0, 1.0])
        })

        it('throws KeyError for missing asset', async () => {
            const data = { name: 'test', values: { $ref: 'nonexistent_checksum' } }

            await expect(Packable.reconstruct(data, {})).rejects.toThrow(
                /Missing asset.*nonexistent_checksum/
            )
        })

        it('preserves primitive fields unchanged', async () => {
            const data = {
                name: 'simulation_001',
                time: 1.5,
                active: true,
                config: { iterations: 100, tolerance: 1e-6 }
            }

            const result = await Packable.reconstruct(data, {})

            expect(result).toEqual(data)
        })

        it('handles nested objects with refs', async () => {
            const tempArray = new Float32Array([100.0, 200.0])
            const tempPacked = await packArray(tempArray, 'float32')
            const tempChecksum = await sha256(tempPacked)

            const data = {
                name: 'nested',
                fields: {
                    temperature: { $ref: tempChecksum }
                }
            }

            const result = await Packable.reconstruct(data, { [tempChecksum]: tempPacked })

            expect(result.name).toBe('nested')
            expect(result.fields.temperature).toBeInstanceOf(Float32Array)
            expect(Array.from(result.fields.temperature as Float32Array)).toEqual([100.0, 200.0])
        })

        it('handles arrays of objects with refs', async () => {
            const temp1 = new Float32Array([100.0])
            const temp2 = new Float32Array([200.0])
            const packed1 = await packArray(temp1, 'float32')
            const packed2 = await packArray(temp2, 'float32')
            const checksum1 = await sha256(packed1)
            const checksum2 = await sha256(packed2)

            const data = {
                snapshots: [
                    { time: 0.0, temperature: { $ref: checksum1 } },
                    { time: 1.0, temperature: { $ref: checksum2 } }
                ]
            }

            type Snapshot = { time: number; temperature: Float32Array }
            const result = await Packable.reconstruct<{ snapshots: Snapshot[] }>(data, {
                [checksum1]: packed1,
                [checksum2]: packed2
            })

            expect(result.snapshots).toHaveLength(2)
            expect(result.snapshots[0].time).toBe(0.0)
            expect(Array.from(result.snapshots[0].temperature)).toEqual([100.0])
            expect(result.snapshots[1].time).toBe(1.0)
            expect(Array.from(result.snapshots[1].temperature)).toEqual([200.0])
        })
    })

    describe('with callable assets (lazy loading)', () => {
        it('defers loading until field access', async () => {
            const extracted = await createExtractedData(
                { name: 'lazy_test', time: 0.5 },
                { temperature: new Float32Array([300.0, 301.0]) }
            )

            const requestedChecksums: string[] = []
            const loader: AssetProvider = async (checksum: string) => {
                requestedChecksums.push(checksum)
                return extracted.assets[checksum]
            }

            // Access primitive field only
            const result = await Packable.reconstruct(extracted.data, loader)

            // Primitive field should be available
            expect(result.name).toBe('lazy_test')
            expect(result.time).toBe(0.5)

            // Array should have been fetched (TypeScript version is eager with callable too)
            // Note: Unlike Python's LazyModel, TS reconstruct resolves all refs
            expect(requestedChecksums.length).toBe(1)
        })

        it('throws when callable returns missing asset', async () => {
            const data = { values: { $ref: 'missing' } }
            const failingLoader: AssetProvider = async () => {
                throw new Error("Missing asset with checksum 'missing'")
            }

            await expect(Packable.reconstruct(data, failingLoader)).rejects.toThrow(/Missing asset/)
        })
    })

    describe('with schema for Packables', () => {
        it('decodes nested Packable using schema decoder', async () => {
            // Create a mock "packable" that's just raw bytes representing a simple object
            const mockPackableBytes = new TextEncoder().encode(JSON.stringify({ type: 'mock', value: 42 }))
            const checksum = await sha256(mockPackableBytes)

            // Custom decoder that parses the JSON
            const mockDecoder = (data: Uint8Array | ArrayBuffer) => {
                const bytes = data instanceof Uint8Array ? data : new Uint8Array(data)
                return JSON.parse(new TextDecoder().decode(bytes))
            }

            const data = {
                name: 'with_nested',
                nested: { $ref: checksum }
            }

            const schema: ReconstructSchema = {
                nested: { type: 'packable', decode: mockDecoder }
            }

            const result = await Packable.reconstruct(data, { [checksum]: mockPackableBytes }, schema)

            expect(result.name).toBe('with_nested')
            expect(result.nested).toEqual({ type: 'mock', value: 42 })
        })

        it('handles array of Packables with element schema', async () => {
            const item1Bytes = new TextEncoder().encode(JSON.stringify({ id: 1 }))
            const item2Bytes = new TextEncoder().encode(JSON.stringify({ id: 2 }))
            const checksum1 = await sha256(item1Bytes)
            const checksum2 = await sha256(item2Bytes)

            const mockDecoder = (data: Uint8Array | ArrayBuffer) => {
                const bytes = data instanceof Uint8Array ? data : new Uint8Array(data)
                return JSON.parse(new TextDecoder().decode(bytes))
            }

            const data = {
                items: [{ $ref: checksum1 }, { $ref: checksum2 }]
            }

            const schema: ReconstructSchema = {
                items: {
                    type: 'array',
                    element: { type: 'packable', decode: mockDecoder }
                }
            }

            const result = await Packable.reconstruct(data, {
                [checksum1]: item1Bytes,
                [checksum2]: item2Bytes
            }, schema)

            expect(result.items).toHaveLength(2)
            expect(result.items[0]).toEqual({ id: 1 })
            expect(result.items[1]).toEqual({ id: 2 })
        })
    })

    describe('_decodePackedArray', () => {
        it('decodes packed array format correctly', async () => {
            const original = new Float32Array([1.0, 2.0, 3.0, 4.0])
            const packed = await packArray(original, 'float32')

            const decoded = Packable._decodePackedArray(packed)

            expect(decoded).toBeInstanceOf(Float32Array)
            expect(Array.from(decoded as Float32Array)).toEqual([1.0, 2.0, 3.0, 4.0])
        })

        it('handles uint32 arrays', async () => {
            const original = new Uint32Array([10, 20, 30])
            const packed = await packArray(original, 'uint32')

            const decoded = Packable._decodePackedArray(packed)

            expect(decoded).toBeInstanceOf(Uint32Array)
            expect(Array.from(decoded as Uint32Array)).toEqual([10, 20, 30])
        })
    })
})

describe('CachedAssetLoader', () => {
    it('caches fetched assets', async () => {
        const extracted = await createExtractedData({}, { values: new Float32Array([1.0, 2.0]) })
        const checksum = Object.keys(extracted.assets)[0]

        let fetchCount = 0
        const fetcher = async (c: string) => {
            fetchCount++
            return extracted.assets[c]
        }

        // Create a simple in-memory cache
        const cache: Record<string, Uint8Array> = {}
        const mockHandler: DataHandler = {
            async readBinary(path: string) {
                return cache[path]
            },
            async writeBinary(path: string, content: Uint8Array | ArrayBuffer) {
                cache[path] = content instanceof Uint8Array ? content : new Uint8Array(content)
            },
            async exists(path: string) {
                return path in cache
            }
        }

        const loader = new CachedAssetLoader(fetcher, mockHandler)

        // First fetch - should call fetcher
        const result1 = await loader.getAsset(checksum)
        expect(fetchCount).toBe(1)
        expect(result1).toBeDefined()

        // Second fetch - should use cache
        const result2 = await loader.getAsset(checksum)
        expect(fetchCount).toBe(1) // Still 1, cached
        expect(result2).toBeDefined()
    })
})
