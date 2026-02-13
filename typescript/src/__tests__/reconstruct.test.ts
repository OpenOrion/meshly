import { MeshoptEncoder } from 'meshoptimizer'
import { describe, expect, it } from 'vitest'
import { ArrayRefInfo } from '../array'
import { AssetProvider } from '../common'
import { Packable, ReconstructSchema } from '../packable'

/**
 * Helper data type for tests - simulates the data + assets from Python
 */
interface TestExtractedData {
    data: Record<string, unknown>
    assets: Record<string, Uint8Array>
}

/**
 * Helper to encode an array using meshoptimizer and return raw encoded bytes.
 * The new format stores raw encoded bytes directly in assets, not packed with metadata.
 */
async function encodeArray(
    values: Float32Array | Uint32Array | Int32Array,
): Promise<Uint8Array> {
    await MeshoptEncoder.ready

    const itemsize = values.BYTES_PER_ELEMENT
    const count = values.length

    // Encode with meshoptimizer vertex buffer encoder
    const encoded = MeshoptEncoder.encodeVertexBuffer(
        new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
        count,
        itemsize
    )

    return encoded
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
 * Now uses the new $ref format with shape/dtype/itemsize metadata in the ref itself
 */
async function createExtractedData(
    fields: Record<string, unknown>,
    arrays: Record<string, Float32Array | Uint32Array | Int32Array>
): Promise<TestExtractedData> {
    const data: Record<string, unknown> = { ...fields }
    const assets: Record<string, Uint8Array> = {}

    for (const [name, values] of Object.entries(arrays)) {
        const dtype = values instanceof Float32Array ? 'float32' :
            values instanceof Uint32Array ? 'uint32' : 'int32'
        const encoded = await encodeArray(values)
        const checksum = await sha256(encoded)

        // New format: $ref includes shape, dtype, itemsize
        const refInfo: ArrayRefInfo = {
            $ref: checksum,
            shape: [values.length],
            dtype,
            itemsize: values.BYTES_PER_ELEMENT
        }
        data[name] = refInfo
        assets[checksum] = encoded
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
            const encoded = await encodeArray(tempArray)
            const tempChecksum = await sha256(encoded)

            const data = {
                name: 'nested',
                fields: {
                    temperature: {
                        $ref: tempChecksum,
                        shape: [2],
                        dtype: 'float32',
                        itemsize: 4
                    }
                }
            }

            const result = await Packable.reconstruct<{
                name: string
                fields: { temperature: Float32Array }
            }>(data, { [tempChecksum]: encoded })

            expect(result.name).toBe('nested')
            expect(result.fields.temperature).toBeInstanceOf(Float32Array)
            expect(Array.from(result.fields.temperature as Float32Array)).toEqual([100.0, 200.0])
        })

        it('handles arrays of objects with refs', async () => {
            const temp1 = new Float32Array([100.0])
            const temp2 = new Float32Array([200.0])
            const encoded1 = await encodeArray(temp1)
            const encoded2 = await encodeArray(temp2)
            const checksum1 = await sha256(encoded1)
            const checksum2 = await sha256(encoded2)

            const data = {
                snapshots: [
                    { time: 0.0, temperature: { $ref: checksum1, shape: [1], dtype: 'float32', itemsize: 4 } },
                    { time: 1.0, temperature: { $ref: checksum2, shape: [1], dtype: 'float32', itemsize: 4 } }
                ]
            }

            type Snapshot = { time: number; temperature: Float32Array }
            const result = await Packable.reconstruct<{ snapshots: Snapshot[] }>(data, {
                [checksum1]: encoded1,
                [checksum2]: encoded2
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

            const result = await Packable.reconstruct(data, { [checksum]: mockPackableBytes }, undefined, schema)

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

            const result = await Packable.reconstruct<{
                items: { id: number }[]
            }>(data, {
                [checksum1]: item1Bytes,
                [checksum2]: item2Bytes
            }, undefined, schema)

            expect(result.items).toHaveLength(2)
            expect(result.items[0]).toEqual({ id: 1 })
            expect(result.items[1]).toEqual({ id: 2 })
        })
    })

    describe('ArrayUtils.reconstruct', () => {
        it('decodes array from ref info correctly', async () => {
            const original = new Float32Array([1.0, 2.0, 3.0, 4.0])
            const encoded = await encodeArray(original)
            const refInfo: ArrayRefInfo = {
                $ref: 'test-checksum',
                shape: [4],
                dtype: 'float32',
                itemsize: 4
            }

            const { ArrayUtils } = await import('../array')
            const decoded = ArrayUtils.reconstruct({ data: encoded, info: refInfo })

            expect(decoded).toBeInstanceOf(Float32Array)
            expect(Array.from(decoded as Float32Array)).toEqual([1.0, 2.0, 3.0, 4.0])
        })

        it('handles uint32 arrays', async () => {
            const original = new Uint32Array([10, 20, 30])
            const encoded = await encodeArray(original)
            const refInfo: ArrayRefInfo = {
                $ref: 'test-checksum',
                shape: [3],
                dtype: 'uint32',
                itemsize: 4
            }

            const { ArrayUtils } = await import('../array')
            const decoded = ArrayUtils.reconstruct({ data: encoded, info: refInfo })

            expect(decoded).toBeInstanceOf(Uint32Array)
            expect(Array.from(decoded as Uint32Array)).toEqual([10, 20, 30])
        })
    })

    describe('with isLazy parameter', () => {
        it('returns LazyModel when isLazy=true', async () => {
            const extracted = await createExtractedData(
                { name: 'LazyTest', value: 123 },
                { vertices: new Float32Array([1.0, 2.0, 3.0]) }
            )

            const result = await Packable.reconstruct(
                extracted.data,
                extracted.assets,
                undefined, // schema
                undefined, // fieldSchemas
                true // isLazy=true
            )

            // Should be a LazyModel proxy
            expect(result).toBeDefined()
            expect(typeof result).toBe('object')

            // Should have LazyModel special properties
            expect('$data' in result).toBe(true)
            expect('$schema' in result).toBe(true)

            // Use $get to access fields - all fields return promises in LazyModel
            const lazyResult = result as { $get: (key: string) => Promise<unknown> }

            // Plain fields are lazily resolved
            expect(await lazyResult.$get('name')).toBe('LazyTest')
            expect(await lazyResult.$get('value')).toBe(123)

            // Array fields are lazily loaded
            const vertices = await lazyResult.$get('vertices')
            expect(vertices).toBeInstanceOf(Float32Array)
            expect(Array.from(vertices as Float32Array)).toEqual([1.0, 2.0, 3.0])
        })

        it('returns eager result when isLazy=false (default)', async () => {
            const extracted = await createExtractedData(
                { name: 'EagerTest' },
                { data: new Float32Array([4.0, 5.0, 6.0]) }
            )

            const result = await Packable.reconstruct(
                extracted.data,
                extracted.assets,
                undefined, // schema
                undefined, // fieldSchemas
                false // isLazy=false (default)
            )

            // Should be plain object, not LazyModel
            expect(result.name).toBe('EagerTest')
            expect(result.data).toBeInstanceOf(Float32Array)
            expect(Array.from(result.data as Float32Array)).toEqual([4.0, 5.0, 6.0])

            // Should NOT have LazyModel special properties
            expect('$data' in result).toBe(false)
            expect('$schema' in result).toBe(false)
        })
    })
})
