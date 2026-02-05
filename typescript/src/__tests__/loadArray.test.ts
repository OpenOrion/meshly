import JSZip from 'jszip'
import { MeshoptEncoder } from 'meshoptimizer'
import { describe, expect, it } from 'vitest'
import { ExportConstants } from '../constants'
import { LazyModel } from '../lazy-model'

/**
 * Helper to encode array data using meshoptimizer
 */
async function encodeArray(data: Uint8Array, count: number, stride: number): Promise<Uint8Array> {
    await MeshoptEncoder.ready
    return MeshoptEncoder.encodeVertexBuffer(data, count, stride)
}

/**
 * Create a test mesh zip in NEW format (with assets/ and extracted.json).
 */
async function createTestMeshZipNewFormat(): Promise<ArrayBuffer> {
    const zip = new JSZip()

    // Encode the indexSizes array
    const indexSizesData = new Uint32Array([3])
    const indexSizesEncoded = await encodeArray(
        new Uint8Array(indexSizesData.buffer), 1, 4
    )

    // Create checksum for the asset (simplified - using length as mock checksum)
    const checksum = `indexsizes_${indexSizesEncoded.length}`

    // Add extracted.json with data and json_schema
    zip.file(ExportConstants.EXTRACTED_FILE, JSON.stringify({
        data: {
            indexSizes: {
                $ref: checksum,
                shape: [1],
                dtype: 'uint32',
                itemsize: 4
            },
            mesh_size: {
                vertex_count: 3,
                vertex_size: 12,
                index_count: 3,
                index_size: 4
            }
        },
        json_schema: {
            title: 'Mesh',
            type: 'object',
            'x-module': 'meshly.mesh.Mesh',
            'x-base': 'mesh',
            properties: {
                indexSizes: { type: 'array' }
            }
        }
    }))

    // Add asset
    zip.file(ExportConstants.assetPath(checksum), indexSizesEncoded)

    return await zip.generateAsync({ type: 'arraybuffer' })
}

/**
 * Create a test mesh zip with nested marker arrays in NEW format.
 */
async function createTestMeshWithMarkersZipNewFormat(): Promise<ArrayBuffer> {
    const zip = new JSZip()

    // Encode arrays
    const inletIndices = new Uint32Array([0, 1])
    const inletIndicesEncoded = await encodeArray(
        new Uint8Array(inletIndices.buffer), 2, 4
    )
    const inletChecksum = `inlet_indices_${inletIndicesEncoded.length}`

    const inletOffsets = new Uint32Array([0])
    const inletOffsetsEncoded = await encodeArray(
        new Uint8Array(inletOffsets.buffer), 1, 4
    )
    const offsetChecksum = `inlet_offsets_${inletOffsetsEncoded.length}`

    // Add extracted.json with data and json_schema
    zip.file(ExportConstants.EXTRACTED_FILE, JSON.stringify({
        data: {
            markerIndices: {
                inlet: {
                    $ref: inletChecksum,
                    shape: [2],
                    dtype: 'uint32',
                    itemsize: 4
                }
            },
            markerOffsets: {
                inlet: {
                    $ref: offsetChecksum,
                    shape: [1],
                    dtype: 'uint32',
                    itemsize: 4
                }
            }
        },
        json_schema: {
            title: 'Mesh',
            type: 'object',
            'x-module': 'meshly.mesh.Mesh',
            'x-base': 'mesh',
            properties: {
                markerIndices: {
                    type: 'object',
                    additionalProperties: { type: 'array' }
                },
                markerOffsets: {
                    type: 'object',
                    additionalProperties: { type: 'array' }
                }
            }
        }
    }))

    // Add assets
    zip.file(ExportConstants.assetPath(inletChecksum), inletIndicesEncoded)
    zip.file(ExportConstants.assetPath(offsetChecksum), inletOffsetsEncoded)

    return await zip.generateAsync({ type: 'arraybuffer' })
}

describe('LazyModel', () => {
    it('should lazily load arrays from a zip file', async () => {
        const zipData = await createTestMeshZipNewFormat()

        // Create lazy packable
        const lazy = await LazyModel.fromZip(zipData)

        // Nothing loaded yet
        expect(lazy.$loaded).toEqual([])

        // Load indexSizes on access
        const indexSizes = await lazy.indexSizes

        expect(indexSizes).toBeInstanceOf(Uint32Array)
        expect((indexSizes as Uint32Array).length).toBe(1)
        expect((indexSizes as Uint32Array)[0]).toBe(3)

        // Now it's loaded
        expect(lazy.$loaded).toContain('indexSizes')
    })

    it('should load nested dictionary arrays', async () => {
        const zipData = await createTestMeshWithMarkersZipNewFormat()

        // Create lazy packable
        const lazy = await LazyModel.fromZip(zipData)

        // Load nested marker array - need to await the parent object first
        const markerIndices = await lazy.markerIndices as Record<string, unknown>
        expect(markerIndices).toBeDefined()

        const inletIndices = markerIndices.inlet
        expect(inletIndices).toBeInstanceOf(Uint32Array)
        expect((inletIndices as Uint32Array).length).toBe(2)
        expect(Array.from(inletIndices as Uint32Array)).toEqual([0, 1])
    })

    it('should cache loaded values', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        // First access
        const first = await lazy.indexSizes
        // Second access should return cached value (same reference)
        const second = lazy.indexSizes  // No await needed for cached value

        expect(first).toBe(second)
    })

    it('should return undefined for non-existent fields', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        // Non-existent field returns undefined
        expect(lazy.nonexistent).toBeUndefined()
    })

    it('should resolve all fields at once', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        // Resolve everything
        const resolved = await lazy.$resolve()

        expect(resolved.indexSizes).toBeInstanceOf(Uint32Array)
        expect(resolved.mesh_size).toBeDefined()
    })

    it('should expose field names via $fields', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        expect(lazy.$fields).toContain('indexSizes')
        expect(lazy.$fields).toContain('mesh_size')
    })

    it('should expose $data for raw data access', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        // $data should contain raw unresolved data
        expect(lazy.$data).toBeDefined()
        expect(lazy.$data.indexSizes).toBeDefined()
        expect((lazy.$data.indexSizes as Record<string, unknown>).$ref).toBeDefined()
        expect(lazy.$data.mesh_size).toBeDefined()
    })

    it('should expose $schema for schema access', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        expect(lazy.$schema).toBeDefined()
        expect(lazy.$schema?.title).toBe('Mesh')
        expect(lazy.$schema?.properties).toBeDefined()
    })

    it('should support $get method for explicit field access', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        // $get should resolve the field
        const indexSizes = await lazy.$get('indexSizes')
        expect(indexSizes).toBeInstanceOf(Uint32Array)
        expect((indexSizes as Uint32Array)[0]).toBe(3)

        // Should now be in $loaded
        expect(lazy.$loaded).toContain('indexSizes')
    })

    it('should track $pending fields during loading', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        // Start loading but don't await
        const promise = lazy.indexSizes as Promise<unknown>

        // Field should be pending (may or may not still be pending depending on timing)
        // After resolution, pending should be empty
        await promise

        // After resolution, should not be pending
        expect(lazy.$pending).not.toContain('indexSizes')
        expect(lazy.$loaded).toContain('indexSizes')
    })

    it('should support "in" operator via has trap', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        // Existing fields
        expect('indexSizes' in lazy).toBe(true)
        expect('mesh_size' in lazy).toBe(true)

        // Non-existent field
        expect('nonexistent' in lazy).toBe(false)

        // Special $ properties should exist
        expect('$loaded' in lazy).toBe(true)
        expect('$fields' in lazy).toBe(true)
    })

    it('should support Object.keys via ownKeys trap', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        const keys = Object.keys(lazy)

        expect(keys).toContain('indexSizes')
        expect(keys).toContain('mesh_size')
        // $ properties should not be in ownKeys
        expect(keys).not.toContain('$loaded')
        expect(keys).not.toContain('$fields')
    })

    it('should work with LazyModel.create() and dict assets', async () => {
        // Create encoded array
        const indexSizesData = new Uint32Array([5, 10, 15])
        const indexSizesEncoded = await encodeArray(
            new Uint8Array(indexSizesData.buffer), 3, 4
        )
        const checksum = 'test_checksum_123'

        // Data with $ref
        const data = {
            values: {
                $ref: checksum,
                shape: [3],
                dtype: 'uint32',
                itemsize: 4
            },
            name: 'test'
        }

        // Assets dict
        const assets: Record<string, Uint8Array> = {
            [checksum]: indexSizesEncoded
        }

        // Create lazy model with dict assets
        const lazy = LazyModel.create(data, assets)

        expect(lazy.$loaded).toEqual([])

        // Load values
        const values = await lazy.values
        expect(values).toBeInstanceOf(Uint32Array)
        expect(Array.from(values as Uint32Array)).toEqual([5, 10, 15])

        // Non-array field should work too
        const name = await lazy.name
        expect(name).toBe('test')
    })

    it('should work with LazyModel.create() and async fetcher', async () => {
        // Create encoded array
        const dataArray = new Float32Array([1.5, 2.5, 3.5])
        const encoded = await encodeArray(
            new Uint8Array(dataArray.buffer), 3, 4
        )
        const checksum = 'float_checksum'

        // Assets storage
        const assetStore: Record<string, Uint8Array> = {
            [checksum]: encoded
        }

        // Async fetcher function
        let fetchCount = 0
        const fetcher = async (cs: string): Promise<Uint8Array> => {
            fetchCount++
            const asset = assetStore[cs]
            if (!asset) throw new Error(`Asset ${cs} not found`)
            return asset
        }

        const data = {
            floats: {
                $ref: checksum,
                shape: [3],
                dtype: 'float32',
                itemsize: 4
            }
        }

        const lazy = LazyModel.create(data, fetcher)

        // Fetcher not called yet
        expect(fetchCount).toBe(0)

        // Access triggers fetch
        const floats = await lazy.floats
        expect(fetchCount).toBe(1)
        expect(floats).toBeInstanceOf(Float32Array)
        expect(Array.from(floats as Float32Array)).toEqual([1.5, 2.5, 3.5])

        // Second access uses cache, no new fetch
        await lazy.$get('floats')
        expect(fetchCount).toBe(1)
    })

    it('should handle concurrent access to same field', async () => {
        const zipData = await createTestMeshZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        // Start multiple accesses concurrently
        const promise1 = lazy.indexSizes as Promise<unknown>
        const promise2 = lazy.indexSizes as Promise<unknown>
        const promise3 = lazy.indexSizes as Promise<unknown>

        // All should resolve to same value
        const [result1, result2, result3] = await Promise.all([promise1, promise2, promise3])

        expect(result1).toBe(result2)
        expect(result2).toBe(result3)
        expect(result1).toBeInstanceOf(Uint32Array)
    })

    it('should handle concurrent access to different fields', async () => {
        const zipData = await createTestMeshWithMarkersZipNewFormat()

        const lazy = await LazyModel.fromZip(zipData)

        // Load both fields concurrently
        const [markerIndices, markerOffsets] = await Promise.all([
            lazy.markerIndices as Promise<unknown>,
            lazy.markerOffsets as Promise<unknown>
        ])

        expect(markerIndices).toBeDefined()
        expect(markerOffsets).toBeDefined()
        expect(lazy.$loaded).toContain('markerIndices')
        expect(lazy.$loaded).toContain('markerOffsets')
    })

    it('should handle missing asset error', async () => {
        const data = {
            missing: {
                $ref: 'nonexistent_checksum',
                shape: [1],
                dtype: 'uint32',
                itemsize: 4
            }
        }

        // Fetcher that always fails
        const fetcher = async (cs: string): Promise<Uint8Array> => {
            throw new Error(`Asset ${cs} not found`)
        }

        const lazy = LazyModel.create(data, fetcher)

        // Should reject with error
        await expect(lazy.missing).rejects.toThrow('nonexistent_checksum')
    })

    it('should exclude $ prefixed fields from $fields', async () => {
        const data = {
            $version: '1.0',
            normalField: 'value'
        }

        const lazy = LazyModel.create(data, {})

        // $fields should not include $ prefixed fields
        expect(lazy.$fields).toEqual(['normalField'])
        expect(lazy.$fields).not.toContain('$version')
    })

    it('should allow access to $ prefixed metadata fields', async () => {
        const data = {
            $version: '1.0',
            regularField: 'test'
        }

        const lazy = LazyModel.create(data, {})

        // Direct access to $ prefixed field should return raw value
        expect((lazy as Record<string, unknown>)['$version']).toBe('1.0')
    })
})
