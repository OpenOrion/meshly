import JSZip from 'jszip'
import { MeshoptEncoder } from 'meshoptimizer'
import { describe, expect, it } from 'vitest'
import { Mesh } from '../mesh'
import { Packable } from '../packable'

/**
 * Helper to encode array data using meshoptimizer
 */
async function encodeArray(data: Uint8Array, count: number, stride: number): Promise<Uint8Array> {
    await MeshoptEncoder.ready
    return MeshoptEncoder.encodeVertexBuffer(data, count, stride)
}

/**
 * Create a minimal test mesh zip with indexSizes array.
 * This simulates what Python meshly would generate.
 */
async function createTestMeshZip(): Promise<ArrayBuffer> {
    const zip = new JSZip()

    // Add metadata
    zip.file('metadata.json', JSON.stringify({
        class_name: 'Mesh',
        module_name: 'meshly.mesh',
        mesh_size: {
            vertex_count: 3,
            vertex_size: 12,
            index_count: 3,
            index_size: 4
        },
        field_data: { dim: 3 }
    }))

    // Add indexSizes array (meshopt-encoded)
    const indexSizesData = new Uint32Array([3])
    const indexSizesEncoded = await encodeArray(
        new Uint8Array(indexSizesData.buffer), 1, 4
    )
    zip.file('arrays/indexSizes/array.bin', indexSizesEncoded)
    zip.file('arrays/indexSizes/metadata.json', JSON.stringify({
        shape: [1],
        dtype: 'uint32',
        itemsize: 4
    }))

    // Add cellTypes array (meshopt-encoded)
    const cellTypesData = new Uint32Array([5]) // VTK_TRIANGLE
    const cellTypesEncoded = await encodeArray(
        new Uint8Array(cellTypesData.buffer), 1, 4
    )
    zip.file('arrays/cellTypes/array.bin', cellTypesEncoded)
    zip.file('arrays/cellTypes/metadata.json', JSON.stringify({
        shape: [1],
        dtype: 'uint32',
        itemsize: 4
    }))

    return await zip.generateAsync({ type: 'arraybuffer' })
}

/**
 * Create a test mesh zip with nested marker arrays.
 */
async function createTestMeshWithMarkersZip(): Promise<ArrayBuffer> {
    const zip = new JSZip()

    // Add metadata
    zip.file('metadata.json', JSON.stringify({
        class_name: 'Mesh',
        module_name: 'meshly.mesh',
        mesh_size: {
            vertex_count: 4,
            vertex_size: 12,
            index_count: 6,
            index_size: 4
        },
        field_data: { dim: 2 }
    }))

    // Add markerIndices.inlet (nested, meshopt-encoded)
    const inletIndices = new Uint32Array([0, 1])
    const inletIndicesEncoded = await encodeArray(
        new Uint8Array(inletIndices.buffer), 2, 4
    )
    zip.file('arrays/markerIndices/inlet/array.bin', inletIndicesEncoded)
    zip.file('arrays/markerIndices/inlet/metadata.json', JSON.stringify({
        shape: [2],
        dtype: 'uint32',
        itemsize: 4
    }))

    // Add markerOffsets.inlet (nested, meshopt-encoded)
    const inletOffsets = new Uint32Array([0])
    const inletOffsetsEncoded = await encodeArray(
        new Uint8Array(inletOffsets.buffer), 1, 4
    )
    zip.file('arrays/markerOffsets/inlet/array.bin', inletOffsetsEncoded)
    zip.file('arrays/markerOffsets/inlet/metadata.json', JSON.stringify({
        shape: [1],
        dtype: 'uint32',
        itemsize: 4
    }))

    return await zip.generateAsync({ type: 'arraybuffer' })
}

describe('loadArray', () => {
    it('should load a single array from a zip file', async () => {
        const zipData = await createTestMeshZip()

        // Load just the indexSizes array
        const indexSizes = await Mesh.loadArray(zipData, 'indexSizes')

        expect(indexSizes).toBeInstanceOf(Uint32Array)
        expect(indexSizes.length).toBe(1)
        expect((indexSizes as Uint32Array)[0]).toBe(3)
    })

    it('should load nested dictionary arrays using dotted notation', async () => {
        const zipData = await createTestMeshWithMarkersZip()

        // Load nested marker array
        const inletIndices = await Mesh.loadArray(zipData, 'markerIndices.inlet')

        expect(inletIndices).toBeInstanceOf(Uint32Array)
        expect(inletIndices.length).toBe(2)
        expect(Array.from(inletIndices as Uint32Array)).toEqual([0, 1])
    })

    it('should load marker offsets using dotted notation', async () => {
        const zipData = await createTestMeshWithMarkersZip()

        // Load marker offsets
        const inletOffsets = await Mesh.loadArray(zipData, 'markerOffsets.inlet')

        expect(inletOffsets).toBeInstanceOf(Uint32Array)
        expect(inletOffsets.length).toBe(1)
    })

    it('should throw error when array not found', async () => {
        const zipData = await createTestMeshZip()

        await expect(Mesh.loadArray(zipData, 'nonexistent')).rejects.toThrow(
            "Array 'nonexistent' not found in zip file"
        )
    })

    it('should be accessible from Packable base class', async () => {
        const zipData = await createTestMeshZip()

        // loadArray is a static method on Packable
        const indexSizes = await Packable.loadArray(zipData, 'indexSizes')

        expect(indexSizes).toBeInstanceOf(Uint32Array)
    })
})
