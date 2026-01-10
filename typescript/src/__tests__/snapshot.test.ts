/**
 * Test snapshot loading functionality
 */
import JSZip from 'jszip'
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer'
import { beforeAll, describe, expect, it } from 'vitest'
import { ArrayUtils, SnapshotUtils } from '../index'


describe('SnapshotUtils', () => {
    beforeAll(async () => {
        await MeshoptDecoder.ready
        await MeshoptEncoder.ready
    })

    /**
     * Helper to create a test snapshot zip buffer (mimicking Python output)
     */
    async function createTestSnapshotZip(): Promise<Uint8Array> {
        const zip = new JSZip()

        // Create test field data
        const velocityData = new Float32Array(300) // 100 vertices * 3 components
        for (let i = 0; i < 300; i++) {
            velocityData[i] = Math.random()
        }

        const pressureData = new Float32Array(100)
        for (let i = 0; i < 100; i++) {
            pressureData[i] = Math.random() * 1000
        }

        // Encode fields using ArrayUtils
        const velocityEncoded = ArrayUtils.encodeArray(velocityData)
        const pressureEncoded = ArrayUtils.encodeArray(pressureData)

        // Create metadata matching Python format
        const metadata = {
            time: 1.5,
            fields: {
                velocity: {
                    name: "velocity",
                    type: "vector",
                    units: "m/s",
                    shape: [100, 3],
                    dtype: "float32",
                    itemsize: 4
                },
                pressure: {
                    name: "pressure",
                    type: "scalar",
                    units: "Pa",
                    shape: [100],
                    dtype: "float32",
                    itemsize: 4
                }
            }
        }

        // Add files to zip
        zip.file("metadata.json", JSON.stringify(metadata, null, 2))
        zip.file("fields/velocity.bin", velocityEncoded.data)
        zip.file("fields/pressure.bin", pressureEncoded.data)

        return await zip.generateAsync({ type: "uint8array" })
    }

    it('should load snapshot metadata', async () => {
        const zipBuffer = await createTestSnapshotZip()
        const metadata = await SnapshotUtils.loadMetadata(zipBuffer)

        expect(metadata.time).toBe(1.5)
        expect(Object.keys(metadata.fields)).toHaveLength(2)
        expect(metadata.fields.velocity).toBeDefined()
        expect(metadata.fields.pressure).toBeDefined()
        expect(metadata.fields.velocity.type).toBe("vector")
        expect(metadata.fields.velocity.units).toBe("m/s")
    })

    it('should get field names', async () => {
        const zipBuffer = await createTestSnapshotZip()
        const fieldNames = await SnapshotUtils.getFieldNames(zipBuffer)

        expect(fieldNames).toContain("velocity")
        expect(fieldNames).toContain("pressure")
        expect(fieldNames).toHaveLength(2)
    })

    it('should get time value', async () => {
        const zipBuffer = await createTestSnapshotZip()
        const time = await SnapshotUtils.getTime(zipBuffer)

        expect(time).toBe(1.5)
    })

    it('should load a specific field', async () => {
        const zipBuffer = await createTestSnapshotZip()
        const velocityField = await SnapshotUtils.loadField(zipBuffer, "velocity")

        expect(velocityField.name).toBe("velocity")
        expect(velocityField.type).toBe("vector")
        expect(velocityField.units).toBe("m/s")
        expect(velocityField.shape).toEqual([100, 3])
        expect(velocityField.data).toBeInstanceOf(Float32Array)
        expect(velocityField.data.length).toBe(300) // 100 * 3
    })

    it('should load multiple specific fields', async () => {
        const zipBuffer = await createTestSnapshotZip()
        const fields = await SnapshotUtils.loadFields(zipBuffer, ["velocity"])

        expect(Object.keys(fields)).toHaveLength(1)
        expect(fields.velocity).toBeDefined()
        expect(fields.velocity.name).toBe("velocity")
    })

    it('should load all fields when no names specified', async () => {
        const zipBuffer = await createTestSnapshotZip()
        const fields = await SnapshotUtils.loadFields(zipBuffer)

        expect(Object.keys(fields)).toHaveLength(2)
        expect(fields.velocity).toBeDefined()
        expect(fields.pressure).toBeDefined()
    })

    it('should load entire snapshot', async () => {
        const zipBuffer = await createTestSnapshotZip()
        const snapshot = await SnapshotUtils.loadFromZip(zipBuffer)

        expect(snapshot.time).toBe(1.5)
        expect(Object.keys(snapshot.fields)).toHaveLength(2)
        expect(snapshot.fields.velocity.data.length).toBe(300)
        expect(snapshot.fields.pressure.data.length).toBe(100)
    })

    it('should throw error for non-existent field', async () => {
        const zipBuffer = await createTestSnapshotZip()

        await expect(
            SnapshotUtils.loadField(zipBuffer, "nonexistent")
        ).rejects.toThrow('Field "nonexistent" not found')
    })
})
