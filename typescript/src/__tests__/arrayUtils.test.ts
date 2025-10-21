import { beforeEach, describe, expect, it } from 'vitest'
import { ArrayUtils } from '../array'
import JSZip from 'jszip'

describe('ArrayUtils', () => {
    let array1d: Float32Array
    let array2d: Float32Array

    beforeEach(() => {
        // Create test arrays
        // 1D array similar to np.linspace(0, 10, 100)
        array1d = new Float32Array(100)
        for (let i = 0; i < 100; i++) {
            array1d[i] = i * (10 / 99)
        }

        // 2D array (flattened) similar to random 50x3 array
        array2d = new Float32Array(50 * 3)
        for (let i = 0; i < array2d.length; i++) {
            array2d[i] = Math.random()
        }
    })

    it('should encode and decode a 1D array', () => {
        // Encode the array
        const encoded = ArrayUtils.encodeArray(array1d)

        // Check that the encoded data is an EncodedArray
        expect(encoded).toBeDefined()
        expect(encoded.data).toBeInstanceOf(Uint8Array)
        expect(encoded.shape).toEqual([array1d.length])
        expect(encoded.dtype).toBe('float32')
        expect(encoded.itemsize).toBe(4)

        // Check that the encoded data is smaller than the original
        // (This might not always be true for small arrays, but should be for our test data)
        expect(encoded.data.length).toBeLessThan(array1d.byteLength)

        // Print compression ratio
        console.log(`1D array compression ratio: ${encoded.data.length / array1d.byteLength}`)

        // Decode the array
        const decoded = ArrayUtils.decodeArray(encoded.data, {
            shape: encoded.shape,
            dtype: encoded.dtype,
            itemsize: encoded.itemsize
        })

        // Check that the decoded array matches the original
        expect(decoded.length).toBe(array1d.length)

        // Check values with small epsilon for floating point comparison
        for (let i = 0; i < array1d.length; i++) {
            expect(decoded[i]).toBeCloseTo(array1d[i], 4)
        }
    })

    it('should encode and decode a 2D array', () => {
        // Encode the array
        const encoded = ArrayUtils.encodeArray(array2d)

        // Check that the encoded data is an EncodedArray
        expect(encoded).toBeDefined()
        expect(encoded.data).toBeInstanceOf(Uint8Array)
        expect(encoded.shape).toEqual([array2d.length])
        expect(encoded.dtype).toBe('float32')
        expect(encoded.itemsize).toBe(4)

        // Check that the encoded data is smaller than the original
        expect(encoded.data.length).toBeLessThan(array2d.byteLength)

        // Decode the array
        const decoded = ArrayUtils.decodeArray(encoded.data, {
            shape: encoded.shape,
            dtype: encoded.dtype,
            itemsize: encoded.itemsize
        })

        // Check that the decoded array matches the original
        expect(decoded.length).toBe(array2d.length)

        // Check values with small epsilon for floating point comparison
        for (let i = 0; i < array2d.length; i++) {
            expect(decoded[i]).toBeCloseTo(array2d[i], 4)
        }
    })

    it('should load array from zip buffer', async () => {
        // Create a test array
        const testArray = new Float32Array([1.0, 2.5, 3.7, 4.2, 5.9])
        
        // Encode the array using ArrayUtils
        const encoded = ArrayUtils.encodeArray(testArray)
        
        // Create a zip buffer manually (simulating what Python creates)
        const zip = new JSZip()
        
        // Add the encoded array data as array.bin
        zip.file('array.bin', encoded.data)
        
        // Add metadata.json
        const metadata = {
            shape: encoded.shape,
            dtype: encoded.dtype,
            itemsize: encoded.itemsize
        }
        zip.file('metadata.json', JSON.stringify(metadata, null, 2))
        
        // Generate the zip buffer
        const zipBuffer = await zip.generateAsync({ type: 'arraybuffer' })
        
        // Test loadFromZip
        const result = await ArrayUtils.loadFromZip(zipBuffer)
        
        // Verify the loaded array matches the original
        expect(result.array.length).toBe(testArray.length)
        expect(result.array).toBeInstanceOf(Float32Array)
        expect(result.customMetadata).toBeUndefined()
        
        // Check values with small epsilon for floating point comparison
        for (let i = 0; i < testArray.length; i++) {
            expect(result.array[i]).toBeCloseTo(testArray[i], 4)
        }
    })

    it('should load array with custom metadata from zip buffer', async () => {
        // Create a test array
        const testArray = new Float32Array([1.0, 2.5, 3.7])
        
        // Encode the array using ArrayUtils
        const encoded = ArrayUtils.encodeArray(testArray)
        
        // Create a zip buffer with custom metadata (simulating what Python creates)
        const zip = new JSZip()
        
        // Add the encoded array data as array.bin
        zip.file('array.bin', encoded.data)
        
        // Add metadata.json
        const metadata = {
            shape: encoded.shape,
            dtype: encoded.dtype,
            itemsize: encoded.itemsize
        }
        zip.file('metadata.json', JSON.stringify(metadata, null, 2))
        
        // Add custom metadata
        const customMetadata = {
            class_name: 'SensorMetadata',
            module_name: '__main__',
            data: {
                sensor_id: 'sensor_001',
                timestamp: 1234567890.123,
                location: [37.7749, -122.4194, 100.0]
            }
        }
        zip.file('custom_metadata.json', JSON.stringify(customMetadata, null, 2))
        
        // Generate the zip buffer
        const zipBuffer = await zip.generateAsync({ type: 'arraybuffer' })
        
        // Test loadFromZip with custom metadata loading enabled
        const result = await ArrayUtils.loadFromZip(zipBuffer, true)
        
        // Verify the loaded array matches the original
        expect(result.array.length).toBe(testArray.length)
        expect(result.array).toBeInstanceOf(Float32Array)
        
        // Verify custom metadata was loaded
        expect(result.customMetadata).toBeDefined()
        expect(result.customMetadata?.sensor_id).toBe('sensor_001')
        expect(result.customMetadata?.timestamp).toBe(1234567890.123)
        expect(result.customMetadata?.location).toEqual([37.7749, -122.4194, 100.0])
        
        // Check array values
        for (let i = 0; i < testArray.length; i++) {
            expect(result.array[i]).toBeCloseTo(testArray[i], 4)
        }
    })

    it('should handle missing files in zip buffer', async () => {
        // Create a zip without the required files
        const zip = new JSZip()
        zip.file('other_file.txt', 'some content')
        
        const zipBuffer = await zip.generateAsync({ type: 'arraybuffer' })
        
        // Test that it throws an error for missing metadata.json
        await expect(ArrayUtils.loadFromZip(zipBuffer)).rejects.toThrow('metadata.json not found in zip file')
    })

    it('should handle missing array.bin in zip buffer', async () => {
        // Create a zip with only metadata.json
        const zip = new JSZip()
        const metadata = {
            shape: [5],
            dtype: 'float32',
            itemsize: 4
        }
        zip.file('metadata.json', JSON.stringify(metadata, null, 2))
        
        const zipBuffer = await zip.generateAsync({ type: 'arraybuffer' })
        
        // Test that it throws an error for missing array.bin
        await expect(ArrayUtils.loadFromZip(zipBuffer)).rejects.toThrow('array.bin not found in zip file')
    })
})