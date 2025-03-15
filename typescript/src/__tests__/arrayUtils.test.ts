import { beforeEach, describe, expect, it } from 'vitest'
import { ArrayUtils } from '../array'

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
})