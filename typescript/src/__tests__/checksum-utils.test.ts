import { describe, expect, it } from 'vitest'
import { ChecksumUtils } from '../checksum-utils'

describe('ChecksumUtils', () => {
    describe('computeBytesChecksum', () => {
        it('should match Python: bytes [1,2,3,4]', async () => {
            const data = new Uint8Array([1, 2, 3, 4])
            const checksum = await ChecksumUtils.computeBytesChecksum(data)

            // Must match Python: ChecksumUtils.compute_bytes_checksum(bytes([1,2,3,4]))
            expect(checksum).toBe('9f64a747e1b97f13')
        })

        it('should match Python: empty bytes', async () => {
            const data = new Uint8Array([])
            const checksum = await ChecksumUtils.computeBytesChecksum(data)

            // Must match Python: ChecksumUtils.compute_bytes_checksum(b'')
            expect(checksum).toBe('e3b0c44298fc1c14')
        })

        it('should produce consistent checksums for same data', async () => {
            const data1 = new Uint8Array([10, 20, 30])
            const data2 = new Uint8Array([10, 20, 30])

            const checksum1 = await ChecksumUtils.computeBytesChecksum(data1)
            const checksum2 = await ChecksumUtils.computeBytesChecksum(data2)

            expect(checksum1).toBe(checksum2)
        })

        it('should produce different checksums for different data', async () => {
            const data1 = new Uint8Array([1, 2, 3])
            const data2 = new Uint8Array([4, 5, 6])

            const checksum1 = await ChecksumUtils.computeBytesChecksum(data1)
            const checksum2 = await ChecksumUtils.computeBytesChecksum(data2)

            expect(checksum1).not.toBe(checksum2)
        })

        it('should handle ArrayBuffer input', async () => {
            const buffer = new ArrayBuffer(4)
            const view = new Uint8Array(buffer)
            view.set([1, 2, 3, 4])

            const checksum = await ChecksumUtils.computeBytesChecksum(buffer)
            expect(checksum).toBe('9f64a747e1b97f13')
        })
    })

    describe('computeDictChecksum', () => {
        it('should match Python: simple dict', async () => {
            const data = { name: 'test', value: 42 }
            const checksum = await ChecksumUtils.computeDictChecksum(data)

            // Must match Python: ChecksumUtils.compute_dict_checksum({'name': 'test', 'value': 42})
            expect(checksum).toBe('9a304be829134dbe')
        })

        it('should match Python: same checksum regardless of key order', async () => {
            const data1 = { a: 1, b: 2, c: 3 }
            const data2 = { c: 3, a: 1, b: 2 }

            const checksum1 = await ChecksumUtils.computeDictChecksum(data1)
            const checksum2 = await ChecksumUtils.computeDictChecksum(data2)

            // Both must match Python: ChecksumUtils.compute_dict_checksum({'a': 1, 'b': 2, 'c': 3})
            expect(checksum1).toBe('e6a3385fb77c287a')
            expect(checksum2).toBe('e6a3385fb77c287a')
        })

        it('should match Python: nested objects', async () => {
            const data1 = { outer: { b: 2, a: 1 } }
            const data2 = { outer: { a: 1, b: 2 } }

            const checksum1 = await ChecksumUtils.computeDictChecksum(data1)
            const checksum2 = await ChecksumUtils.computeDictChecksum(data2)

            // Both must match Python: ChecksumUtils.compute_dict_checksum({'outer': {'b': 2, 'a': 1}})
            expect(checksum1).toBe('8a14b37c210b85f4')
            expect(checksum2).toBe('8a14b37c210b85f4')
        })

        it('should match Python with deeply nested reverse key order', async () => {
            // Keys in reverse alphabetical order - must still match Python
            const nested_reverse = { z: { y: 2, x: 1 }, a: 0 }
            const checksum = await ChecksumUtils.computeDictChecksum(nested_reverse)

            // Must match Python: ChecksumUtils.compute_dict_checksum({'z': {'y': 2, 'x': 1}, 'a': 0})
            expect(checksum).toBe('966d207fa33c472f')
        })

        it('should match Python: dict with array', async () => {
            const data = { items: [1, 2, 3], name: 'list' }
            const checksum = await ChecksumUtils.computeDictChecksum(data)

            // Must match Python: ChecksumUtils.compute_dict_checksum({'items': [1, 2, 3], 'name': 'list'})
            expect(checksum).toBe('1533ee0c4e0b32e1')
        })

        it('should match Python: $ref format for asset references', async () => {
            const data = {
                vertices: { $ref: 'abc123def45678', shape: [100, 3], dtype: 'float32', itemsize: 4 },
                name: 'mesh'
            }
            const checksum = await ChecksumUtils.computeDictChecksum(data)

            // Must match Python: ChecksumUtils.compute_dict_checksum(ref_data)
            expect(checksum).toBe('d0b3917fcd207a40')
        })
    })

    describe('computeFullChecksum', () => {
        it('should compute full 64-char SHA256 checksum', async () => {
            const data = new Uint8Array([1, 2, 3, 4])
            const checksum = await ChecksumUtils.computeFullChecksum(data)

            expect(checksum).toHaveLength(64)
            expect(checksum).toMatch(/^[0-9a-f]{64}$/)
        })

        it('should produce well-known hash for empty data', async () => {
            const data = new Uint8Array([])
            const checksum = await ChecksumUtils.computeFullChecksum(data)

            // SHA256 of empty string
            expect(checksum).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
        })
    })

    describe('toSortedJson', () => {
        it('should produce compact JSON with sorted keys', () => {
            const obj = { z: 1, a: 2, m: 3 }
            const json = ChecksumUtils.toSortedJson(obj)

            expect(json).toBe('{"a":2,"m":3,"z":1}')
        })

        it('should sort nested object keys', () => {
            const obj = { outer: { z: 1, a: 2 } }
            const json = ChecksumUtils.toSortedJson(obj)

            expect(json).toBe('{"outer":{"a":2,"z":1}}')
        })

        it('should preserve array order', () => {
            const obj = { items: [3, 1, 2] }
            const json = ChecksumUtils.toSortedJson(obj)

            expect(json).toBe('{"items":[3,1,2]}')
        })

        it('should handle mixed nested structures', () => {
            const obj = {
                b: [{ y: 1, x: 2 }],
                a: { d: 3, c: 4 }
            }
            const json = ChecksumUtils.toSortedJson(obj)

            expect(json).toBe('{"a":{"c":4,"d":3},"b":[{"x":2,"y":1}]}')
        })
    })
})
