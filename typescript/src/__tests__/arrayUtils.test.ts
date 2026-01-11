import JSZip from 'jszip'
import { describe, expect, it } from 'vitest'
import { ArrayUtils } from '../array'

describe('ArrayUtils', () => {
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
