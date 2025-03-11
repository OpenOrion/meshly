import { decodeVertexBuffer, decodeIndexBuffer, decodeArray } from '../decoder';
import { ArrayMetadata } from '../types';

describe('Decoder functions', () => {
  describe('decodeVertexBuffer', () => {
    it('should decode a vertex buffer', () => {
      // Create a mock encoded vertex buffer
      // In a real scenario, this would be the output of the Python library
      const mockData = new Uint8Array([0xA0, 0x01, 0x02, 0x03, 0x04]);
      
      // Decode the vertex buffer
      const result = decodeVertexBuffer(2, 12, mockData);
      
      // Check that the result is a Float32Array
      expect(result).toBeInstanceOf(Float32Array);
      
      // Check that the result has the correct length
      expect(result.length).toBe(6); // 2 vertices * 3 components
    });
    
    it('should throw an error for invalid data', () => {
      // Create an empty buffer
      const mockData = new Uint8Array([]);
      
      // Expect an error when decoding
      expect(() => decodeVertexBuffer(2, 12, mockData)).toThrow();
    });
    
    it('should throw an error for invalid header', () => {
      // Create a buffer with an invalid header
      const mockData = new Uint8Array([0xB0, 0x01, 0x02, 0x03, 0x04]);
      
      // Expect an error when decoding
      expect(() => decodeVertexBuffer(2, 12, mockData)).toThrow();
    });
  });
  
  describe('decodeIndexBuffer', () => {
    it('should decode an index buffer', () => {
      // Create a mock encoded index buffer
      const mockData = new Uint8Array([0xE0, 0x01, 0x02, 0x03, 0x04]);
      
      // Decode the index buffer
      const result = decodeIndexBuffer(3, 4, mockData);
      
      // Check that the result is a Uint32Array
      expect(result).toBeInstanceOf(Uint32Array);
      
      // Check that the result has the correct length
      expect(result.length).toBe(3);
    });
    
    it('should throw an error for invalid data', () => {
      // Create an empty buffer
      const mockData = new Uint8Array([]);
      
      // Expect an error when decoding
      expect(() => decodeIndexBuffer(3, 4, mockData)).toThrow();
    });
    
    it('should throw an error for invalid header', () => {
      // Create a buffer with an invalid header
      const mockData = new Uint8Array([0xF0, 0x01, 0x02, 0x03, 0x04]);
      
      // Expect an error when decoding
      expect(() => decodeIndexBuffer(3, 4, mockData)).toThrow();
    });
  });
  
  describe('decodeArray', () => {
    it('should decode an array', () => {
      // Create a mock encoded array
      const mockData = new Uint8Array([0xA0, 0x01, 0x02, 0x03, 0x04]);
      
      // Create mock metadata
      const mockMetadata: ArrayMetadata = {
        shape: [2, 3],
        dtype: 'float32',
        itemsize: 4
      };
      
      // Decode the array
      const result = decodeArray(mockData, mockMetadata);
      
      // Check that the result is a Float32Array
      expect(result).toBeInstanceOf(Float32Array);
      
      // Check that the result has the correct length
      expect(result.length).toBe(6); // 2 * 3 = 6 elements
    });
  });
});