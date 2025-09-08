import { beforeEach, describe, expect, it } from 'vitest'
import { MeshUtils, Mesh, EncodedMesh } from '../mesh'
import './setup'

// Interface for a textured mesh with dictionary arrays
interface TexturedMesh extends Mesh {
  textures: {
    diffuse: Float32Array
    normal: Float32Array
    specular: Float32Array
  }
  material_data: {
    surface: {
      roughness: Float32Array
      metallic: Float32Array
    }
    lighting: {
      emission: Float32Array
    }
  }
  material_name: string
  vertex_colors: Float32Array
}

describe('Dictionary Arrays', () => {
  let sampleMesh: TexturedMesh

  beforeEach(() => {
    // Create a simple triangle mesh
    const vertices = new Float32Array([
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      0.5, 1.0, 0.0
    ])

    const indices = new Uint32Array([0, 1, 2])

    // Create texture arrays
    const diffuseTexture = new Float32Array(64 * 64 * 3)
    const normalTexture = new Float32Array(64 * 64 * 3)
    const specularTexture = new Float32Array(64 * 64 * 1)
    
    // Fill with random data
    for (let i = 0; i < diffuseTexture.length; i++) {
      diffuseTexture[i] = Math.random()
    }
    for (let i = 0; i < normalTexture.length; i++) {
      normalTexture[i] = Math.random()
    }
    for (let i = 0; i < specularTexture.length; i++) {
      specularTexture[i] = Math.random()
    }

    // Create material property arrays
    const roughnessMap = new Float32Array(32 * 32)
    const metallicMap = new Float32Array(32 * 32)
    const emissionMap = new Float32Array(32 * 32 * 3)
    
    // Fill with random data
    for (let i = 0; i < roughnessMap.length; i++) {
      roughnessMap[i] = Math.random()
    }
    for (let i = 0; i < metallicMap.length; i++) {
      metallicMap[i] = Math.random()
    }
    for (let i = 0; i < emissionMap.length; i++) {
      emissionMap[i] = Math.random()
    }

    // Create vertex colors
    const vertexColors = new Float32Array(3 * 4) // RGB + Alpha for each vertex
    for (let i = 0; i < vertexColors.length; i++) {
      vertexColors[i] = Math.random()
    }

    sampleMesh = {
      vertices,
      indices,
      textures: {
        diffuse: diffuseTexture,
        normal: normalTexture,
        specular: specularTexture
      },
      material_data: {
        surface: {
          roughness: roughnessMap,
          metallic: metallicMap
        },
        lighting: {
          emission: emissionMap
        }
      },
      material_name: "test_material",
      vertex_colors: vertexColors
    }
  })

  it('should extract nested arrays from dictionary structures', () => {
    // Use the private method through bracket notation for testing
    const extractNestedArrays = (MeshUtils as any).extractNestedArrays
    const allArrays = extractNestedArrays(sampleMesh)

    // Check that all arrays are detected with dotted notation
    const expectedKeys = [
      'vertices', 'indices', 'vertex_colors',
      'textures.diffuse', 'textures.normal', 'textures.specular',
      'material_data.surface.roughness', 'material_data.surface.metallic',
      'material_data.lighting.emission'
    ]

    expectedKeys.forEach(key => {
      expect(allArrays).toHaveProperty(key)
      // Check for either Float32Array or Uint32Array (indices are Uint32Array)
      const array = allArrays[key]
      expect(array instanceof Float32Array || array instanceof Uint32Array).toBe(true)
    })

    // Check that non-array fields are not included
    expect(allArrays).not.toHaveProperty('material_name')
  })

  it('should reconstruct dictionary structure from dotted keys', () => {
    // Create a simple test case
    const arrays = {
      'vertices': new Float32Array([1, 2, 3]),
      'textures.diffuse': new Float32Array([4, 5, 6]),
      'textures.normal': new Float32Array([7, 8, 9]),
      'material_data.surface.roughness': new Float32Array([10, 11, 12])
    }

    // Use the private method through bracket notation for testing
    const reconstructDictionaries = (MeshUtils as any).reconstructDictionaries
    const result = reconstructDictionaries(arrays)

    // Check direct array field
    expect(result.vertices).toEqual(arrays.vertices)

    // Check reconstructed dictionary structure
    expect(result.textures).toBeDefined()
    expect(result.textures.diffuse).toEqual(arrays['textures.diffuse'])
    expect(result.textures.normal).toEqual(arrays['textures.normal'])

    expect(result.material_data).toBeDefined()
    expect(result.material_data.surface).toBeDefined()
    expect(result.material_data.surface.roughness).toEqual(arrays['material_data.surface.roughness'])
  })

  it('should encode and decode mesh with dictionary arrays', () => {
    // Encode the mesh
    const encodedMesh = MeshUtils.encode(sampleMesh)

    // Check that all arrays are encoded
    expect(encodedMesh.arrays).toBeDefined()
    
    const expectedArrayNames = [
      'vertex_colors',
      'textures.diffuse', 'textures.normal', 'textures.specular',
      'material_data.surface.roughness', 'material_data.surface.metallic',
      'material_data.lighting.emission'
    ]

    expectedArrayNames.forEach(name => {
      expect(encodedMesh.arrays).toHaveProperty(name)
    })

    // Decode the mesh
    const decodedMesh = MeshUtils.decode<TexturedMesh>(encodedMesh)

    // Verify basic mesh properties
    expect(decodedMesh.vertices).toBeInstanceOf(Float32Array)
    expect(decodedMesh.indices).toBeInstanceOf(Uint32Array)
    expect(decodedMesh.vertices.length).toBe(sampleMesh.vertices.length)
    expect(decodedMesh.indices!.length).toBe(sampleMesh.indices!.length)

    // Verify dictionary structure is preserved
    expect(decodedMesh.textures).toBeDefined()
    expect(decodedMesh.material_data).toBeDefined()

    // Verify texture arrays
    expect(decodedMesh.textures.diffuse).toBeInstanceOf(Float32Array)
    expect(decodedMesh.textures.normal).toBeInstanceOf(Float32Array)
    expect(decodedMesh.textures.specular).toBeInstanceOf(Float32Array)

    // Verify nested material data
    expect(decodedMesh.material_data.surface).toBeDefined()
    expect(decodedMesh.material_data.lighting).toBeDefined()
    expect(decodedMesh.material_data.surface.roughness).toBeInstanceOf(Float32Array)
    expect(decodedMesh.material_data.surface.metallic).toBeInstanceOf(Float32Array)
    expect(decodedMesh.material_data.lighting.emission).toBeInstanceOf(Float32Array)

    // Verify direct array field
    expect(decodedMesh.vertex_colors).toBeInstanceOf(Float32Array)

    // Check array shapes/lengths
    expect(decodedMesh.textures.diffuse.length).toBe(sampleMesh.textures.diffuse.length)
    expect(decodedMesh.textures.normal.length).toBe(sampleMesh.textures.normal.length)
    expect(decodedMesh.textures.specular.length).toBe(sampleMesh.textures.specular.length)
    expect(decodedMesh.material_data.surface.roughness.length).toBe(sampleMesh.material_data.surface.roughness.length)
    expect(decodedMesh.material_data.surface.metallic.length).toBe(sampleMesh.material_data.surface.metallic.length)
    expect(decodedMesh.material_data.lighting.emission.length).toBe(sampleMesh.material_data.lighting.emission.length)
    expect(decodedMesh.vertex_colors.length).toBe(sampleMesh.vertex_colors.length)

    // Verify data integrity with floating point tolerance
    for (let i = 0; i < sampleMesh.vertices.length; i++) {
      expect(decodedMesh.vertices[i]).toBeCloseTo(sampleMesh.vertices[i], 4)
    }

    for (let i = 0; i < sampleMesh.textures.diffuse.length; i++) {
      expect(decodedMesh.textures.diffuse[i]).toBeCloseTo(sampleMesh.textures.diffuse[i], 4)
    }

    for (let i = 0; i < sampleMesh.material_data.surface.roughness.length; i++) {
      expect(decodedMesh.material_data.surface.roughness[i]).toBeCloseTo(sampleMesh.material_data.surface.roughness[i], 4)
    }
  })

  it('should handle empty dictionary fields', () => {
    // Create a mesh with minimal data
    const minimalMesh: Mesh = {
      vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0.5, 1, 0]),
      indices: new Uint32Array([0, 1, 2])
    }

    // Encode and decode
    const encoded = MeshUtils.encode(minimalMesh)
    const decoded = MeshUtils.decode(encoded)

    // Should work without issues
    expect(decoded.vertices).toBeInstanceOf(Float32Array)
    expect(decoded.indices).toBeInstanceOf(Uint32Array)
    expect(decoded.vertices.length).toBe(minimalMesh.vertices.length)
    expect(decoded.indices!.length).toBe(minimalMesh.indices!.length)
  })

  it('should handle complex nested structures', () => {
    // Create a mesh with deeper nesting
    const complexMesh: Mesh & {
      physics: {
        collision: {
          shapes: {
            box: Float32Array
            sphere: Float32Array
          }
        }
      }
    } = {
      vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0.5, 1, 0]),
      indices: new Uint32Array([0, 1, 2]),
      physics: {
        collision: {
          shapes: {
            box: new Float32Array([1, 2, 3, 4]),
            sphere: new Float32Array([5, 6, 7, 8])
          }
        }
      }
    }

    // Encode and decode
    const encoded = MeshUtils.encode(complexMesh)
    const decoded = MeshUtils.decode(encoded)

    // Check that deep nesting is preserved
    expect(decoded.physics).toBeDefined()
    expect(decoded.physics.collision).toBeDefined()
    expect(decoded.physics.collision.shapes).toBeDefined()
    expect(decoded.physics.collision.shapes.box).toBeInstanceOf(Float32Array)
    expect(decoded.physics.collision.shapes.sphere).toBeInstanceOf(Float32Array)

    // Verify data integrity
    expect(decoded.physics.collision.shapes.box.length).toBe(4)
    expect(decoded.physics.collision.shapes.sphere.length).toBe(4)

    for (let i = 0; i < 4; i++) {
      expect(decoded.physics.collision.shapes.box[i]).toBeCloseTo(complexMesh.physics.collision.shapes.box[i], 4)
      expect(decoded.physics.collision.shapes.sphere[i]).toBeCloseTo(complexMesh.physics.collision.shapes.sphere[i], 4)
    }
  })
})