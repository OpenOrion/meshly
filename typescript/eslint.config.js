import js from '@eslint/js';
import * as tseslint from 'typescript-eslint';
import jestPlugin from 'eslint-plugin-jest';

export default [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.ts'],
    languageOptions: {
      parserOptions: {
        project: './tsconfig.json',
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: {
      '@typescript-eslint': tseslint.plugin,
      'jest': jestPlugin,
    },
    rules: {
      // Add any custom rules here
    },
  },
  {
    // For test files, enable Jest globals
    files: ['**/*.test.ts', '**/*.spec.ts'],
    languageOptions: {
      globals: {
        jest: 'readonly',
        describe: 'readonly',
        it: 'readonly',
        expect: 'readonly',
        beforeEach: 'readonly',
        afterEach: 'readonly',
        beforeAll: 'readonly',
        afterAll: 'readonly',
      },
    },
  },
];