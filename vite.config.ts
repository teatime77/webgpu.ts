import { defineConfig, mergeConfig } from 'vite';
import tsconfigPaths from 'vite-tsconfig-paths';
import { baseConfig } from '../vite.config.base';
import { resolve } from 'path';

export default defineConfig(
  mergeConfig(baseConfig, {
    plugins: [tsconfigPaths()],
    build: {
      lib: {
        entry: resolve(__dirname, 'ts/util.ts'),
        fileName: 'webgpu',
        formats: ['es']
      },
      outDir: '../dist/webgpu/assets'
    }
  })
);