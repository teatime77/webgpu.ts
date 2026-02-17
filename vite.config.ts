import { defineConfig, mergeConfig } from 'vite';
import { baseConfig } from '../vite.config.base';
import { resolve } from 'path';

export default defineConfig(
  mergeConfig(baseConfig, {
    build: {
      lib: {
        entry: resolve(__dirname, 'ts/util.ts'),
        fileName: 'index',
        formats: ['es']
      },
      outDir: './webgpu.ts/public/dist'
    }
  })
);