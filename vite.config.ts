// root/webgpu.ts/vite.config.ts
import { defineConfig, mergeConfig } from 'vite';
import { baseConfig } from '../vite.config.base';
import { resolve } from 'path';

export default defineConfig(
  mergeConfig(baseConfig, {
    build: {
      // 各プロジェクト固有の出力先
      outDir: '../dist/webgpu',
      rollupOptions: {
        input: {
          // 各プロジェクト固有のエントリーポイント
          main: resolve(__dirname, 'index.html'),
        }
      }
    }
  })
);