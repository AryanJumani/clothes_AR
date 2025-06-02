import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    lib: {
      entry: 'src/main.js',
      name: 'ProcessImageLib',
      fileName: () => 'bundle.js',
      formats: ['iife'], // <--- this is key
    }
  }
});
