import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: '../assets/seg',
    assetsDir: '',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: null,
        inlineDynamicImports: true
      }
    }
  }
});
