import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:6008',
        changeOrigin: true,
      },
      '/config': {
        target: 'http://127.0.0.1:6008',
        changeOrigin: true,
      },
      '/v1': {
        target: 'http://127.0.0.1:6008',
        changeOrigin: true,
      }
    },
    allowedHosts: 'all',
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.js',
  }
})
