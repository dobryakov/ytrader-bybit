import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 4051,
    watch: {
      usePolling: true, // Enable polling for Docker volume mounts
    },
    hmr: {
      clientPort: 4051, // Important for HMR to work in Docker
    },
    proxy: {
      '^/api': {
        // Always use internal Docker service name for target
        // This is the backend service accessible from within Docker network
        target: 'http://dashboard-api:4050',
        changeOrigin: true,
        secure: false,
        // When using regex pattern '^/api', Vite keeps the full path including /api
        // So /api/v1/... stays as /api/v1/... (no rewrite needed)
      },
    },
  },
})

