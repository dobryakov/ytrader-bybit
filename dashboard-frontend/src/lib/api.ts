import axios from 'axios'

// In dev mode (Docker), use relative path /api which is proxied by Vite to dashboard-api:4050
// In production or when VITE_API_URL is set, use the full URL
// Note: VITE_API_URL should NOT be set in dev mode to use Vite proxy
const API_URL = import.meta.env.VITE_API_URL || '/api'

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': import.meta.env.VITE_API_KEY || '',
  },
})

// Request interceptor to add API key
api.interceptors.request.use(
  (config) => {
    const apiKey = import.meta.env.VITE_API_KEY
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      console.error('Authentication failed. Please check API key.')
    }
    return Promise.reject(error)
  }
)

export default api

