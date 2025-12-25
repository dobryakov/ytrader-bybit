import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export interface Container {
  name: string
  image: string
  service: string
  created: string
  status: string
  health_status: 'healthy' | 'unhealthy' | 'starting' | 'running' | 'restarting' | 'exited' | 'unknown'
  ports: string[]
}

export interface ContainersResponse {
  containers: Container[]
  count: number
}

export function useContainers() {
  return useQuery<ContainersResponse>({
    queryKey: ['containers'],
    queryFn: async () => {
      const response = await api.get('/v1/containers')
      return response.data
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  })
}

