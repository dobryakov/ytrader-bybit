import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export interface Position {
  id: string
  asset: string
  size: string
  average_entry_price: string | null
  current_price: string | null
  unrealized_pnl: string | null
  realized_pnl: string | null
  mode: string
  long_size: string | null
  short_size: string | null
  long_avg_price: string | null
  short_avg_price: string | null
  last_updated: string
  created_at: string
  closed_at: string | null
}

export interface PositionsResponse {
  positions: Position[]
  count: number
}

export function usePositions(filters?: {
  asset?: string
  mode?: string
  size_min?: number
  size_max?: number
  position_id?: string
}) {
  return useQuery<PositionsResponse>({
    queryKey: ['positions', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters?.asset) params.append('asset', filters.asset)
      if (filters?.mode) params.append('mode', filters.mode)
      if (filters?.size_min) params.append('size_min', filters.size_min.toString())
      if (filters?.size_max) params.append('size_max', filters.size_max.toString())
      if (filters?.position_id) params.append('position_id', filters.position_id)

      const response = await api.get(`/v1/positions?${params.toString()}`)
      return response.data
    },
    refetchInterval: 10000, // Refetch every 10 seconds
  })
}

export function usePosition(asset: string) {
  return useQuery<Position>({
    queryKey: ['position', asset],
    queryFn: async () => {
      const response = await api.get(`/v1/positions/${asset}`)
      return response.data
    },
    enabled: !!asset,
    refetchInterval: 10000,
  })
}

