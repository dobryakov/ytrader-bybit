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
  closed?: boolean // Filter by closed status: true = closed, false = active, undefined = all
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

      // Use API parameters for filtering by closed status
      if (filters?.closed === true) {
        params.append('closed_only', 'true')
      } else if (filters?.closed === false) {
        // Default behavior: only active positions (no parameter needed)
        // But we can explicitly set include_closed=false for clarity
      } else {
        // undefined: get all positions
        params.append('include_closed', 'true')
      }

      const response = await api.get(`/v1/positions?${params.toString()}`)
      
      return response.data
    },
    refetchInterval: 10000, // Refetch every 10 seconds
  })
}

export function usePosition(asset: string, mode: string = 'one-way') {
  return useQuery<Position>({
    queryKey: ['position', asset, mode],
    queryFn: async () => {
      const response = await api.get(`/v1/positions/${asset}?mode=${mode}`)
      return response.data
    },
    enabled: !!asset,
    refetchInterval: 10000,
  })
}

export function usePositionById(positionId: string) {
  return useQuery<Position>({
    queryKey: ['position', positionId],
    queryFn: async () => {
      const response = await api.get(`/v1/positions/${positionId}`)
      return response.data
    },
    enabled: !!positionId,
    refetchInterval: 10000,
  })
}


export interface PositionOrder {
  id: string
  order_id: string
  bybit_order_id?: string
  signal_id: string | null
  asset: string
  side: string
  order_type: string
  quantity: string
  price: string | null
  status: string
  filled_quantity: string
  average_price: string | null
  fees: string | null
  created_at: string
  updated_at: string
  executed_at: string | null
  relationship_type: 'opened' | 'increased' | 'decreased' | 'closed' | 'reversed'
  size_delta: string
  execution_price: string
  po_executed_at: string | null
}

export interface PositionOrdersResponse {
  orders: PositionOrder[]
  count: number
  position_id: string
}

export function usePositionOrders(
  asset: string,
  mode: string = 'one-way',
  relationshipType?: 'opened' | 'increased' | 'decreased' | 'closed' | 'reversed'
) {
  return useQuery<PositionOrdersResponse>({
    queryKey: ['position-orders', asset, mode, relationshipType],
    queryFn: async () => {
      const params = new URLSearchParams()
      params.append('mode', mode)
      if (relationshipType) params.append('relationship_type', relationshipType)

      const response = await api.get(`/v1/positions/${asset}/orders?${params.toString()}`)
      return response.data
    },
    enabled: !!asset,
    refetchInterval: 10000, // Refetch every 10 seconds
  })
}

export function usePositionOrdersById(
  positionId: string,
  relationshipType?: 'opened' | 'increased' | 'decreased' | 'closed' | 'reversed'
) {
  return useQuery<PositionOrdersResponse>({
    queryKey: ['position-orders', positionId, relationshipType],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (relationshipType) params.append('relationship_type', relationshipType)

      const response = await api.get(`/v1/positions/${positionId}/orders?${params.toString()}`)
      return response.data
    },
    enabled: !!positionId,
    refetchInterval: 10000, // Refetch every 10 seconds
  })
}

