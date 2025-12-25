import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export interface Order {
  id: string
  order_id: string
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
  rejection_reason: string | null
  position_id: string | null
}

export interface OrdersResponse {
  orders: Order[]
  pagination: {
    page: number
    page_size: number
    total: number
    total_pages: number
  }
}

export function useOrders(filters?: {
  asset?: string
  status?: string
  signal_id?: string
  order_id?: string
  side?: string
  date_from?: string
  date_to?: string
  position_id?: string
  page?: number
  page_size?: number
  sort_by?: string
  sort_order?: string
}) {
  return useQuery<OrdersResponse>({
    queryKey: ['orders', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters?.asset) params.append('asset', filters.asset)
      if (filters?.status) params.append('status', filters.status)
      if (filters?.signal_id) params.append('signal_id', filters.signal_id)
      if (filters?.order_id) params.append('order_id', filters.order_id)
      if (filters?.side) params.append('side', filters.side)
      if (filters?.date_from) params.append('date_from', filters.date_from)
      if (filters?.date_to) params.append('date_to', filters.date_to)
      if (filters?.position_id) params.append('position_id', filters.position_id)
      params.append('page', (filters?.page || 1).toString())
      params.append('page_size', (filters?.page_size || 20).toString())
      params.append('sort_by', filters?.sort_by || 'created_at')
      params.append('sort_order', filters?.sort_order || 'desc')

      const response = await api.get(`/v1/orders?${params.toString()}`)
      return response.data
    },
    refetchInterval: 10000,
  })
}

