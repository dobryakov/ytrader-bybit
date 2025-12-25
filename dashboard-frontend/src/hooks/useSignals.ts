import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export interface Signal {
  signal_id: string
  signal_type: string
  asset: string
  amount: string
  confidence: number | null
  strategy_id: string
  model_version: string | null
  timestamp: string
  is_warmup: boolean
  horizon: number | null
  model_prediction: string | null  // "UP", "DOWN", or null
  actual_movement: {
    price_from: number | null
    price_to: number | null
    direction: string | null  // "UP", "DOWN", or null
    return_value: number | null
    status: string | null  // "computed", "pending", "waiting", "obsolete", or null
  } | null
  total_pnl: string | null // Total PnL from execution events
  is_model_active: boolean // Whether the model is currently active
}

export interface SignalsResponse {
  signals: Signal[]
  pagination: {
    page: number
    page_size: number
    total: number
    total_pages: number
  }
}

export function useSignals(filters?: {
  signal_type?: string
  asset?: string
  strategy_id?: string
  date_from?: string
  date_to?: string
  page?: number
  page_size?: number
}) {
  return useQuery<SignalsResponse>({
    queryKey: ['signals', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters?.signal_type) params.append('signal_type', filters.signal_type)
      if (filters?.asset) params.append('asset', filters.asset)
      if (filters?.strategy_id) params.append('strategy_id', filters.strategy_id)
      if (filters?.date_from) params.append('date_from', filters.date_from)
      if (filters?.date_to) params.append('date_to', filters.date_to)
      params.append('page', (filters?.page || 1).toString())
      params.append('page_size', (filters?.page_size || 20).toString())

      const response = await api.get(`/v1/signals?${params.toString()}`)
      return response.data
    },
  })
}

