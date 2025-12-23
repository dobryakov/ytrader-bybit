import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export interface PnLChartData {
  time: string
  unrealized_pnl: string
  realized_pnl: string
}

export interface PnLChartResponse {
  data: PnLChartData[]
  count: number
}

export interface SignalsConfidenceChartData {
  time: string
  avg_confidence: number | null
  signal_count: number
}

export interface SignalsConfidenceChartResponse {
  data: SignalsConfidenceChartData[]
  count: number
}

export function usePnLChart(filters?: {
  date_from?: string
  date_to?: string
  interval?: string
}) {
  return useQuery<PnLChartResponse>({
    queryKey: ['charts', 'pnl', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters?.date_from) params.append('date_from', filters.date_from)
      if (filters?.date_to) params.append('date_to', filters.date_to)
      params.append('interval', filters?.interval || '1h')

      const response = await api.get(`/v1/charts/pnl?${params.toString()}`)
      return response.data
    },
  })
}

export function useSignalsConfidenceChart(filters?: {
  asset?: string
  strategy_id?: string
  date_from?: string
  date_to?: string
}) {
  return useQuery<SignalsConfidenceChartResponse>({
    queryKey: ['charts', 'signals-confidence', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters?.asset) params.append('asset', filters.asset)
      if (filters?.strategy_id) params.append('strategy_id', filters.strategy_id)
      if (filters?.date_from) params.append('date_from', filters.date_from)
      if (filters?.date_to) params.append('date_to', filters.date_to)

      const response = await api.get(`/v1/charts/signals-confidence?${params.toString()}`)
      return response.data
    },
  })
}

