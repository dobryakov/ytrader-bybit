import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export type DatasetStatus = 'building' | 'ready' | 'failed'

export type SplitStrategy = 'time_based' | 'walk_forward'

export interface Dataset {
  id: string
  symbol: string
  status: DatasetStatus
  split_strategy: SplitStrategy
  strategy_id: string | null
  train_period_start: string | null
  train_period_end: string | null
  validation_period_start: string | null
  validation_period_end: string | null
  test_period_start: string | null
  test_period_end: string | null
  walk_forward_config: any | null
  target_config: any
  feature_registry_version: string
  target_registry_version?: string
  train_records: number
  validation_records: number
  test_records: number
  output_format: string
  storage_path: string | null
  created_at: string
  completed_at: string | null
  estimated_completion: string | null
  error_message: string | null
}

export function useDatasets(filters?: {
  symbol?: string
  status?: DatasetStatus
  limit?: number
}) {
  return useQuery<Dataset[]>({
    queryKey: ['datasets', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters?.symbol) params.append('symbol', filters.symbol)
      if (filters?.status) params.append('status', filters.status)
      if (filters?.limit) params.append('limit', filters.limit.toString())
      else params.append('limit', '100')

      const response = await api.get(`/v1/datasets?${params.toString()}`)
      return response.data
    },
    refetchInterval: 5000, // Auto-refresh every 5 seconds to track building datasets
  })
}

export function useDataset(datasetId: string) {
  return useQuery<Dataset>({
    queryKey: ['dataset', datasetId],
    queryFn: async () => {
      const response = await api.get(`/v1/datasets/${datasetId}`)
      return response.data
    },
    enabled: !!datasetId,
  })
}

