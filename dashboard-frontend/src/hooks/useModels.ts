import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export interface ModelMetrics {
  f1_score: number | null
  precision_score: number | null
  recall_score: number | null
  accuracy_score: number | null
  roc_auc_score: number | null
  pr_auc_score: number | null
  balanced_accuracy: number | null
}

export interface Model {
  id: string
  version: string
  file_path: string
  model_type: string
  strategy_id: string | null
  symbol: string | null
  trained_at: string
  training_duration_seconds: number | null
  training_dataset_size: number | null
  training_config: any
  is_active: boolean
  metrics: ModelMetrics | null
}

export interface ModelsResponse {
  models: Model[]
  count: number
}

export function useModels(filters?: {
  symbol?: string
  strategy_id?: string
  is_active?: boolean
}) {
  return useQuery<ModelsResponse>({
    queryKey: ['models', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters?.symbol) params.append('symbol', filters.symbol)
      if (filters?.strategy_id) params.append('strategy_id', filters.strategy_id)
      if (filters?.is_active !== undefined) params.append('is_active', filters.is_active.toString())

      const response = await api.get(`/v1/models?${params.toString()}`)
      return response.data
    },
  })
}

