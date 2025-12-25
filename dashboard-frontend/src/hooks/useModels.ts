import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export interface ModelMetrics {
  // Classification metrics
  accuracy: number | null
  balanced_accuracy: number | null
  f1_score: number | null
  pr_auc: number | null
  precision: number | null
  recall: number | null
  roc_auc: number | null
  // Regression metrics
  mae: number | null
  mse: number | null
  r2_score: number | null
  rmse: number | null
  // Trading performance metrics
  avg_pnl: number | null
  max_drawdown: number | null
  profit_factor: number | null
  sharpe_ratio: number | null
  total_pnl: number | null
  win_rate: number | null
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

export interface ModelTrainingHistoryItem {
  id: string
  version: string
  model_type: string
  strategy_id: string | null
  symbol: string | null
  trained_at: string
  training_duration_seconds: number | null
  training_dataset_size: number | null
  dataset_id: string | null
  feature_registry_version: string | null
  target_registry_version: string | null
  feature_count: number | null
  is_active: boolean
  created_at: string
  metrics: ModelMetrics | null
}

export function useModelTrainingHistory(filters?: {
  symbol?: string
  strategy_id?: string
  limit?: number
}) {
  return useQuery<ModelTrainingHistoryItem[]>({
    queryKey: ['modelTrainingHistory', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters?.symbol) params.append('symbol', filters.symbol)
      if (filters?.strategy_id) params.append('strategy_id', filters.strategy_id)
      if (filters?.limit) params.append('limit', filters.limit.toString())
      else params.append('limit', '100')

      const response = await api.get(`/v1/models/history?${params.toString()}`)
      return response.data
    },
  })
}

