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

export interface SignalSuccessRateItem {
  hour: string
  total_signals: number
  evaluated_signals: number
  successful_by_direction: number
  successful_by_pnl: number
  success_rate_direction_percent: number | null
  success_rate_pnl_percent: number | null
  avg_confidence: number | null
  buy_signals: number
  sell_signals: number
  total_pnl_sum: number | null
  avg_pnl: number | null
}

export interface SignalSuccessRateResponse {
  data: SignalSuccessRateItem[]
  count: number
}

export function useSignalSuccessRate(filters?: {
  model_version?: string
  asset?: string
  strategy_id?: string
  start_date?: string
  end_date?: string
}) {
  return useQuery<SignalSuccessRateResponse>({
    queryKey: ['signalSuccessRate', filters],
    queryFn: async () => {
      if (!filters?.model_version || !filters?.asset || !filters?.strategy_id) {
        return { data: [], count: 0 }
      }

      const params = new URLSearchParams()
      params.append('model_version', filters.model_version)
      params.append('asset', filters.asset)
      params.append('strategy_id', filters.strategy_id)
      if (filters?.start_date) params.append('start_date', filters.start_date)
      if (filters?.end_date) params.append('end_date', filters.end_date)

      const response = await api.get(`/v1/models/signal-success-rate?${params.toString()}`)
      return response.data
    },
    enabled: !!(filters?.model_version && filters?.asset && filters?.strategy_id),
  })
}

export function useAvailableAssets() {
  return useQuery<string[]>({
    queryKey: ['availableAssets'],
    queryFn: async () => {
      const response = await api.get('/v1/models/available-assets')
      return response.data.assets
    },
  })
}

export function useAvailableStrategies() {
  return useQuery<string[]>({
    queryKey: ['availableStrategies'],
    queryFn: async () => {
      const response = await api.get('/v1/models/available-strategies')
      return response.data.strategies
    },
  })
}

export function useActiveModelVersion(filters?: {
  asset?: string
  strategy_id?: string
}) {
  return useQuery<string | null>({
    queryKey: ['activeModelVersion', filters],
    queryFn: async () => {
      if (!filters?.asset || !filters?.strategy_id) {
        return null
      }

      const params = new URLSearchParams()
      params.append('asset', filters.asset)
      params.append('strategy_id', filters.strategy_id)

      const response = await api.get(`/v1/models/active-version?${params.toString()}`)
      return response.data.version || null
    },
    enabled: !!(filters?.asset && filters?.strategy_id),
  })
}

export interface PredictionInfo {
  split: string
  count: number
  dataset_id: string | null
  created_at: string | null
}

export interface ModelMetricsDetail {
  accuracy: number | null
  precision: number | null
  recall: number | null
  f1_score: number | null
  balanced_accuracy: number | null
  roc_auc: number | null
  pr_auc: number | null
}

export interface BaselineMetricsDetail {
  accuracy: number | null
  precision: number | null
  recall: number | null
  f1_score: number | null
  balanced_accuracy: number | null
  roc_auc: number | null
  pr_auc: number | null
}

export interface TopKMetrics {
  k: number
  accuracy: number | null
  precision: number | null
  recall: number | null
  f1_score: number | null
  balanced_accuracy: number | null
  roc_auc: number | null
  pr_auc: number | null
  lift: number | null
  coverage: number | null
  precision_class_1: number | null
  recall_class_1: number | null
  f1_class_1: number | null
}

export interface MetricComparison {
  model: number | null
  baseline: number | null
  difference: number | null
}

export interface ConfidenceThresholdInfo {
  threshold_value: number
  threshold_source: 'top_k' | 'static'
  top_k_percentage?: number | null
  static_threshold?: number | null
  metric_name?: string | null
}

export interface ModelAnalysisResponse {
  model_version: string
  model_id: string
  predictions: PredictionInfo[]
  model_metrics: ModelMetricsDetail
  baseline_metrics: BaselineMetricsDetail
  top_k_metrics: TopKMetrics[]
  comparison: {
    accuracy: MetricComparison
    f1_score: MetricComparison
    pr_auc: MetricComparison
    roc_auc: MetricComparison
  }
  confidence_threshold_info?: ConfidenceThresholdInfo | null
}

export function useModelAnalysis(version: string) {
  return useQuery<ModelAnalysisResponse>({
    queryKey: ['modelAnalysis', version],
    queryFn: async () => {
      const response = await api.get(`/v1/models/${version}/analysis`)
      return response.data
    },
    enabled: !!version,
  })
}

