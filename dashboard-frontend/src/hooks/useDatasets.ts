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
  split_statistics?: {
    train?: {
      class_distribution?: Record<string, number>
      class_balance_ratio?: number
      minority_class_size?: number
      total_classes?: number
      target_statistics?: {
        mean: number
        median: number
        std: number
        min: number
        max: number
        count: number
        zero_targets_count?: number
        zero_targets_percentage?: number
        near_zero_count?: number
        near_zero_percentage?: number
        positive_count?: number
        negative_count?: number
        percentiles?: {
          p1?: number
          p5?: number
          p10?: number
          p25?: number
          p50?: number
          p75?: number
          p90?: number
          p95?: number
          p99?: number
        }
        consecutive_zeros?: {
          sequences_count?: number
          max_consecutive_length?: number
          total_consecutive_zeros?: number
          longest_sequence_start?: string
          longest_sequence_end?: string
        }
      }
    }
    validation?: {
      class_distribution?: Record<string, number>
      class_balance_ratio?: number
      minority_class_size?: number
      total_classes?: number
      target_statistics?: {
        mean: number
        median: number
        std: number
        min: number
        max: number
        count: number
        zero_targets_count?: number
        zero_targets_percentage?: number
        near_zero_count?: number
        near_zero_percentage?: number
        positive_count?: number
        negative_count?: number
        percentiles?: {
          p1?: number
          p5?: number
          p10?: number
          p25?: number
          p50?: number
          p75?: number
          p90?: number
          p95?: number
          p99?: number
        }
        consecutive_zeros?: {
          sequences_count?: number
          max_consecutive_length?: number
          total_consecutive_zeros?: number
          longest_sequence_start?: string
          longest_sequence_end?: string
        }
      }
    }
    test?: {
      class_distribution?: Record<string, number>
      class_balance_ratio?: number
      minority_class_size?: number
      total_classes?: number
      target_statistics?: {
        mean: number
        median: number
        std: number
        min: number
        max: number
        count: number
        zero_targets_count?: number
        zero_targets_percentage?: number
        near_zero_count?: number
        near_zero_percentage?: number
        positive_count?: number
        negative_count?: number
        percentiles?: {
          p1?: number
          p5?: number
          p10?: number
          p25?: number
          p50?: number
          p75?: number
          p90?: number
          p95?: number
          p99?: number
        }
        consecutive_zeros?: {
          sequences_count?: number
          max_consecutive_length?: number
          total_consecutive_zeros?: number
          longest_sequence_start?: string
          longest_sequence_end?: string
        }
      }
    }
    outlier_detection?: {
      threshold: number
      train_mean: number
      train_std: number
      lower_bound: number
      upper_bound: number
      method: string
      total_outliers_detected: number
    }
  }
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
  feature_correlations?: Record<string, number>
  data_quality?: {
    problematic_periods_excluded?: number
    problematic_periods?: Array<{
      start: string
      end: string
      current_price: number
      future_price: number
      current_volume: number
      future_volume: number
    }>
  }
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

export function usePreviousDataset(
  symbol: string | null | undefined,
  strategyId: string | null | undefined,
  currentDatasetId: string | null | undefined,
) {
  return useQuery<Dataset | null>({
    queryKey: ['previousDataset', symbol, strategyId, currentDatasetId],
    queryFn: async () => {
      if (!symbol || !strategyId || !currentDatasetId) {
        return null
      }

      const params = new URLSearchParams()
      params.append('symbol', symbol)
      params.append('status', 'ready')
      params.append('limit', '100')

      const response = await api.get(`/v1/datasets?${params.toString()}`)
      const datasets: Dataset[] = response.data

      // Filter by strategy_id and find previous dataset
      const filtered = datasets.filter(
        (d) =>
          d.id !== currentDatasetId &&
          d.status === 'ready' &&
          (d.strategy_id === strategyId || (d.strategy_id === null && strategyId === null))
      )

      // Sort by created_at descending and get the first one (most recent before current)
      const sorted = filtered.sort((a, b) => {
        const dateA = new Date(a.created_at).getTime()
        const dateB = new Date(b.created_at).getTime()
        return dateB - dateA
      })

      return sorted.length > 0 ? sorted[0] : null
    },
    enabled: !!(symbol && strategyId && currentDatasetId),
  })
}

