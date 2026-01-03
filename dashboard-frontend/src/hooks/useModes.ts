import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '@/lib/api'

export interface Mode {
  id: string
  name: string
  asset: string
  strategy_id: string
  feature_registry_version: string
  target_registry_version: string
  train_duration_days: number
  validation_duration_days: number
  test_duration_days: number
  description: string | null
  is_active: boolean
  created_at: string
  updated_at: string
  created_by: string | null
}

export interface RebuildDatasetResponse {
  mode_id: string
  dataset_id: string
  computed_periods: {
    train_period_start: string
    train_period_end: string
    validation_period_start: string
    validation_period_end: string
    test_period_start: string
    test_period_end: string
  }
  message: string
}

export function useModes(filters?: {
  asset?: string
  strategy_id?: string
  is_active?: boolean
  limit?: number
}) {
  return useQuery<Mode[]>({
    queryKey: ['modes', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters?.asset) params.append('asset', filters.asset)
      if (filters?.strategy_id) params.append('strategy_id', filters.strategy_id)
      if (filters?.is_active !== undefined) params.append('is_active', filters.is_active.toString())
      if (filters?.limit) params.append('limit', filters.limit.toString())
      else params.append('limit', '100')

      const response = await api.get(`/v1/modes?${params.toString()}`)
      return response.data
    },
  })
}

export function useMode(modeId: string) {
  return useQuery<Mode>({
    queryKey: ['mode', modeId],
    queryFn: async () => {
      const response = await api.get(`/v1/modes/${modeId}`)
      return response.data
    },
    enabled: !!modeId,
  })
}

export function useCreateMode() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (mode: Omit<Mode, 'id' | 'created_at' | 'updated_at' | 'created_by'>) => {
      const response = await api.post('/v1/modes', mode)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['modes'] })
    },
  })
}

export function useUpdateMode() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ id, ...mode }: Partial<Mode> & { id: string }) => {
      const response = await api.put(`/v1/modes/${id}`, mode)
      return response.data
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['modes'] })
      queryClient.invalidateQueries({ queryKey: ['mode', variables.id] })
    },
  })
}

export function useDeleteMode() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (modeId: string) => {
      await api.delete(`/v1/modes/${modeId}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['modes'] })
    },
  })
}

export function useRebuildDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (modeId: string) => {
      const response = await api.post(`/v1/modes/${modeId}/rebuild-dataset`, {})
      return response.data as RebuildDatasetResponse
    },
    onSuccess: () => {
      // Invalidate datasets to show the new one
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
    },
  })
}

