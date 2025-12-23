import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export interface OverviewMetrics {
  total_unrealized_pnl: string
  total_realized_pnl: string
  open_positions_count: number
  total_positions_count: number
  balance: string | null
  available_balance: string | null
}

export interface PortfolioMetrics {
  portfolio: Array<{
    asset: string
    unrealized_pnl: string
    realized_pnl: string
    exposure: string
  }>
  count: number
}

export function useOverviewMetrics() {
  return useQuery<OverviewMetrics>({
    queryKey: ['metrics', 'overview'],
    queryFn: async () => {
      const response = await api.get('/v1/metrics/overview')
      return response.data
    },
    refetchInterval: 10000,
  })
}

export function usePortfolioMetrics() {
  return useQuery<PortfolioMetrics>({
    queryKey: ['metrics', 'portfolio'],
    queryFn: async () => {
      const response = await api.get('/v1/metrics/portfolio')
      return response.data
    },
    refetchInterval: 10000,
  })
}

export interface BalanceByAsset {
  coin: string
  available_balance: string  // Available for trading
  wallet_balance: string    // Total balance
  frozen: string           // Frozen balance
  last_updated: string | null
}

export interface BalancesResponse {
  balances: BalanceByAsset[]
  count: number
}

export function useBalances() {
  return useQuery<BalancesResponse>({
    queryKey: ['metrics', 'balances'],
    queryFn: async () => {
      const response = await api.get('/v1/metrics/balances')
      return response.data
    },
    refetchInterval: 10000,
  })
}

