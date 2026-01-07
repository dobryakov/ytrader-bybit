import { useQuery } from '@tanstack/react-query'
import api from '@/lib/api'

export interface CandlestickData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface CandlesResponse {
  symbol: string
  start_time: string
  end_time: string
  interval: number
  klines: CandlestickData[]
  count: number
}

/**
 * Hook to fetch candlestick data from Feature Service API
 * @param symbol Trading pair symbol (e.g., BTCUSDT)
 * @param startTime Start timestamp (ISO string or Date)
 * @param endTime End timestamp (ISO string or Date)
 * @param interval Kline interval in minutes (default: 1)
 */
export function useCandles(
  symbol: string,
  startTime: string | Date,
  endTime: string | Date,
  interval: number = 1
) {
  const startTimeStr = startTime instanceof Date ? startTime.toISOString() : startTime
  const endTimeStr = endTime instanceof Date ? endTime.toISOString() : endTime

  return useQuery<CandlestickData[]>({
    queryKey: ['candles', symbol, interval, startTimeStr, endTimeStr],
    queryFn: async () => {
      const params = new URLSearchParams()
      params.append('symbol', symbol)
      params.append('start_time', startTimeStr)
      params.append('end_time', endTimeStr)
      params.append('interval', interval.toString())

      // Request goes through Vite proxy (/api) → dashboard-api → feature-service
      const response = await api.get(`/v1/historical/klines?${params.toString()}`)
      
      // Feature Service returns: { symbol, start_time, end_time, interval, klines: [...], count }
      if (response.data?.klines && Array.isArray(response.data.klines)) {
        return response.data.klines
      }
      
      return []
    },
    enabled: !!symbol && !!startTimeStr && !!endTimeStr,
    staleTime: 60000, // 1 minute
  })
}

