import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from 'recharts'
import { format } from 'date-fns'

export interface CandlestickData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface CandlestickChartProps {
  data: CandlestickData[]
  entryPrice?: number
  exitPrice?: number
  height?: number
}

export function CandlestickChart({ data, entryPrice, exitPrice, height = 400 }: CandlestickChartProps) {
  const formatTime = (timestamp: string) => {
    try {
      return format(new Date(timestamp), 'HH:mm')
    } catch {
      return timestamp
    }
  }

  const formatTooltipTime = (timestamp: string) => {
    try {
      return format(new Date(timestamp), 'dd.MM.yyyy HH:mm')
    } catch {
      return timestamp
    }
  }

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
          <p className="font-semibold mb-2">{formatTooltipTime(data.timestamp)}</p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Open:</span>
              <span className="font-medium">${data.open.toFixed(2)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Close:</span>
              <span className="font-medium">${data.close.toFixed(2)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">High:</span>
              <span className="font-medium text-green-600">${data.high.toFixed(2)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Low:</span>
              <span className="font-medium text-red-600">${data.low.toFixed(2)}</span>
            </div>
            {data.volume && (
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Volume:</span>
                <span className="font-medium">{data.volume.toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>
      )
    }
    return null
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        Нет данных для отображения
      </div>
    )
  }

  // Check if all values are the same (data quality issue)
  const hasVariation = data.some(item => 
    item.open !== item.close || 
    item.high !== item.low || 
    item.open !== item.high
  )
  
  if (!hasVariation && data.length > 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground space-y-2">
        <div className="text-lg font-semibold">Проблема с данными</div>
        <div className="text-sm text-center max-w-md">
          Все значения (open, high, low, close) одинаковые в данных.
          <br />
          Это указывает на проблему в источнике данных или при их сохранении.
        </div>
        <div className="text-xs mt-2">
          Пример: open={data[0]?.open?.toFixed(2)}, high={data[0]?.high?.toFixed(2)}, 
          low={data[0]?.low?.toFixed(2)}, close={data[0]?.close?.toFixed(2)}
        </div>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" />
        <XAxis
          dataKey="timestamp"
          tickFormatter={formatTime}
          tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
          angle={-45}
          textAnchor="end"
          height={60}
        />
        <YAxis
          domain={['auto', 'auto']}
          tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
          tickFormatter={(value) => `$${value.toFixed(0)}`}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        
        {/* Open price line */}
        <Line
          type="monotone"
          dataKey="open"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          name="Open"
          legendType="line"
        />
        
        {/* Close price line */}
        <Line
          type="monotone"
          dataKey="close"
          stroke="#10b981"
          strokeWidth={2}
          dot={false}
          name="Close"
          legendType="line"
        />
        
        {/* Entry price reference line */}
        {entryPrice && (
          <ReferenceLine
            y={entryPrice}
            stroke="#3b82f6"
            strokeWidth={2}
            strokeDasharray="5 5"
            label={{ 
              value: `Entry: $${entryPrice.toFixed(2)}`, 
              position: 'right', 
              fill: '#3b82f6',
              fontSize: 12,
              fontWeight: 'bold'
            }}
            isFront={true}
          />
        )}
        
        {/* Exit price reference line */}
        {exitPrice && (
          <ReferenceLine
            y={exitPrice}
            stroke="#8b5cf6"
            strokeWidth={2}
            strokeDasharray="5 5"
            label={{ 
              value: `Exit: $${exitPrice.toFixed(2)}`, 
              position: 'right', 
              fill: '#8b5cf6',
              fontSize: 12,
              fontWeight: 'bold'
            }}
            isFront={true}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  )
}
