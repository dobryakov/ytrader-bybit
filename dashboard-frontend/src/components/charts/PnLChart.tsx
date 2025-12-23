import { usePnLChart } from '@/hooks/useCharts'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { format, parseISO } from 'date-fns'
import { Skeleton } from '@/components/ui/skeleton'

export function PnLChart() {
  const { data, isLoading } = usePnLChart({
    interval: '1h',
    date_from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
  })

  if (isLoading) {
    return <Skeleton className="h-64 w-full" />
  }

  if (!data || data.data.length === 0) {
    return <div className="text-center text-muted-foreground py-8">Нет данных для отображения</div>
  }

  const chartData = data.data.map((item) => ({
    time: format(parseISO(item.time), 'dd.MM HH:mm'),
    unrealized_pnl: parseFloat(item.unrealized_pnl),
    realized_pnl: parseFloat(item.realized_pnl),
    total: parseFloat(item.unrealized_pnl) + parseFloat(item.realized_pnl),
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="unrealized_pnl" stroke="#8884d8" name="Unrealized PnL" />
        <Line type="monotone" dataKey="realized_pnl" stroke="#82ca9d" name="Realized PnL" />
        <Line type="monotone" dataKey="total" stroke="#ffc658" name="Total PnL" />
      </LineChart>
    </ResponsiveContainer>
  )
}

