import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricCardProps {
  title: string
  value: string
  description?: string
  className?: string
  currentValue?: number | null // Raw numeric value for comparison
  previousValue?: number | null
  isHigherBetter?: boolean // true if higher value is better (e.g., accuracy, R²), false if lower is better (e.g., MAE, RMSE)
}

export function MetricCard({ 
  title, 
  value, 
  description, 
  className,
  currentValue,
  previousValue,
  isHigherBetter = true,
}: MetricCardProps) {
  const getComparisonIcon = (current: number | null, previous: number | null) => {
    if (current === null || previous === null || current === undefined || previous === undefined) {
      return null
    }

    const difference = current - previous
    
    // For metrics where higher is better (accuracy, R², etc.)
    if (isHigherBetter) {
      if (difference > 0) return <TrendingUp className="h-4 w-4 text-green-500" />
      if (difference < 0) return <TrendingDown className="h-4 w-4 text-red-500" />
    } else {
      // For metrics where lower is better (MAE, RMSE, etc.)
      if (difference < 0) return <TrendingDown className="h-4 w-4 text-green-500" />
      if (difference > 0) return <TrendingUp className="h-4 w-4 text-red-500" />
    }
    
    return <Minus className="h-4 w-4 text-gray-400" />
  }

  const getComparisonText = (current: number | null, previous: number | null) => {
    if (current === null || previous === null || current === undefined || previous === undefined) {
      return null
    }

    const difference = current - previous
    const percentChange = previous !== 0 ? ((difference / Math.abs(previous)) * 100) : 0
    
    if (Math.abs(difference) < 1e-10) {
      return 'Без изменений'
    }

    const absPercent = Math.abs(percentChange).toFixed(2)
    const sign = difference > 0 ? '+' : ''
    
    // Determine format based on current value format
    const isPercentageFormat = value.includes('%')
    const decimalPlaces = isPercentageFormat ? 2 : 4
    
    // Format previous value to match current value format
    const formatPrevious = (val: number) => {
      if (isPercentageFormat) {
        // Format as percentage (assuming value is in 0-1 range)
        return (val * 100).toFixed(decimalPlaces) + '%'
      } else {
        // Format as decimal with appropriate precision
        // Try to match the precision of current value
        const currentDecimals = value.match(/\.(\d+)/)?.[1]?.length || decimalPlaces
        return val.toFixed(currentDecimals)
      }
    }
    
    // Format difference to match value format
    const formatDifference = (diff: number) => {
      if (isPercentageFormat) {
        return (diff * 100).toFixed(decimalPlaces)
      } else {
        const currentDecimals = value.match(/\.(\d+)/)?.[1]?.length || decimalPlaces
        return diff.toFixed(currentDecimals)
      }
    }
    
    return `Предыдущее: ${formatPrevious(previous)} (${sign}${formatDifference(difference)}, ${sign}${absPercent}%)`
  }

  const currentNumeric = currentValue

  return (
    <Card className={cn(className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {currentNumeric !== null && previousValue !== null && getComparisonIcon(currentNumeric, previousValue)}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {previousValue !== null && previousValue !== undefined && currentNumeric !== null && (
          <p className="text-xs text-muted-foreground mt-1">
            {getComparisonText(currentNumeric, previousValue)}
          </p>
        )}
        {description && !previousValue && <p className="text-xs text-muted-foreground mt-1">{description}</p>}
      </CardContent>
    </Card>
  )
}

