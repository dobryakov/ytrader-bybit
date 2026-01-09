import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
import { useMemo, useState, useEffect } from 'react'
import { usePositionById, usePositionOrdersById, PositionOrder } from '@/hooks/usePositions'
import { useCandles } from '@/hooks/useCandles'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { MetricCard } from '@/components/metrics/MetricCard'
import { CandlestickChart } from '@/components/charts/CandlestickChart'
import { format } from 'date-fns'
import { parseISO, subMinutes } from 'date-fns'
import { ArrowLeft } from 'lucide-react'

export default function PositionDetail() {
  const { positionId } = useParams<{ positionId: string }>()
  const navigate = useNavigate()
  
  const { data: position, isLoading, error } = usePositionById(positionId || '')
  
  // Get position orders using position ID
  const { data: positionOrders, isLoading: ordersLoading } = usePositionOrdersById(
    positionId || ''
  )
  
  // Calculate time range for candlestick chart
  // Show 5 minutes before position opening to current time or closing time
  // Limit to maximum 7 days or 1000 candles to prevent browser freeze
  const MAX_CHART_DAYS = 7
  const MAX_CANDLES = 1000
  
  // Determine position opening time for chart:
  // 
  // Используем created_at как время открытия позиции (каждая запись создается один раз).
  // График показывает данные с 5 минут ДО времени открытия позиции.
  let positionOpenTime: string | null = null
  let positionOpenTimeSource: string = 'unknown'
  
  if (position) {
    // Используем created_at как время открытия позиции
    if (position.created_at) {
      positionOpenTime = position.created_at
      positionOpenTimeSource = 'created_at (время создания позиции)'
    } else if (position.closed_at === null) {
      // Открытая позиция без created_at: используем last_updated как fallback
      positionOpenTime = position.last_updated || null
      positionOpenTimeSource = position.last_updated ? 'last_updated (fallback, created_at отсутствует)' : 'unknown'
    } else {
      // Закрытая позиция без created_at: оцениваем 1 час до закрытия
      const closedAt = parseISO(position.closed_at)
      positionOpenTime = new Date(closedAt.getTime() - 60 * 60 * 1000).toISOString()
      positionOpenTimeSource = 'оценка (1 час до закрытия, created_at отсутствует)'
    }
  }
  
  // График начинается за 5 минут до времени открытия позиции
  const chartStartTime = useMemo(() => {
    return positionOpenTime 
      ? subMinutes(parseISO(positionOpenTime), 5)
      : null
  }, [positionOpenTime])
  
  // Конец графика: closed_at или текущее время (в зависимости от того, что меньше)
  // Для открытой позиции обновляем endTime каждую минуту, для закрытой - фиксируем
  const [chartEndTime, setChartEndTime] = useState<Date>(() => {
    const now = new Date()
    if (position?.closed_at) {
      const closedAt = parseISO(position.closed_at)
      return closedAt < now ? closedAt : now
    }
    return now
  })
  
  // Обновляем chartEndTime для открытых позиций каждую минуту
  useEffect(() => {
    if (!position?.closed_at && chartStartTime) {
      const updateInterval = setInterval(() => {
        const now = new Date()
        // Ограничиваем максимальный диапазон
        const maxEndTime = new Date(chartStartTime.getTime() + MAX_CHART_DAYS * 24 * 60 * 60 * 1000)
        setChartEndTime(now < maxEndTime ? now : maxEndTime)
      }, 60000) // Обновляем каждую минуту
      
      return () => clearInterval(updateInterval)
    } else if (position?.closed_at) {
      // Для закрытой позиции фиксируем endTime
      const closedAt = parseISO(position.closed_at)
      setChartEndTime(closedAt)
    }
  }, [position?.closed_at, chartStartTime])
  
  // Мемоизируем финальный endTime с учетом ограничений
  const finalChartEndTime = useMemo(() => {
    if (!chartStartTime) return new Date()
    
    // Limit time range if too large
    const daysDiff = (chartEndTime.getTime() - chartStartTime.getTime()) / (1000 * 60 * 60 * 24)
    if (daysDiff > MAX_CHART_DAYS) {
      return new Date(chartStartTime.getTime() + MAX_CHART_DAYS * 24 * 60 * 60 * 1000)
    }
    
    return chartEndTime
  }, [chartStartTime, chartEndTime])
  
  // Calculate appropriate interval based on time range
  const timeRangeMinutes = chartStartTime && finalChartEndTime 
    ? (finalChartEndTime.getTime() - chartStartTime.getTime()) / (1000 * 60)
    : 0
  
  // Use larger interval for longer periods to limit data size
  const interval = useMemo(() => {
    if (timeRangeMinutes <= MAX_CANDLES) return 1
    
    // Calculate interval to get approximately MAX_CANDLES candles
    const calculatedInterval = Math.ceil(timeRangeMinutes / MAX_CANDLES)
    // Round to common intervals: 1, 3, 5, 15, 30, 60
    if (calculatedInterval <= 3) return 3
    else if (calculatedInterval <= 5) return 5
    else if (calculatedInterval <= 15) return 15
    else if (calculatedInterval <= 30) return 30
    else return 60
  }, [timeRangeMinutes])
  
  const { data: candles, isLoading: candlesLoading } = useCandles(
    position?.asset || '',
    chartStartTime || new Date(),
    finalChartEndTime,
    interval
  )

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  if (error || !position) {
    return (
      <div className="space-y-6">
        <div>
          <Button variant="outline" onClick={() => navigate('/positions')} className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Назад к позициям
          </Button>
          <div className="text-center text-muted-foreground py-8">
            {error ? 'Ошибка загрузки данных позиции' : 'Позиция не найдена'}
            {positionId && <div className="mt-2">Position ID: {positionId}</div>}
          </div>
        </div>
      </div>
    )
  }

  const formatCurrency = (value: string | null) => {
    if (!value) return 'N/A'
    const num = parseFloat(value)
    return new Intl.NumberFormat('ru-RU', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 8,
    }).format(num)
  }

  const formatPercent = (value: string | null) => {
    if (!value) return 'N/A'
    const num = parseFloat(value)
    return `${num >= 0 ? '+' : ''}${num.toFixed(2)}%`
  }

  // Group orders by relationship type
  const groupedOrders = positionOrders?.orders.reduce((acc, order) => {
    const type = order.relationship_type
    if (!acc[type]) {
      acc[type] = []
    }
    acc[type].push(order)
    return acc
  }, {} as Record<string, PositionOrder[]>) || {}

  const relationshipTypeLabels: Record<string, string> = {
    opened: 'Открыли позицию',
    increased: 'Увеличили позицию',
    decreased: 'Уменьшили позицию',
    closed: 'Закрыли позицию',
    reversed: 'Развернули позицию',
  }

  const relationshipTypeColors: Record<string, string> = {
    opened: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    increased: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    decreased: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    closed: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    reversed: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
  }

  const isOpen = position.closed_at === null && parseFloat(position.size || '0') !== 0
  
  // Determine position sign based on relationship_type and side of orders
  // For "opened" orders, use the sign of size_delta to determine initial position direction
  // For other orders, use relationship_type to determine if it's an increase or decrease
  const getPositionSignAtOrderTime = (order: PositionOrder): 'long' | 'short' => {
    // If this is the opening order, use the sign of size_delta
    if (order.relationship_type === 'opened') {
      return parseFloat(order.size_delta) >= 0 ? 'long' : 'short'
    }
    
    // For other orders, determine based on relationship_type and side
    // If relationship_type is "increased", the position is growing in its current direction
    // If relationship_type is "decreased", the position is shrinking
    // We need to track the cumulative position size, but for simplicity,
    // we'll use the relationship_type as a hint
    
    // If we have orders, find the opening order to determine initial direction
    if (positionOrders?.orders) {
      const openingOrder = positionOrders.orders.find(o => o.relationship_type === 'opened')
      if (openingOrder) {
        const initialSign = parseFloat(openingOrder.size_delta) >= 0 ? 'long' : 'short'
        // If relationship_type is "increased", position is growing in same direction
        // If "decreased", position is shrinking
        // If "reversed", position changed direction
        if (order.relationship_type === 'reversed') {
          // After reversal, position direction is opposite of initial
          return initialSign === 'long' ? 'short' : 'long'
        }
        return initialSign
      }
    }
    
    // Fallback: use current position size
    return parseFloat(position.size || '0') >= 0 ? 'long' : 'short'
  }
  
  // Helper function to format size_delta with correct sign and color
  // Use relationship_type as the primary indicator:
  // - "opened" and "increased" = increase in position size (always green with +)
  // - "decreased" and "closed" = decrease in position size (always red with -)
  // - "reversed" = position reversed direction (purple)
  const formatSizeDelta = (order: PositionOrder): { display: string; color: string } => {
    const delta = parseFloat(order.size_delta)
    
    // Use relationship_type as the primary indicator
    if (order.relationship_type === 'opened' || order.relationship_type === 'increased') {
      // This is an increase in position size - always show as positive (green)
      return {
        display: '+' + Math.abs(delta).toFixed(8),
        color: 'text-green-600'
      }
    } else if (order.relationship_type === 'decreased' || order.relationship_type === 'closed') {
      // This is a decrease in position size - always show as negative (red)
      return {
        display: '-' + Math.abs(delta).toFixed(8),
        color: 'text-red-600'
      }
    } else if (order.relationship_type === 'reversed') {
      // Position reversed direction - show actual delta with purple color
      return {
        display: (delta >= 0 ? '+' : '') + delta.toFixed(8),
        color: 'text-purple-600'
      }
    }
    
    // Fallback: use raw delta value
    return {
      display: (delta >= 0 ? '+' : '') + delta.toFixed(8),
      color: delta >= 0 ? 'text-green-600' : 'text-red-600'
    }
  }
  
  // Entry price: use average_entry_price from position, or calculate from orders
  let entryPrice: number | undefined = undefined
  if (position.average_entry_price) {
    entryPrice = parseFloat(position.average_entry_price)
  } else if (positionOrders?.orders) {
    // Calculate from orders that opened or increased the position
    const entryOrders = positionOrders.orders.filter(
      order => (order.relationship_type === 'opened' || order.relationship_type === 'increased' || order.relationship_type === 'reversed') 
        && order.execution_price
    )
    if (entryOrders.length > 0) {
      // Calculate weighted average entry price
      let totalQuantity = 0
      let totalValue = 0
      entryOrders.forEach(order => {
        const quantity = Math.abs(parseFloat(order.filled_quantity || order.size_delta || '0'))
        const price = parseFloat(order.execution_price)
        if (quantity > 0 && price > 0) {
          totalQuantity += quantity
          totalValue += quantity * price
        }
      })
      if (totalQuantity > 0) {
        entryPrice = totalValue / totalQuantity
      } else if (entryOrders.length > 0) {
        // Fallback: use first entry order's execution price
        entryPrice = parseFloat(entryOrders[0].execution_price)
      }
    }
  }
  
  // Exit price: for closed positions, get from orders with relationship_type='closed' or 'decreased'
  // Calculate average exit price from all closing orders
  let exitPrice: number | undefined = undefined
  if (!isOpen && positionOrders?.orders && positionOrders.orders.length > 0) {
    // First try orders that explicitly closed the position
    let closingOrders = positionOrders.orders.filter(
      order => order.relationship_type === 'closed' && order.execution_price
    )
    
    // If no explicit closing orders, try decreased orders (position might have been closed by decreasing to zero)
    if (closingOrders.length === 0) {
      closingOrders = positionOrders.orders.filter(
        order => order.relationship_type === 'decreased' && order.execution_price
      )
    }
    
    // If still no closing orders, use the last order (position might have been closed by a single order)
    // Sort all orders by execution time descending to get the most recent order
    if (closingOrders.length === 0) {
      const allOrdersWithPrice = positionOrders.orders
        .filter(order => order.execution_price)
        .sort((a, b) => {
          const timeA = a.po_executed_at || a.executed_at || ''
          const timeB = b.po_executed_at || b.executed_at || ''
          return timeB.localeCompare(timeA)
        })
      if (allOrdersWithPrice.length > 0) {
        closingOrders = [allOrdersWithPrice[0]]
      }
    } else {
      // Sort by execution time descending to get the most recent (likely closing) orders
      closingOrders.sort((a, b) => {
        const timeA = a.po_executed_at || a.executed_at || ''
        const timeB = b.po_executed_at || b.executed_at || ''
        return timeB.localeCompare(timeA)
      })
    }
    
    if (closingOrders.length > 0) {
      // Calculate weighted average exit price
      let totalQuantity = 0
      let totalValue = 0
      closingOrders.forEach(order => {
        const quantity = Math.abs(parseFloat(order.filled_quantity || order.size_delta || '0'))
        const price = parseFloat(order.execution_price)
        if (quantity > 0 && price > 0) {
          totalQuantity += quantity
          totalValue += quantity * price
        }
      })
      if (totalQuantity > 0) {
        exitPrice = totalValue / totalQuantity
      } else if (closingOrders.length > 0) {
        // Fallback: use first closing order's execution price
        exitPrice = parseFloat(closingOrders[0].execution_price)
      }
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Button variant="outline" onClick={() => navigate('/positions')} className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Назад к позициям
          </Button>
          <h2 className="text-3xl font-bold tracking-tight">Детали позиции</h2>
          <p className="text-muted-foreground">
            {position.asset} • {position.mode} • ID: {position.id.slice(0, 8)}...
          </p>
        </div>
        <Badge variant={isOpen ? 'default' : 'secondary'}>
          {isOpen ? 'Открыта' : 'Закрыта'}
        </Badge>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Размер позиции"
          value={parseFloat(position.size || '0').toFixed(8)}
        />
        <MetricCard
          title="Цена входа"
          value={formatCurrency(position.average_entry_price)}
        />
        <MetricCard
          title="Текущая цена"
          value={formatCurrency(position.current_price)}
        />
        <MetricCard
          title="Unrealized PnL"
          value={formatCurrency(position.unrealized_pnl)}
          className={parseFloat(position.unrealized_pnl || '0') >= 0 ? 'text-green-600' : 'text-red-600'}
        />
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Realized PnL"
          value={formatCurrency(position.realized_pnl)}
        />
        {position.unrealized_pnl_pct && (
          <MetricCard
            title="Unrealized PnL %"
            value={formatPercent(position.unrealized_pnl_pct)}
            className={parseFloat(position.unrealized_pnl_pct) >= 0 ? 'text-green-600' : 'text-red-600'}
          />
        )}
        {position.time_held_minutes && (
          <MetricCard
            title="Время удержания"
            value={`${Math.floor(parseFloat(position.time_held_minutes) / 60)}ч ${Math.floor(parseFloat(position.time_held_minutes) % 60)}м`}
          />
        )}
        {position.position_size_norm && (
          <MetricCard
            title="Нормализованный размер"
            value={parseFloat(position.position_size_norm).toFixed(4)}
          />
        )}
      </div>

             {/* Candlestick Chart */}
             <Card>
               <CardHeader>
                 <CardTitle>График движения цены</CardTitle>
                 <CardDescription>
                   График движения ассета от момента за 5 минут до открытия позиции 
                   {isOpen ? ' до текущего времени' : ' до момента закрытия позиции'}
                      {chartStartTime && (
                     <div className="mt-2 text-xs text-muted-foreground">
                      <div>Период графика: {chartStartTime ? format(chartStartTime, 'dd.MM.yyyy HH:mm') : 'N/A'} - {chartEndTime ? format(chartEndTime, 'dd.MM.yyyy HH:mm') : 'N/A'}</div>
                      <div className="mt-1">
                        Время открытия позиции: {positionOpenTime ? (() => {
                          try {
                            return format(parseISO(positionOpenTime), 'dd.MM.yyyy HH:mm:ss');
                          } catch {
                            return positionOpenTime;
                          }
                        })() : 'N/A'}
                        <span className="text-muted-foreground"> ({positionOpenTimeSource})</span>
                      </div>
                      {position && (
                        <div className="mt-1 text-xs opacity-75">
                          {(() => {
                            try {
                              return `created_at: ${format(parseISO(position.created_at), 'dd.MM.yyyy HH:mm:ss')} | `;
                            } catch {
                              return `created_at: ${position.created_at} | `;
                            }
                          })()}
                          {(() => {
                            try {
                              return `last_updated: ${format(parseISO(position.last_updated), 'dd.MM.yyyy HH:mm:ss')}`;
                            } catch {
                              return `last_updated: ${position.last_updated}`;
                            }
                          })()}
                          {position.closed_at && (() => {
                            try {
                              return ` | closed_at: ${format(parseISO(position.closed_at), 'dd.MM.yyyy HH:mm:ss')}`;
                            } catch {
                              return ` | closed_at: ${position.closed_at}`;
                            }
                          })()}
                        </div>
                      )}
                       {(entryPrice || exitPrice) && (
                         <div className="mt-2 text-xs space-y-1">
                           {entryPrice && (
                             <div className="flex items-center gap-2">
                               <span className="inline-block w-3 h-3 rounded border-2 border-blue-500 bg-blue-500/20"></span>
                               <span>Цена входа: <span className="font-semibold text-blue-600">${entryPrice.toFixed(2)}</span></span>
                             </div>
                           )}
                           {exitPrice && (
                             <div className="flex items-center gap-2">
                               <span className="inline-block w-3 h-3 rounded border-2 border-purple-500 bg-purple-500/20"></span>
                               <span>Цена выхода: <span className="font-semibold text-purple-600">${exitPrice.toFixed(2)}</span></span>
                             </div>
                           )}
                         </div>
                       )}
                     </div>
                   )}
                 </CardDescription>
               </CardHeader>
        <CardContent>
          {candlesLoading ? (
            <Skeleton className="h-96 w-full" />
          ) : candles && candles.length > 0 ? (
            <CandlestickChart
              data={candles}
              entryPrice={entryPrice}
              exitPrice={exitPrice}
              height={500}
            />
          ) : (
            <div className="flex items-center justify-center h-64 text-muted-foreground">
              Нет данных для отображения графика
            </div>
          )}
        </CardContent>
      </Card>

      {/* Position Orders */}
      <Card>
        <CardHeader>
          <CardTitle>Ордера позиции</CardTitle>
          <CardDescription>
            Ордера, которые открыли, переоткрыли или изменили эту позицию
          </CardDescription>
        </CardHeader>
        <CardContent>
          {ordersLoading ? (
            <Skeleton className="h-64 w-full" />
          ) : positionOrders && positionOrders.orders.length > 0 ? (
            <div className="space-y-6">
              {Object.entries(groupedOrders).map(([type, orders]) => (
                <div key={type} className="space-y-2">
                  <div className="flex items-center gap-2 mb-3">
                    <Badge className={relationshipTypeColors[type] || ''}>
                      {relationshipTypeLabels[type]} ({orders.length})
                    </Badge>
                  </div>
                  <div className="rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Время</TableHead>
                          <TableHead>Order ID</TableHead>
                          <TableHead>Side</TableHead>
                          <TableHead>Тип</TableHead>
                          <TableHead>Количество</TableHead>
                          <TableHead>Цена</TableHead>
                          <TableHead>Size Delta</TableHead>
                          <TableHead>Статус</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {orders.map((order) => (
                          <TableRow key={order.id}>
                            <TableCell className="font-mono text-xs">
                              {order.po_executed_at
                                ? format(parseISO(order.po_executed_at), 'dd.MM.yyyy HH:mm:ss')
                                : order.executed_at
                                ? format(parseISO(order.executed_at), 'dd.MM.yyyy HH:mm:ss')
                                : 'N/A'}
                            </TableCell>
                            <TableCell className="font-mono text-xs">
                              {order.order_id ? order.order_id.slice(0, 12) + '...' : order.bybit_order_id ? order.bybit_order_id.slice(0, 12) + '...' : 'N/A'}
                            </TableCell>
                            <TableCell>
                              <Badge variant={order.side === 'Buy' ? 'default' : 'secondary'}>
                                {order.side}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <Badge variant="outline">{order.order_type}</Badge>
                            </TableCell>
                            <TableCell>{parseFloat(order.filled_quantity || order.quantity).toFixed(8)}</TableCell>
                            <TableCell>{formatCurrency(order.execution_price || order.average_price || order.price)}</TableCell>
                            <TableCell>
                              {(() => {
                                const formatted = formatSizeDelta(order)
                                return (
                                  <span className={formatted.color}>
                                    {formatted.display}
                                  </span>
                                )
                              })()}
                            </TableCell>
                            <TableCell>
                              <Badge variant={order.status === 'filled' ? 'default' : 'secondary'}>
                                {order.status}
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              Нет ордеров для этой позиции
            </div>
          )}
        </CardContent>
      </Card>

      {/* Position Details Table */}
      <Card>
        <CardHeader>
          <CardTitle>Детальная информация</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableBody>
              <TableRow>
                <TableCell className="font-medium">Position ID</TableCell>
                <TableCell className="font-mono text-xs">{position.id}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Asset</TableCell>
                <TableCell>{position.asset}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Mode</TableCell>
                <TableCell>
                  <Badge variant="outline">{position.mode}</Badge>
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Size</TableCell>
                <TableCell>{parseFloat(position.size || '0').toFixed(8)}</TableCell>
              </TableRow>
              {position.mode === 'hedge' && (
                <>
                  <TableRow>
                    <TableCell className="font-medium">Long Size</TableCell>
                    <TableCell>{position.long_size ? parseFloat(position.long_size).toFixed(8) : 'N/A'}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">Short Size</TableCell>
                    <TableCell>{position.short_size ? parseFloat(position.short_size).toFixed(8) : 'N/A'}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">Long Avg Price</TableCell>
                    <TableCell>{formatCurrency(position.long_avg_price)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">Short Avg Price</TableCell>
                    <TableCell>{formatCurrency(position.short_avg_price)}</TableCell>
                  </TableRow>
                </>
              )}
              <TableRow>
                <TableCell className="font-medium">Average Entry Price</TableCell>
                <TableCell>{formatCurrency(position.average_entry_price)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Current Price</TableCell>
                <TableCell>{formatCurrency(position.current_price)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Unrealized PnL</TableCell>
                <TableCell>
                  <span className={parseFloat(position.unrealized_pnl || '0') >= 0 ? 'text-green-600' : 'text-red-600'}>
                    {formatCurrency(position.unrealized_pnl)}
                  </span>
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Realized PnL</TableCell>
                <TableCell>{formatCurrency(position.realized_pnl)}</TableCell>
              </TableRow>
              {position.unrealized_pnl_pct && (
                <TableRow>
                  <TableCell className="font-medium">Unrealized PnL %</TableCell>
                  <TableCell>
                    <span className={parseFloat(position.unrealized_pnl_pct) >= 0 ? 'text-green-600' : 'text-red-600'}>
                      {formatPercent(position.unrealized_pnl_pct)}
                    </span>
                  </TableCell>
                </TableRow>
              )}
              {position.time_held_minutes && (
                <TableRow>
                  <TableCell className="font-medium">Time Held (minutes)</TableCell>
                  <TableCell>{parseFloat(position.time_held_minutes).toFixed(2)}</TableCell>
                </TableRow>
              )}
              {position.position_size_norm && (
                <TableRow>
                  <TableCell className="font-medium">Position Size Norm</TableCell>
                  <TableCell>{parseFloat(position.position_size_norm).toFixed(4)}</TableCell>
                </TableRow>
              )}
              <TableRow>
                <TableCell className="font-medium">Created At</TableCell>
                <TableCell>
                  {position.created_at ? format(parseISO(position.created_at), 'dd.MM.yyyy HH:mm:ss') : 'N/A'}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Last Updated</TableCell>
                <TableCell>
                  {position.last_updated ? format(parseISO(position.last_updated), 'dd.MM.yyyy HH:mm:ss') : 'N/A'}
                </TableCell>
              </TableRow>
              {position.closed_at && (
                <TableRow>
                  <TableCell className="font-medium">Closed At</TableCell>
                  <TableCell>
                    {format(parseISO(position.closed_at), 'dd.MM.yyyy HH:mm:ss')}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}

