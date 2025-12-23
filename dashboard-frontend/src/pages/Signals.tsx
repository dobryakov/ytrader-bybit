import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSignals } from '@/hooks/useSignals'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'

export default function Signals() {
  const navigate = useNavigate()
  const [page, setPage] = useState(1)
  const [filters, setFilters] = useState<{ asset?: string; signal_type?: string }>({})
  const { data, isLoading } = useSignals({ ...filters, page, page_size: 20 })
  
  const handleViewOrders = (signalId: string) => {
    navigate(`/orders?signal_id=${signalId}`)
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Сигналы</h2>
        <p className="text-muted-foreground">История торговых сигналов</p>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <input
          type="text"
          placeholder="Asset"
          className="px-3 py-2 border rounded-md"
          value={filters.asset || ''}
          onChange={(e) => {
            setFilters({ ...filters, asset: e.target.value || undefined })
            setPage(1)
          }}
        />
        <select
          className="px-3 py-2 border rounded-md"
          value={filters.signal_type || ''}
          onChange={(e) => {
            setFilters({ ...filters, signal_type: e.target.value || undefined })
            setPage(1)
          }}
        >
          <option value="">All Types</option>
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
      </div>

      {/* Table */}
      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : (
        <>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Signal ID</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Asset</TableHead>
                <TableHead>Amount</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Model Prediction</TableHead>
                <TableHead>Actual Movement</TableHead>
                <TableHead>Strategy</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Orders</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.signals.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={11} className="text-center text-muted-foreground">
                    Нет сигналов
                  </TableCell>
                </TableRow>
              ) : (
                data?.signals.map((signal) => (
                  <TableRow key={signal.signal_id}>
                    <TableCell className="font-mono text-xs">{signal.signal_id.slice(0, 8)}...</TableCell>
                    <TableCell>
                      <Badge variant={signal.signal_type === 'buy' ? 'default' : 'destructive'}>
                        {signal.signal_type.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>{signal.asset}</TableCell>
                    <TableCell>{parseFloat(signal.amount).toFixed(2)} USDT</TableCell>
                    <TableCell>
                      {signal.confidence ? (signal.confidence * 100).toFixed(2) + '%' : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {signal.model_prediction ? (
                        <Badge variant={signal.model_prediction === 'UP' ? 'default' : 'destructive'}>
                          {signal.model_prediction}
                        </Badge>
                      ) : (
                        <span className="text-muted-foreground">N/A</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {signal.actual_movement && signal.actual_movement.price_from && signal.actual_movement.price_to ? (
                        <div className="flex flex-col gap-1">
                          <div className="text-sm">
                            <span className="text-muted-foreground">От:</span>{' '}
                            <span className="font-mono">{signal.actual_movement.price_from.toFixed(2)}</span>
                          </div>
                          <div className="text-sm">
                            <span className="text-muted-foreground">До:</span>{' '}
                            <span className="font-mono">{signal.actual_movement.price_to.toFixed(2)}</span>
                            {signal.actual_movement.direction && (
                              <Badge 
                                variant={signal.actual_movement.direction === 'UP' ? 'default' : 'destructive'}
                                className="ml-2"
                              >
                                {signal.actual_movement.direction}
                              </Badge>
                            )}
                          </div>
                          {signal.actual_movement.return_value !== null && (
                            <div className="text-xs text-muted-foreground">
                              {(signal.actual_movement.return_value * 100).toFixed(4)}%
                            </div>
                          )}
                        </div>
                      ) : (
                        <span className="text-muted-foreground">N/A</span>
                      )}
                    </TableCell>
                    <TableCell>{signal.strategy_id || 'N/A'}</TableCell>
                    <TableCell>{signal.model_version || 'N/A'}</TableCell>
                    <TableCell>{format(parseISO(signal.timestamp), 'dd.MM.yyyy HH:mm:ss')}</TableCell>
                    <TableCell>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleViewOrders(signal.signal_id)}
                      >
                        Ордера
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>

          {/* Pagination */}
          {data && data.pagination.total_pages > 1 && (
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Страница {data.pagination.page} из {data.pagination.total_pages} ({data.pagination.total} всего)
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                >
                  Предыдущая
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.min(data.pagination.total_pages, p + 1))}
                  disabled={page === data.pagination.total_pages}
                >
                  Следующая
                </Button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

