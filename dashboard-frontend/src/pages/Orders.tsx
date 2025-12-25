import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { useOrders } from '@/hooks/useOrders'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'

export default function Orders() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const signalIdFromUrl = searchParams.get('signal_id') || undefined
  const positionIdFromUrl = searchParams.get('position_id') || undefined
  
  const [page, setPage] = useState(1)
  const [filters, setFilters] = useState<{ asset?: string; status?: string; signal_id?: string; position_id?: string }>({})
  
  // Sync filters from URL
  useEffect(() => {
    if (signalIdFromUrl) {
      setFilters(prev => ({ ...prev, signal_id: signalIdFromUrl }))
    }
    if (positionIdFromUrl) {
      setFilters(prev => ({ ...prev, position_id: positionIdFromUrl }))
    }
  }, [signalIdFromUrl, positionIdFromUrl])
  
  const { data, isLoading } = useOrders({ ...filters, signal_id: signalIdFromUrl, position_id: positionIdFromUrl, page, page_size: 20 })
  
  const handleViewPosition = (positionId: string | null, asset: string) => {
    if (positionId) {
      navigate(`/positions?position_id=${positionId}`)
    } else {
      // Fallback to asset filter if position_id is not available
      navigate(`/positions?asset=${asset}`)
    }
  }

  const formatCurrency = (value: string | null) => {
    if (!value) return 'N/A'
    const num = parseFloat(value)
    return new Intl.NumberFormat('ru-RU', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(num)
  }

  const getStatusBadgeVariant = (status: string) => {
    if (status === 'filled') return 'default'
    if (status === 'pending') return 'secondary'
    if (status === 'cancelled') return 'outline'
    return 'outline'
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Ордера</h2>
        <p className="text-muted-foreground">История торговых ордеров</p>
      </div>

      {/* Filters */}
      <div className="flex gap-4 items-center">
        {positionIdFromUrl && (
          <div className="px-3 py-2 bg-muted rounded-md text-sm">
            Фильтр по Position ID: <span className="font-mono">{positionIdFromUrl.slice(0, 8)}...</span>
            <Button
              variant="ghost"
              size="sm"
              className="ml-2 h-6 px-2"
              onClick={() => {
                const newParams = new URLSearchParams(searchParams)
                newParams.delete('position_id')
                setSearchParams(newParams)
                setFilters(prev => {
                  const { position_id, ...rest } = prev
                  return rest
                })
              }}
            >
              ✕
            </Button>
          </div>
        )}
        {signalIdFromUrl && !positionIdFromUrl && (
          <div className="px-3 py-2 bg-muted rounded-md text-sm">
            Фильтр по Signal ID: <span className="font-mono">{signalIdFromUrl.slice(0, 8)}...</span>
            <Button
              variant="ghost"
              size="sm"
              className="ml-2 h-6 px-2"
              onClick={() => {
                const newParams = new URLSearchParams(searchParams)
                newParams.delete('signal_id')
                setSearchParams(newParams)
                setFilters(prev => {
                  const { signal_id, ...rest } = prev
                  return rest
                })
              }}
            >
              ✕
            </Button>
          </div>
        )}
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
          value={filters.status || ''}
          onChange={(e) => {
            setFilters({ ...filters, status: e.target.value || undefined })
            setPage(1)
          }}
        >
          <option value="">All Statuses</option>
          <option value="pending">Pending</option>
          <option value="filled">Filled</option>
          <option value="cancelled">Cancelled</option>
          <option value="rejected">Rejected</option>
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
                <TableHead>Order ID</TableHead>
                <TableHead>Asset</TableHead>
                <TableHead>Side</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Quantity</TableHead>
                <TableHead>Price</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Filled</TableHead>
                <TableHead>Avg Price</TableHead>
                <TableHead>Created At</TableHead>
                <TableHead>Position</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.orders.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={11} className="text-center text-muted-foreground">
                    Нет ордеров
                  </TableCell>
                </TableRow>
              ) : (
                data?.orders.map((order) => (
                  <TableRow key={order.id}>
                    <TableCell className="font-mono text-xs">{order.order_id}</TableCell>
                    <TableCell>{order.asset}</TableCell>
                    <TableCell>
                      <Badge variant={order.side === 'Buy' ? 'default' : 'destructive'}>
                        {order.side}
                      </Badge>
                    </TableCell>
                    <TableCell>{order.order_type}</TableCell>
                    <TableCell>{parseFloat(order.quantity).toFixed(8)}</TableCell>
                    <TableCell>{formatCurrency(order.price)}</TableCell>
                    <TableCell>
                      <div className="flex flex-col gap-1">
                        <Badge variant={getStatusBadgeVariant(order.status)}>
                          {order.status}
                        </Badge>
                        {(order.status === 'rejected' || order.status === 'cancelled') && order.rejection_reason && (
                          <span className="text-xs text-muted-foreground">
                            {order.rejection_reason}
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>{parseFloat(order.filled_quantity).toFixed(8)}</TableCell>
                    <TableCell>{formatCurrency(order.average_price)}</TableCell>
                    <TableCell>{format(parseISO(order.created_at), 'dd.MM.yyyy HH:mm:ss')}</TableCell>
                    <TableCell>
                      {order.position_id ? (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleViewPosition(order.position_id, order.asset)}
                        >
                          Позиция
                        </Button>
                      ) : (
                        <span className="text-muted-foreground text-sm">N/A</span>
                      )}
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

