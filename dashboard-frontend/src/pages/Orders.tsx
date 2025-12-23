import { useState } from 'react'
import { useOrders } from '@/hooks/useOrders'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'

export default function Orders() {
  const [page, setPage] = useState(1)
  const [filters, setFilters] = useState<{ asset?: string; status?: string }>({})
  const { data, isLoading } = useOrders({ ...filters, page, page_size: 20 })

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
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.orders.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={10} className="text-center text-muted-foreground">
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
                      <Badge variant={getStatusBadgeVariant(order.status)}>
                        {order.status}
                      </Badge>
                    </TableCell>
                    <TableCell>{parseFloat(order.filled_quantity).toFixed(8)}</TableCell>
                    <TableCell>{formatCurrency(order.average_price)}</TableCell>
                    <TableCell>{format(parseISO(order.created_at), 'dd.MM.yyyy HH:mm:ss')}</TableCell>
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

