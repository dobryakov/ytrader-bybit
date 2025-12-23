import { useState } from 'react'
import { usePositions } from '@/hooks/usePositions'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'

export default function Positions() {
  const [filters, setFilters] = useState<{ asset?: string; mode?: string }>({})
  const { data, isLoading } = usePositions(filters)

  const formatCurrency = (value: string | null) => {
    if (!value) return 'N/A'
    const num = parseFloat(value)
    return new Intl.NumberFormat('ru-RU', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(num)
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Позиции</h2>
        <p className="text-muted-foreground">Текущие и исторические торговые позиции</p>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <input
          type="text"
          placeholder="Asset (e.g., BTCUSDT)"
          className="px-3 py-2 border rounded-md"
          value={filters.asset || ''}
          onChange={(e) => setFilters({ ...filters, asset: e.target.value || undefined })}
        />
        <select
          className="px-3 py-2 border rounded-md"
          value={filters.mode || ''}
          onChange={(e) => setFilters({ ...filters, mode: e.target.value || undefined })}
        >
          <option value="">All Modes</option>
          <option value="one-way">One-way</option>
          <option value="hedge">Hedge</option>
        </select>
      </div>

      {/* Table */}
      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Asset</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Entry Price</TableHead>
              <TableHead>Current Price</TableHead>
              <TableHead>Unrealized PnL</TableHead>
              <TableHead>Realized PnL</TableHead>
              <TableHead>Mode</TableHead>
              <TableHead>Last Updated</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data?.positions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center text-muted-foreground">
                  Нет позиций
                </TableCell>
              </TableRow>
            ) : (
              data?.positions.map((position) => (
                <TableRow key={position.id}>
                  <TableCell className="font-medium">{position.asset}</TableCell>
                  <TableCell>{parseFloat(position.size).toFixed(8)}</TableCell>
                  <TableCell>{formatCurrency(position.average_entry_price)}</TableCell>
                  <TableCell>{formatCurrency(position.current_price)}</TableCell>
                  <TableCell>
                    <span className={parseFloat(position.unrealized_pnl || '0') >= 0 ? 'text-green-600' : 'text-red-600'}>
                      {formatCurrency(position.unrealized_pnl)}
                    </span>
                  </TableCell>
                  <TableCell>{formatCurrency(position.realized_pnl)}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{position.mode}</Badge>
                  </TableCell>
                  <TableCell>{format(parseISO(position.last_updated), 'dd.MM.yyyy HH:mm:ss')}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      )}
    </div>
  )
}

