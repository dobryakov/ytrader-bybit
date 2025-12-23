import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { usePositions } from '@/hooks/usePositions'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'

export default function Positions() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const assetFromUrl = searchParams.get('asset') || undefined
  const positionIdFromUrl = searchParams.get('position_id') || undefined
  
  const [filters, setFilters] = useState<{ asset?: string; mode?: string; position_id?: string }>({})
  
  // Sync filters from URL
  useEffect(() => {
    if (assetFromUrl) {
      setFilters(prev => ({ ...prev, asset: assetFromUrl }))
    }
    if (positionIdFromUrl) {
      setFilters(prev => ({ ...prev, position_id: positionIdFromUrl }))
    }
  }, [assetFromUrl, positionIdFromUrl])
  
  const { data, isLoading } = usePositions({ ...filters, asset: assetFromUrl, position_id: positionIdFromUrl })
  
  const handleViewOrders = (positionId: string) => {
    navigate(`/orders?position_id=${positionId}`)
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

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Позиции</h2>
        <p className="text-muted-foreground">Текущие и исторические торговые позиции</p>
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
                setSearchParams({})
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
        {assetFromUrl && !positionIdFromUrl && (
          <div className="px-3 py-2 bg-muted rounded-md text-sm">
            Фильтр по Asset: <span className="font-mono">{assetFromUrl}</span>
            <Button
              variant="ghost"
              size="sm"
              className="ml-2 h-6 px-2"
              onClick={() => {
                setSearchParams({})
                setFilters(prev => {
                  const { asset, ...rest } = prev
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
              <TableHead>Position ID</TableHead>
              <TableHead>Asset</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Entry Price</TableHead>
              <TableHead>Current Price</TableHead>
              <TableHead>Unrealized PnL</TableHead>
              <TableHead>Realized PnL</TableHead>
              <TableHead>Mode</TableHead>
              <TableHead>Last Updated</TableHead>
              <TableHead>Orders</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data?.positions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={11} className="text-center text-muted-foreground">
                  Нет позиций
                </TableCell>
              </TableRow>
            ) : (
              data?.positions.map((position) => {
                // Position is open if closed_at is null AND size is not zero
                const isOpen = position.closed_at === null && parseFloat(position.size || '0') !== 0
                return (
                  <TableRow key={position.id}>
                    <TableCell className="font-mono text-xs">{position.id.slice(0, 8)}...</TableCell>
                    <TableCell className="font-medium">{position.asset}</TableCell>
                    <TableCell>
                      <Badge variant={isOpen ? 'default' : 'secondary'}>
                        {isOpen ? 'Открыта' : 'Закрыта'}
                      </Badge>
                    </TableCell>
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
                    <TableCell>{position.last_updated ? format(parseISO(position.last_updated), 'dd.MM.yyyy HH:mm:ss') : 'N/A'}</TableCell>
                    <TableCell>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleViewOrders(position.id)}
                      >
                        Ордера
                      </Button>
                    </TableCell>
                  </TableRow>
                )
              })
            )}
          </TableBody>
        </Table>
      )}
    </div>
  )
}

