import { useState } from 'react'
import { useDatasets, type Dataset, type DatasetStatus } from '@/hooks/useDatasets'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'

function getStatusBadgeVariant(status: DatasetStatus): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status) {
    case 'ready':
      return 'default'
    case 'building':
      return 'secondary'
    case 'failed':
      return 'destructive'
    default:
      return 'outline'
  }
}

function formatDate(dateString: string | null): string {
  if (!dateString) return 'N/A'
  try {
    return format(parseISO(dateString), 'dd.MM.yyyy HH:mm:ss')
  } catch {
    return dateString
  }
}

function formatDateShort(dateString: string | null): string {
  if (!dateString) return 'N/A'
  try {
    return format(parseISO(dateString), 'dd.MM.yyyy')
  } catch {
    return dateString
  }
}

export default function Datasets() {
  const [symbolFilter, setSymbolFilter] = useState<string>('')
  const [statusFilter, setStatusFilter] = useState<DatasetStatus | undefined>(undefined)

  const { data: datasets, isLoading } = useDatasets({
    symbol: symbolFilter || undefined,
    status: statusFilter,
    limit: 100,
  })

  const getStatusLabel = (status: DatasetStatus) => {
    switch (status) {
      case 'ready':
        return 'Готов'
      case 'building':
        return 'Сборка'
      case 'failed':
        return 'Ошибка'
      default:
        return status
    }
  }

  const getSplitStrategyLabel = (strategy: string) => {
    switch (strategy) {
      case 'time_based':
        return 'По времени'
      case 'walk_forward':
        return 'Walk-forward'
      default:
        return strategy
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Датасеты</h2>
        <p className="text-muted-foreground">Датасеты для обучения моделей</p>
      </div>

      {/* Filters */}
      <div className="flex gap-4 items-end">
        <div className="flex-1 max-w-xs">
          <label htmlFor="symbol-filter" className="block text-sm font-medium mb-1">
            Символ
          </label>
          <input
            id="symbol-filter"
            type="text"
            value={symbolFilter}
            onChange={(e) => setSymbolFilter(e.target.value)}
            placeholder="BTCUSDT"
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          />
        </div>
        <div className="flex-1 max-w-xs">
          <label htmlFor="status-filter" className="block text-sm font-medium mb-1">
            Статус
          </label>
          <select
            id="status-filter"
            value={statusFilter || ''}
            onChange={(e) => setStatusFilter(e.target.value ? (e.target.value as DatasetStatus) : undefined)}
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          >
            <option value="">Все</option>
            <option value="building">Сборка</option>
            <option value="ready">Готов</option>
            <option value="failed">Ошибка</option>
          </select>
        </div>
      </div>

      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Символ</TableHead>
                <TableHead>Статус</TableHead>
                <TableHead>Стратегия</TableHead>
                <TableHead>Train</TableHead>
                <TableHead>Val</TableHead>
                <TableHead>Test</TableHead>
                <TableHead>Train период</TableHead>
                <TableHead>Val период</TableHead>
                <TableHead>Test период</TableHead>
                <TableHead>Версия фич</TableHead>
                <TableHead>Версия таргета</TableHead>
                <TableHead>Создан</TableHead>
                <TableHead>Завершен</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {!datasets || datasets.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={13} className="text-center text-muted-foreground">
                    Нет датасетов
                  </TableCell>
                </TableRow>
              ) : (
                datasets.map((dataset) => (
                  <TableRow key={dataset.id}>
                    <TableCell className="font-mono text-xs">{dataset.id.slice(0, 8)}...</TableCell>
                    <TableCell className="font-medium">{dataset.symbol}</TableCell>
                    <TableCell>
                      <Badge variant={getStatusBadgeVariant(dataset.status)}>
                        {getStatusLabel(dataset.status)}
                      </Badge>
                    </TableCell>
                    <TableCell>{dataset.strategy_id || 'N/A'}</TableCell>
                    <TableCell>{dataset.train_records?.toLocaleString() || 0}</TableCell>
                    <TableCell>{dataset.validation_records?.toLocaleString() || 0}</TableCell>
                    <TableCell>{dataset.test_records?.toLocaleString() || 0}</TableCell>
                    <TableCell className="text-xs">
                      {dataset.train_period_start && dataset.train_period_end
                        ? `${formatDateShort(dataset.train_period_start)} - ${formatDateShort(dataset.train_period_end)}`
                        : 'N/A'}
                    </TableCell>
                    <TableCell className="text-xs">
                      {dataset.validation_period_start && dataset.validation_period_end
                        ? `${formatDateShort(dataset.validation_period_start)} - ${formatDateShort(dataset.validation_period_end)}`
                        : 'N/A'}
                    </TableCell>
                    <TableCell className="text-xs">
                      {dataset.test_period_start && dataset.test_period_end
                        ? `${formatDateShort(dataset.test_period_start)} - ${formatDateShort(dataset.test_period_end)}`
                        : 'N/A'}
                    </TableCell>
                    <TableCell className="text-xs">{dataset.feature_registry_version}</TableCell>
                    <TableCell className="text-xs">{dataset.target_registry_version || 'N/A'}</TableCell>
                    <TableCell className="text-xs">{formatDate(dataset.created_at)}</TableCell>
                    <TableCell className="text-xs">{formatDate(dataset.completed_at)}</TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}

