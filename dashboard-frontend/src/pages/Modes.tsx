import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  useModes,
  useCreateMode,
  useUpdateMode,
  useDeleteMode,
  useRebuildDataset,
  type Mode,
} from '@/hooks/useModes'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'
import ModeForm from '@/components/modes/ModeForm'

export default function Modes() {
  const [assetFilter, setAssetFilter] = useState<string>('')
  const [strategyFilter, setStrategyFilter] = useState<string>('')
  const [isActiveFilter, setIsActiveFilter] = useState<boolean | undefined>(undefined)
  const [editingMode, setEditingMode] = useState<Mode | null>(null)
  const [isFormOpen, setIsFormOpen] = useState(false)
  const [rebuildingModeId, setRebuildingModeId] = useState<string | null>(null)

  const { data: modes, isLoading } = useModes({
    asset: assetFilter || undefined,
    strategy_id: strategyFilter || undefined,
    is_active: isActiveFilter,
    limit: 100,
  })

  const createMode = useCreateMode()
  const updateMode = useUpdateMode()
  const deleteMode = useDeleteMode()
  const rebuildDataset = useRebuildDataset()

  const handleCreateMode = () => {
    setEditingMode(null)
    setIsFormOpen(true)
  }

  const handleEditMode = (mode: Mode) => {
    setEditingMode(mode)
    setIsFormOpen(true)
  }

  const handleDeleteMode = async (modeId: string) => {
    if (window.confirm('Вы уверены, что хотите удалить этот режим?')) {
      try {
        await deleteMode.mutateAsync(modeId)
      } catch (error) {
        console.error('Failed to delete mode:', error)
        alert('Ошибка при удалении режима')
      }
    }
  }

  const handleRebuildDataset = async (modeId: string) => {
    if (window.confirm('Запустить сборку датасета для этого режима?')) {
      try {
        setRebuildingModeId(modeId)
        const result = await rebuildDataset.mutateAsync(modeId)
        alert(`Датасет создан успешно!\nDataset ID: ${result.dataset_id}\n\nВы будете перенаправлены на страницу датасета.`)
        // Redirect to dataset page
        window.location.href = `/datasets/${result.dataset_id}`
      } catch (error: any) {
        console.error('Failed to rebuild dataset:', error)
        alert(`Ошибка при создании датасета: ${error.response?.data?.detail || error.message}`)
      } finally {
        setRebuildingModeId(null)
      }
    }
  }

  const handleFormSubmit = async (modeData: Omit<Mode, 'id' | 'created_at' | 'updated_at' | 'created_by'>) => {
    try {
      if (editingMode) {
        await updateMode.mutateAsync({ id: editingMode.id, ...modeData })
      } else {
        await createMode.mutateAsync(modeData)
      }
      setIsFormOpen(false)
      setEditingMode(null)
    } catch (error: any) {
      console.error('Failed to save mode:', error)
      alert(`Ошибка при сохранении режима: ${error.response?.data?.detail || error.message}`)
    }
  }

  const handleFormCancel = () => {
    setIsFormOpen(false)
    setEditingMode(null)
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Режимы</h2>
          <p className="text-muted-foreground">Конфигурации для сборки датасетов</p>
        </div>
        <Button onClick={handleCreateMode}>Добавить режим</Button>
      </div>

      {/* Filters */}
      <div className="flex gap-4 items-end">
        <div className="flex-1 max-w-xs">
          <label htmlFor="asset-filter" className="block text-sm font-medium mb-1">
            Ассет
          </label>
          <input
            id="asset-filter"
            type="text"
            value={assetFilter}
            onChange={(e) => setAssetFilter(e.target.value)}
            placeholder="BTCUSDT"
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          />
        </div>
        <div className="flex-1 max-w-xs">
          <label htmlFor="strategy-filter" className="block text-sm font-medium mb-1">
            Стратегия
          </label>
          <input
            id="strategy-filter"
            type="text"
            value={strategyFilter}
            onChange={(e) => setStrategyFilter(e.target.value)}
            placeholder="test-strategy"
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          />
        </div>
        <div className="flex-1 max-w-xs">
          <label htmlFor="active-filter" className="block text-sm font-medium mb-1">
            Статус
          </label>
          <select
            id="active-filter"
            value={isActiveFilter === undefined ? '' : isActiveFilter.toString()}
            onChange={(e) => setIsActiveFilter(e.target.value === '' ? undefined : e.target.value === 'true')}
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          >
            <option value="">Все</option>
            <option value="true">Активные</option>
            <option value="false">Неактивные</option>
          </select>
        </div>
      </div>

      {/* Form Modal */}
      {isFormOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-background rounded-lg shadow-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <ModeForm
              mode={editingMode}
              onSubmit={handleFormSubmit}
              onCancel={handleFormCancel}
            />
          </div>
        </div>
      )}

      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Название</TableHead>
                <TableHead>Ассет</TableHead>
                <TableHead>Стратегия</TableHead>
                <TableHead>Feature Registry</TableHead>
                <TableHead>Target Registry</TableHead>
                <TableHead>Длительности (дни)</TableHead>
                <TableHead>Статус</TableHead>
                <TableHead>Создан</TableHead>
                <TableHead>Действия</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {!modes || modes.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={9} className="text-center text-muted-foreground">
                    Нет режимов
                  </TableCell>
                </TableRow>
              ) : (
                modes.map((mode) => (
                  <TableRow key={mode.id}>
                    <TableCell className="font-medium">{mode.name}</TableCell>
                    <TableCell>{mode.asset}</TableCell>
                    <TableCell>{mode.strategy_id}</TableCell>
                    <TableCell className="text-xs">{mode.feature_registry_version}</TableCell>
                    <TableCell className="text-xs">{mode.target_registry_version}</TableCell>
                    <TableCell className="text-xs">
                      Train: {mode.train_duration_days}d / Val: {mode.validation_duration_days}d / Test: {mode.test_duration_days}d
                    </TableCell>
                    <TableCell>
                      <Badge variant={mode.is_active ? 'default' : 'secondary'}>
                        {mode.is_active ? 'Активен' : 'Неактивен'}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs">
                      {format(parseISO(mode.created_at), 'dd.MM.yyyy HH:mm')}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleEditMode(mode)}
                        >
                          Редактировать
                        </Button>
                        <Button
                          size="sm"
                          variant="default"
                          onClick={() => handleRebuildDataset(mode.id)}
                          disabled={rebuildingModeId === mode.id || !mode.is_active}
                        >
                          {rebuildingModeId === mode.id ? 'Создание...' : 'Rebuild Dataset'}
                        </Button>
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => handleDeleteMode(mode.id)}
                        >
                          Удалить
                        </Button>
                      </div>
                    </TableCell>
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

