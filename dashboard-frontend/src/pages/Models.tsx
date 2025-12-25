import { useState, useMemo } from 'react'
import { useModels, useModelTrainingHistory } from '@/hooks/useModels'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import api from '@/lib/api'

export default function Models() {
  const { data, isLoading } = useModels({ is_active: true })
  const { data: trainingHistory, isLoading: isHistoryLoading } = useModelTrainingHistory({ limit: 100 })
  const [retrainingModelId, setRetrainingModelId] = useState<string | null>(null)
  
  // Filters for metrics chart
  const [chartSymbolFilter, setChartSymbolFilter] = useState<string>('')
  const [chartStrategyFilter, setChartStrategyFilter] = useState<string>('')
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([
    'accuracy',
    'f1_score',
    'precision',
    'recall',
    'roc_auc',
    'pr_auc',
    'balanced_accuracy',
  ])

  // Get unique symbols and strategies from history
  const uniqueSymbols = useMemo(() => {
    if (!trainingHistory) return []
    const symbols = new Set<string>()
    trainingHistory.forEach((item) => {
      if (item.symbol) symbols.add(item.symbol)
    })
    return Array.from(symbols).sort()
  }, [trainingHistory])

  const uniqueStrategies = useMemo(() => {
    if (!trainingHistory) return []
    const strategies = new Set<string>()
    trainingHistory.forEach((item) => {
      if (item.strategy_id) strategies.add(item.strategy_id)
    })
    return Array.from(strategies).sort()
  }, [trainingHistory])

  // Prepare chart data
  const chartData = useMemo(() => {
    if (!trainingHistory) return []

    let filtered = trainingHistory

    // Apply filters
    if (chartSymbolFilter) {
      filtered = filtered.filter((item) => item.symbol === chartSymbolFilter)
    }
    if (chartStrategyFilter) {
      filtered = filtered.filter((item) => item.strategy_id === chartStrategyFilter)
    }

    // Sort by training date
    filtered = [...filtered].sort((a, b) => new Date(a.trained_at).getTime() - new Date(b.trained_at).getTime())

    // Prepare data for chart
    return filtered.map((item) => {
      const dataPoint: any = {
        date: format(parseISO(item.trained_at), 'dd.MM.yyyy HH:mm'),
        version: item.version,
      }

      // Add selected metrics
      selectedMetrics.forEach((metricName) => {
        const value = item.metrics?.[metricName as keyof typeof item.metrics]
        if (value !== null && value !== undefined) {
          // For percentage metrics (0-1 range), multiply by 100 for better visualization
          if (['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy', 'win_rate', 'pr_auc', 'roc_auc'].includes(metricName)) {
            dataPoint[metricName] = value * 100
          } else {
            dataPoint[metricName] = value
          }
        }
      })

      return dataPoint
    })
  }, [trainingHistory, chartSymbolFilter, chartStrategyFilter, selectedMetrics])

  // All available metrics
  const allMetrics = [
    // Classification
    { name: 'accuracy', label: 'Accuracy (%)', type: 'classification' },
    { name: 'f1_score', label: 'F1 Score (%)', type: 'classification' },
    { name: 'precision', label: 'Precision (%)', type: 'classification' },
    { name: 'recall', label: 'Recall (%)', type: 'classification' },
    { name: 'roc_auc', label: 'ROC AUC (%)', type: 'classification' },
    { name: 'pr_auc', label: 'PR AUC (%)', type: 'classification' },
    { name: 'balanced_accuracy', label: 'Balanced Accuracy (%)', type: 'classification' },
    // Regression
    { name: 'mae', label: 'MAE', type: 'regression' },
    { name: 'mse', label: 'MSE', type: 'regression' },
    { name: 'r2_score', label: 'R² Score', type: 'regression' },
    { name: 'rmse', label: 'RMSE', type: 'regression' },
    // Trading Performance
    { name: 'avg_pnl', label: 'Avg PnL', type: 'trading' },
    { name: 'max_drawdown', label: 'Max Drawdown', type: 'trading' },
    { name: 'profit_factor', label: 'Profit Factor', type: 'trading' },
    { name: 'sharpe_ratio', label: 'Sharpe Ratio', type: 'trading' },
    { name: 'total_pnl', label: 'Total PnL', type: 'trading' },
    { name: 'win_rate', label: 'Win Rate (%)', type: 'trading' },
  ]

  const toggleMetric = (metricName: string) => {
    setSelectedMetrics((prev) =>
      prev.includes(metricName) ? prev.filter((m) => m !== metricName) : [...prev, metricName]
    )
  }

  const handleRetrain = async (model: { id: string; symbol: string | null; strategy_id: string | null }) => {
    if (!model.symbol || !model.strategy_id) {
      alert('Модель должна иметь symbol и strategy_id для переобучения')
      return
    }

    setRetrainingModelId(model.id)
    try {
      const response = await api.post('/v1/training/dataset/build', {
        symbol: model.symbol,
        strategy_id: model.strategy_id,
      })
      alert(`Переобучение запущено. Dataset ID: ${response.data.dataset_id}`)
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Ошибка при запуске переобучения'
      alert(`Ошибка: ${errorMessage}`)
    } finally {
      setRetrainingModelId(null)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Модели</h2>
        <p className="text-muted-foreground">ML модели и метрики качества</p>
      </div>

      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Version</TableHead>
              <TableHead>Symbol</TableHead>
              <TableHead>Strategy</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Accuracy</TableHead>
              <TableHead>F1 Score</TableHead>
              <TableHead>Precision</TableHead>
              <TableHead>Recall</TableHead>
              <TableHead>ROC AUC</TableHead>
              <TableHead>PR AUC</TableHead>
              <TableHead>Balanced Acc</TableHead>
              <TableHead>Trained At</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data?.models.length === 0 ? (
              <TableRow>
                <TableCell colSpan={14} className="text-center text-muted-foreground">
                  Нет моделей
                </TableCell>
              </TableRow>
            ) : (
              data?.models.map((model) => (
                <TableRow key={model.id}>
                  <TableCell className="font-medium">{model.version}</TableCell>
                  <TableCell>{model.symbol || 'All'}</TableCell>
                  <TableCell>{model.strategy_id || 'N/A'}</TableCell>
                  <TableCell>{model.model_type}</TableCell>
                  <TableCell>
                    {model.metrics?.accuracy ? (model.metrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {model.metrics?.f1_score ? (model.metrics.f1_score * 100).toFixed(2) + '%' : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {model.metrics?.precision ? (model.metrics.precision * 100).toFixed(2) + '%' : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {model.metrics?.recall ? (model.metrics.recall * 100).toFixed(2) + '%' : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {model.metrics?.roc_auc ? model.metrics.roc_auc.toFixed(4) : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {model.metrics?.pr_auc ? model.metrics.pr_auc.toFixed(4) : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {model.metrics?.balanced_accuracy ? (model.metrics.balanced_accuracy * 100).toFixed(2) + '%' : 'N/A'}
                  </TableCell>
                  <TableCell>{format(parseISO(model.trained_at), 'dd.MM.yyyy HH:mm:ss')}</TableCell>
                  <TableCell>
                    <Badge variant={model.is_active ? 'default' : 'outline'}>
                      {model.is_active ? 'Active' : 'Inactive'}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Button
                      onClick={() => handleRetrain(model)}
                      disabled={retrainingModelId === model.id || !model.symbol || !model.strategy_id}
                      size="sm"
                      variant="outline"
                    >
                      {retrainingModelId === model.id ? 'Запуск...' : 'Retrain'}
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      )}

      {/* Training History Table */}
      <div className="mt-12">
        <div className="mb-4">
          <h3 className="text-2xl font-bold tracking-tight">История обучения моделей</h3>
          <p className="text-muted-foreground">Полная история всех обучений моделей</p>
        </div>

        {isHistoryLoading ? (
          <Skeleton className="h-64 w-full" />
        ) : (
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Версия</TableHead>
                  <TableHead>Символ</TableHead>
                  <TableHead>Стратегия</TableHead>
                  <TableHead>Тип</TableHead>
                  <TableHead>Дата обучения</TableHead>
                  <TableHead>Фичей</TableHead>
                  <TableHead>Версия фич</TableHead>
                  <TableHead>Версия таргета</TableHead>
                  <TableHead>Dataset ID</TableHead>
                  <TableHead>Accuracy</TableHead>
                  <TableHead>F1</TableHead>
                  <TableHead>Precision</TableHead>
                  <TableHead>Recall</TableHead>
                  <TableHead>ROC AUC</TableHead>
                  <TableHead>PR AUC</TableHead>
                  <TableHead>Balanced Acc</TableHead>
                  <TableHead>Sharpe Ratio</TableHead>
                  <TableHead>Win Rate</TableHead>
                  <TableHead>Статус</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {!trainingHistory || trainingHistory.length === 0 ? (
                  <TableRow>
                  <TableCell colSpan={19} className="text-center text-muted-foreground">
                    Нет истории обучения
                  </TableCell>
                  </TableRow>
                ) : (
                  trainingHistory.map((item) => (
                    <TableRow key={item.id}>
                      <TableCell className="font-medium">{item.version}</TableCell>
                      <TableCell>{item.symbol || 'All'}</TableCell>
                      <TableCell>{item.strategy_id || 'N/A'}</TableCell>
                      <TableCell>{item.model_type}</TableCell>
                      <TableCell className="text-xs">
                        {format(parseISO(item.trained_at), 'dd.MM.yyyy HH:mm:ss')}
                      </TableCell>
                      <TableCell>{item.feature_count || 'N/A'}</TableCell>
                      <TableCell className="text-xs">{item.feature_registry_version || 'N/A'}</TableCell>
                      <TableCell className="text-xs">{item.target_registry_version || 'N/A'}</TableCell>
                      <TableCell className="font-mono text-xs">
                        {item.dataset_id ? `${item.dataset_id.slice(0, 8)}...` : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {item.metrics?.accuracy ? (item.metrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {item.metrics?.f1_score ? (item.metrics.f1_score * 100).toFixed(2) + '%' : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {item.metrics?.precision ? (item.metrics.precision * 100).toFixed(2) + '%' : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {item.metrics?.recall ? (item.metrics.recall * 100).toFixed(2) + '%' : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {item.metrics?.roc_auc ? item.metrics.roc_auc.toFixed(4) : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {item.metrics?.pr_auc ? item.metrics.pr_auc.toFixed(4) : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {item.metrics?.balanced_accuracy ? (item.metrics.balanced_accuracy * 100).toFixed(2) + '%' : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {item.metrics?.sharpe_ratio ? item.metrics.sharpe_ratio.toFixed(4) : 'N/A'}
                      </TableCell>
                      <TableCell>
                        {item.metrics?.win_rate ? (item.metrics.win_rate * 100).toFixed(2) + '%' : 'N/A'}
                      </TableCell>
                      <TableCell>
                        <Badge variant={item.is_active ? 'default' : 'outline'}>
                          {item.is_active ? 'Active' : 'Inactive'}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      {/* Metrics Chart */}
      <div className="mt-12">
        <div className="mb-4">
          <h3 className="text-2xl font-bold tracking-tight">График метрик по истории обучения</h3>
          <p className="text-muted-foreground">Визуализация изменений метрик качества моделей во времени</p>
        </div>

        {/* Filters and Metric Selection */}
        <div className="mb-6 space-y-4">
          <div className="flex gap-4 items-end">
            <div className="flex-1 max-w-xs">
              <label htmlFor="chart-symbol-filter" className="block text-sm font-medium mb-1">
                Символ (Asset)
              </label>
              <select
                id="chart-symbol-filter"
                value={chartSymbolFilter}
                onChange={(e) => setChartSymbolFilter(e.target.value)}
                className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
              >
                <option value="">Все символы</option>
                {uniqueSymbols.map((symbol) => (
                  <option key={symbol} value={symbol}>
                    {symbol}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex-1 max-w-xs">
              <label htmlFor="chart-strategy-filter" className="block text-sm font-medium mb-1">
                Стратегия
              </label>
              <select
                id="chart-strategy-filter"
                value={chartStrategyFilter}
                onChange={(e) => setChartStrategyFilter(e.target.value)}
                className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
              >
                <option value="">Все стратегии</option>
                {uniqueStrategies.map((strategy) => (
                  <option key={strategy} value={strategy}>
                    {strategy}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Metric Selection */}
          <div className="border rounded-md p-4">
            <div className="mb-2 text-sm font-medium">Выберите метрики для отображения:</div>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
              {allMetrics.map((metric) => (
                <label key={metric.name} className="flex items-center space-x-2 cursor-pointer text-sm">
                  <input
                    type="checkbox"
                    checked={selectedMetrics.includes(metric.name)}
                    onChange={() => toggleMetric(metric.name)}
                    className="rounded border-gray-300"
                  />
                  <span>{metric.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Chart */}
        {isHistoryLoading ? (
          <Skeleton className="h-96 w-full" />
        ) : chartData.length === 0 ? (
          <div className="text-center text-muted-foreground py-8 border rounded-md">
            Нет данных для отображения. Выберите другие фильтры или подождите загрузки истории обучения.
          </div>
        ) : (
          <div className="border rounded-md p-4">
            <ResponsiveContainer width="100%" height={500}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                {selectedMetrics.map((metricName, index) => {
                  const metric = allMetrics.find((m) => m.name === metricName)
                  if (!metric) return null

                  // Color palette for different metrics
                  const colors = [
                    '#8884d8',
                    '#82ca9d',
                    '#ffc658',
                    '#ff7300',
                    '#00ff00',
                    '#0088fe',
                    '#00c49f',
                    '#ffbb28',
                    '#ff8042',
                    '#8884d8',
                    '#82ca9d',
                    '#ffc658',
                    '#ff7300',
                    '#00ff00',
                    '#0088fe',
                    '#00c49f',
                    '#ffbb28',
                  ]

                  return (
                    <Line
                      key={metricName}
                      type="monotone"
                      dataKey={metricName}
                      stroke={colors[index % colors.length]}
                      name={metric.label}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      connectNulls
                    />
                  )
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

