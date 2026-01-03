import { useState, useMemo, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  useModels,
  useModelTrainingHistory,
  useSignalSuccessRate,
  useAvailableAssets,
  useAvailableStrategies,
  useActiveModelVersion,
  useDeactivateModel
} from '@/hooks/useModels'
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
  const [relearningModelId, setRelearningModelId] = useState<string | null>(null)
  const [deactivatingModelVersion, setDeactivatingModelVersion] = useState<string | null>(null)
  
  const deactivateModel = useDeactivateModel()

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

  // Filters for signal success rate chart
  const [successRateModelVersion, setSuccessRateModelVersion] = useState<string>('')
  const [successRateAsset, setSuccessRateAsset] = useState<string>('')
  const [successRateStrategy, setSuccessRateStrategy] = useState<string>('')

  // Fetch available assets and strategies
  const { data: availableAssets = [], isLoading: isLoadingAssets } = useAvailableAssets()
  const { data: availableStrategies = [], isLoading: isLoadingStrategies } = useAvailableStrategies()

  // Fetch active model version when asset and strategy are selected
  const { data: activeModelVersion } = useActiveModelVersion({
    asset: successRateAsset || undefined,
    strategy_id: successRateStrategy || undefined,
  })

  // Auto-fill model version when asset or strategy changes
  useEffect(() => {
    if (activeModelVersion && successRateAsset && successRateStrategy) {
      setSuccessRateModelVersion(activeModelVersion)
    } else if ((!successRateAsset || !successRateStrategy) && successRateModelVersion) {
      // Clear version if asset or strategy is cleared
      setSuccessRateModelVersion('')
    }
  }, [activeModelVersion, successRateAsset, successRateStrategy])

  // Fetch signal success rate data
  const { data: successRateData, isLoading: isLoadingSuccessRate } = useSignalSuccessRate({
    model_version: successRateModelVersion || undefined,
    asset: successRateAsset || undefined,
    strategy_id: successRateStrategy || undefined,
  })

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
    { name: 'directional_accuracy', label: 'Directional Accuracy (%)', type: 'regression' },

    { name: 'information_coefficient', label: 'Information Coefficient (IC)', type: 'regression' },
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

  const handleRelearn = async (model: { id: string; symbol: string | null; strategy_id: string | null; training_config: any }) => {
    if (!model.symbol) {
      alert('Модель должна иметь symbol для Re-Learn')
      return
    }

    // Extract dataset_id from training_config
    let datasetId: string | null = null
    if (model.training_config) {
      if (typeof model.training_config === 'string') {
        try {
          const config = JSON.parse(model.training_config)
          datasetId = config.dataset_id || null
        } catch {
          // Ignore parse errors
        }
      } else if (typeof model.training_config === 'object' && model.training_config !== null) {
        datasetId = model.training_config.dataset_id || null
      }
    }

    if (!datasetId) {
      alert('Не удалось найти dataset_id в конфигурации модели. Re-Learn доступен только для моделей, обученных на датасете.')
      return
    }

    setRelearningModelId(model.id)
    try {
      const response = await api.post('/v1/models/relearn', {
        dataset_id: datasetId,
        symbol: model.symbol,
        strategy_id: model.strategy_id || null,
      })
      alert(`Re-Learn запущен. Dataset ID: ${datasetId}, Trace ID: ${response.data.trace_id}`)
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Ошибка при запуске Re-Learn'
      alert(`Ошибка: ${errorMessage}`)
    } finally {
      setRelearningModelId(null)
    }
  }

  const handleDeactivate = async (model: { version: string; is_active: boolean }) => {
    if (!model.is_active) {
      alert('Модель уже деактивирована')
      return
    }

    if (!window.confirm(`Вы уверены, что хотите деактивировать модель ${model.version}?`)) {
      return
    }

    setDeactivatingModelVersion(model.version)
    try {
      await deactivateModel.mutateAsync(model.version)
      alert(`Модель ${model.version} успешно деактивирована`)
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Ошибка при деактивации модели'
      alert(`Ошибка: ${errorMessage}`)
    } finally {
      setDeactivatingModelVersion(null)
    }
  }

  // Determine if model is regression based on metrics or training_config
  const isRegressionModel = (model: { metrics: any; training_config?: any }) => {
    // Check training_config first (most reliable)
    if (model.training_config) {
      let config = model.training_config
      if (typeof config === 'string') {
        try {
          config = JSON.parse(config)
        } catch {
          // Ignore parse errors
        }
      }
      if (config && typeof config === 'object' && config.task_type === 'regression') {
        return true
      }
    }

    // Fallback: check metrics - regression has r2_score, mse, mae, rmse
    if (model.metrics) {
      const hasRegressionMetrics = (
        (model.metrics.r2_score !== null && model.metrics.r2_score !== undefined) ||
        (model.metrics.mse !== null && model.metrics.mse !== undefined) ||
        (model.metrics.mae !== null && model.metrics.mae !== undefined) ||
        (model.metrics.rmse !== null && model.metrics.rmse !== undefined)
      )
      const hasClassificationMetrics = (
        (model.metrics.accuracy !== null && model.metrics.accuracy !== undefined) ||
        (model.metrics.f1_score !== null && model.metrics.f1_score !== undefined) ||
        (model.metrics.precision !== null && model.metrics.precision !== undefined)
      )

      // If has regression metrics but no classification metrics, it's regression
      if (hasRegressionMetrics && !hasClassificationMetrics) {
        return true
      }
    }

    return false
  }

  // Get metric value for a column based on model type
  const getMetricValue = (model: { metrics: any; training_config?: any }, column: string) => {
    const isRegression = isRegressionModel(model)
    const metrics = model.metrics || {}

    if (isRegression) {
      // Map classification columns to regression metrics
      switch (column) {
        case 'accuracy':
          return metrics.r2_score !== null && metrics.r2_score !== undefined
            ? { value: metrics.r2_score, label: 'R² Score', format: (v: number) => v.toFixed(4) }
            : null
        case 'f1_score':
          return metrics.directional_accuracy !== null && metrics.directional_accuracy !== undefined
            ? { value: metrics.directional_accuracy, label: 'Directional Accuracy', format: (v: number) => (v * 100).toFixed(2) + '%' }
            : null
        case 'precision':
          return metrics.sharpe_ratio !== null && metrics.sharpe_ratio !== undefined
            ? { value: metrics.sharpe_ratio, label: 'Sharpe Ratio', format: (v: number) => v.toFixed(4) }
            : null
        case 'recall':
          return metrics.information_coefficient !== null && metrics.information_coefficient !== undefined
            ? { value: metrics.information_coefficient, label: 'Information Coefficient (IC)', format: (v: number) => v.toFixed(4) }
            : null
        case 'roc_auc':
          return metrics.rmse !== null && metrics.rmse !== undefined
            ? { value: metrics.rmse, label: 'RMSE', format: (v: number) => v.toFixed(6) }
            : null
        case 'pr_auc':
          return metrics.mae !== null && metrics.mae !== undefined
            ? { value: metrics.mae, label: 'MAE', format: (v: number) => v.toFixed(6) }
            : null
        case 'balanced_accuracy':
          return metrics.mse !== null && metrics.mse !== undefined
            ? { value: metrics.mse, label: 'MSE', format: (v: number) => v.toFixed(6) }
            : null
        default:
          return null
      }
    } else {
      // Classification metrics
      switch (column) {
        case 'accuracy':
          return metrics.accuracy !== null && metrics.accuracy !== undefined
            ? { value: metrics.accuracy, label: 'Accuracy', format: (v: number) => (v * 100).toFixed(2) + '%' }
            : null
        case 'f1_score':
          return metrics.f1_score !== null && metrics.f1_score !== undefined
            ? { value: metrics.f1_score, label: 'F1 Score', format: (v: number) => (v * 100).toFixed(2) + '%' }
            : null
        case 'precision':
          return metrics.precision !== null && metrics.precision !== undefined
            ? { value: metrics.precision, label: 'Precision', format: (v: number) => (v * 100).toFixed(2) + '%' }
            : null
        case 'recall':
          return metrics.recall !== null && metrics.recall !== undefined
            ? { value: metrics.recall, label: 'Recall', format: (v: number) => (v * 100).toFixed(2) + '%' }
            : null
        case 'roc_auc':
          return metrics.roc_auc !== null && metrics.roc_auc !== undefined
            ? { value: metrics.roc_auc, label: 'ROC AUC', format: (v: number) => v.toFixed(4) }
            : null
        case 'pr_auc':
          return metrics.pr_auc !== null && metrics.pr_auc !== undefined
            ? { value: metrics.pr_auc, label: 'PR AUC', format: (v: number) => v.toFixed(4) }
            : null
        case 'balanced_accuracy':
          return metrics.balanced_accuracy !== null && metrics.balanced_accuracy !== undefined
            ? { value: metrics.balanced_accuracy, label: 'Balanced Accuracy', format: (v: number) => (v * 100).toFixed(2) + '%' }
            : null
        default:
          return null
      }
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
              data?.models.map((model) => {
                const accuracyMetric = getMetricValue(model, 'accuracy')
                const f1Metric = getMetricValue(model, 'f1_score')
                const precisionMetric = getMetricValue(model, 'precision')
                const recallMetric = getMetricValue(model, 'recall')
                const rocAucMetric = getMetricValue(model, 'roc_auc')
                const prAucMetric = getMetricValue(model, 'pr_auc')
                const balancedAccMetric = getMetricValue(model, 'balanced_accuracy')

                return (
                  <TableRow key={model.id}>
                    <TableCell className="font-medium">
                      <Link
                        to={`/models/${model.version}`}
                        className="text-primary hover:underline"
                      >
                        {model.version}
                      </Link>
                    </TableCell>
                    <TableCell>{model.symbol || 'All'}</TableCell>
                    <TableCell>{model.strategy_id || 'N/A'}</TableCell>
                    <TableCell>
                      <Badge variant="outline">
                        {isRegressionModel(model) ? 'Regression' : 'Classification'}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {accuracyMetric ? (
                        <div className="flex flex-col">
                          <span className="text-xs text-muted-foreground">{accuracyMetric.label}</span>
                          <span>{accuracyMetric.format(accuracyMetric.value)}</span>
                        </div>
                      ) : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {f1Metric ? (
                        <div className="flex flex-col">
                          <span className="text-xs text-muted-foreground">{f1Metric.label}</span>
                          <span>{f1Metric.format(f1Metric.value)}</span>
                        </div>
                      ) : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {precisionMetric ? (
                        <div className="flex flex-col">
                          <span className="text-xs text-muted-foreground">{precisionMetric.label}</span>
                          <span>{precisionMetric.format(precisionMetric.value)}</span>
                        </div>
                      ) : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {recallMetric ? (
                        <div className="flex flex-col">
                          <span className="text-xs text-muted-foreground">{recallMetric.label}</span>
                          <span>{recallMetric.format(recallMetric.value)}</span>
                        </div>
                      ) : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {rocAucMetric ? (
                        <div className="flex flex-col">
                          <span className="text-xs text-muted-foreground">{rocAucMetric.label}</span>
                          <span>{rocAucMetric.format(rocAucMetric.value)}</span>
                        </div>
                      ) : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {prAucMetric ? (
                        <div className="flex flex-col">
                          <span className="text-xs text-muted-foreground">{prAucMetric.label}</span>
                          <span>{prAucMetric.format(prAucMetric.value)}</span>
                        </div>
                      ) : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {balancedAccMetric ? (
                        <div className="flex flex-col">
                          <span className="text-xs text-muted-foreground">{balancedAccMetric.label}</span>
                          <span>{balancedAccMetric.format(balancedAccMetric.value)}</span>
                        </div>
                      ) : 'N/A'}
                    </TableCell>
                    <TableCell>{format(parseISO(model.trained_at), 'dd.MM.yyyy HH:mm:ss')}</TableCell>
                    <TableCell>
                      <Badge variant={model.is_active ? 'default' : 'outline'}>
                        {model.is_active ? 'Active' : 'Inactive'}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-2">
                        <Button
                          onClick={() => handleRetrain(model)}
                          disabled={retrainingModelId === model.id || !model.symbol || !model.strategy_id}
                          size="sm"
                          variant="outline"
                        >
                          {retrainingModelId === model.id ? 'Запуск...' : 'Retrain'}
                        </Button>
                        <Button
                          onClick={() => handleRelearn(model)}
                          disabled={relearningModelId === model.id || !model.symbol || !model.training_config}
                          size="sm"
                          variant="outline"
                          title="Перезапустить обучение на том же датасете"
                        >
                          {relearningModelId === model.id ? 'Запуск...' : 'Re-Learn'}
                        </Button>
                        {model.is_active && (
                          <Button
                            onClick={() => handleDeactivate(model)}
                            disabled={deactivatingModelVersion === model.version}
                            size="sm"
                            variant="destructive"
                          >
                            {deactivatingModelVersion === model.version ? 'Деактивация...' : 'Деактивировать'}
                          </Button>
                        )}
                        <Button
                          asChild
                          size="sm"
                          variant="default"
                        >
                          <Link to={`/models/${model.version}`}>
                            Детали
                          </Link>
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })
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
                  <TableHead>Win Rate</TableHead>
                  <TableHead>Статус</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {!trainingHistory || trainingHistory.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={18} className="text-center text-muted-foreground">
                      Нет истории обучения
                    </TableCell>
                  </TableRow>
                ) : (
                  trainingHistory.map((item) => {
                    const isRegression = isRegressionModel(item)
                    const accuracyMetric = getMetricValue(item, 'accuracy')
                    const f1Metric = getMetricValue(item, 'f1_score')
                    const precisionMetric = getMetricValue(item, 'precision')
                    const recallMetric = getMetricValue(item, 'recall')
                    const rocAucMetric = getMetricValue(item, 'roc_auc')
                    const prAucMetric = getMetricValue(item, 'pr_auc')
                    const balancedAccMetric = getMetricValue(item, 'balanced_accuracy')

                    return (
                      <TableRow key={item.id}>
                        <TableCell className="font-medium">
                          <Link
                            to={`/models/${item.version}`}
                            className="text-primary hover:underline"
                          >
                            {item.version}
                          </Link>
                        </TableCell>
                        <TableCell>{item.symbol || 'All'}</TableCell>
                        <TableCell>{item.strategy_id || 'N/A'}</TableCell>
                        <TableCell>
                          <Badge variant="outline">
                            {isRegression ? 'Regression' : 'Classification'}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-xs">
                          {format(parseISO(item.trained_at), 'dd.MM.yyyy HH:mm:ss')}
                        </TableCell>
                        <TableCell>{item.feature_count || 'N/A'}</TableCell>
                        <TableCell className="text-xs">{item.feature_registry_version || 'N/A'}</TableCell>
                        <TableCell className="text-xs">{item.target_registry_version || 'N/A'}</TableCell>
                        <TableCell className="font-mono text-xs">
                          {item.dataset_id ? (
                            <Link
                              to={`/datasets/${item.dataset_id}`}
                              className="text-primary hover:underline"
                            >
                              {item.dataset_id.slice(0, 8)}...
                            </Link>
                          ) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {accuracyMetric ? (
                            <div className="flex flex-col">
                              <span className="text-xs text-muted-foreground">{accuracyMetric.label}</span>
                              <span>{accuracyMetric.format(accuracyMetric.value)}</span>
                            </div>
                          ) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {f1Metric ? (
                            <div className="flex flex-col">
                              <span className="text-xs text-muted-foreground">{f1Metric.label}</span>
                              <span>{f1Metric.format(f1Metric.value)}</span>
                            </div>
                          ) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {precisionMetric ? (
                            <div className="flex flex-col">
                              <span className="text-xs text-muted-foreground">{precisionMetric.label}</span>
                              <span>{precisionMetric.format(precisionMetric.value)}</span>
                            </div>
                          ) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {recallMetric ? (
                            <div className="flex flex-col">
                              <span className="text-xs text-muted-foreground">{recallMetric.label}</span>
                              <span>{recallMetric.format(recallMetric.value)}</span>
                            </div>
                          ) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {rocAucMetric ? (
                            <div className="flex flex-col">
                              <span className="text-xs text-muted-foreground">{rocAucMetric.label}</span>
                              <span>{rocAucMetric.format(rocAucMetric.value)}</span>
                            </div>
                          ) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {prAucMetric ? (
                            <div className="flex flex-col">
                              <span className="text-xs text-muted-foreground">{prAucMetric.label}</span>
                              <span>{prAucMetric.format(prAucMetric.value)}</span>
                            </div>
                          ) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {balancedAccMetric ? (
                            <div className="flex flex-col">
                              <span className="text-xs text-muted-foreground">{balancedAccMetric.label}</span>
                              <span>{balancedAccMetric.format(balancedAccMetric.value)}</span>
                            </div>
                          ) : 'N/A'}
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
                    )
                  })
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

      {/* Signal Success Rate Chart */}
      <div className="mt-12">
        <div className="mb-4">
          <h3 className="text-2xl font-bold tracking-tight">Статистика успешности сигналов</h3>
          <p className="text-muted-foreground">Процент успешности сигналов по часам с группировкой</p>
        </div>

        {/* Filters */}
        <div className="mb-6 space-y-4">
          <div className="flex gap-4 items-end">
            <div className="flex-1 max-w-xs">
              <label htmlFor="success-rate-model-version" className="block text-sm font-medium mb-1">
                Версия модели
              </label>
              <input
                id="success-rate-model-version"
                type="text"
                value={successRateModelVersion}
                onChange={(e) => setSuccessRateModelVersion(e.target.value)}
                placeholder="Например: v1.0"
                className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
              />
            </div>
            <div className="flex-1 max-w-xs">
              <label htmlFor="success-rate-asset" className="block text-sm font-medium mb-1">
                Ассет
              </label>
              <select
                id="success-rate-asset"
                value={successRateAsset}
                onChange={(e) => setSuccessRateAsset(e.target.value)}
                className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
                disabled={isLoadingAssets}
              >
                <option value="">Выберите ассет</option>
                {availableAssets.map((asset) => (
                  <option key={asset} value={asset}>
                    {asset}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex-1 max-w-xs">
              <label htmlFor="success-rate-strategy" className="block text-sm font-medium mb-1">
                Стратегия
              </label>
              <select
                id="success-rate-strategy"
                value={successRateStrategy}
                onChange={(e) => setSuccessRateStrategy(e.target.value)}
                className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
                disabled={isLoadingStrategies}
              >
                <option value="">Выберите стратегию</option>
                {availableStrategies.map((strategy) => (
                  <option key={strategy} value={strategy}>
                    {strategy}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Chart */}
        {isLoadingSuccessRate ? (
          <Skeleton className="h-96 w-full" />
        ) : !successRateModelVersion || !successRateAsset || !successRateStrategy ? (
          <div className="text-center text-muted-foreground py-8 border rounded-md">
            Заполните все поля фильтров для отображения статистики успешности сигналов
          </div>
        ) : !successRateData || successRateData.data.length === 0 ? (
          <div className="text-center text-muted-foreground py-8 border rounded-md">
            Нет данных для отображения. Проверьте выбранные фильтры.
          </div>
        ) : (
          <div className="space-y-6">
            {/* Success Rate by Direction Chart */}
            <div className="border rounded-md p-4">
              <h4 className="text-lg font-semibold mb-4">Процент успешности по направлению</h4>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={successRateData.data.map((item) => ({
                  hour: format(parseISO(item.hour), 'dd.MM.yyyy HH:mm'),
                  successRate: item.success_rate_direction_percent,
                  totalSignals: item.total_signals,
                  successfulSignals: item.successful_by_direction,
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" angle={-45} textAnchor="end" height={100} />
                  <YAxis label={{ value: 'Процент успешности (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    formatter={(value: any, name: string) => {
                      if (name === 'successRate') return [`${value?.toFixed(2) || 0}%`, 'Процент успешности']
                      if (name === 'totalSignals') return [value, 'Всего сигналов']
                      if (name === 'successfulSignals') return [value, 'Успешных сигналов']
                      return [value, name]
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="successRate"
                    stroke="#8884d8"
                    name="Процент успешности (%)"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Success Rate by PnL Chart */}
            <div className="border rounded-md p-4">
              <h4 className="text-lg font-semibold mb-4">Процент успешности по финансовому результату (PnL)</h4>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={successRateData.data.map((item) => ({
                  hour: format(parseISO(item.hour), 'dd.MM.yyyy HH:mm'),
                  successRate: item.success_rate_pnl_percent,
                  totalSignals: item.total_signals,
                  successfulSignals: item.successful_by_pnl,
                  totalPnl: item.total_pnl_sum,
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" angle={-45} textAnchor="end" height={100} />
                  <YAxis label={{ value: 'Процент успешности (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    formatter={(value: any, name: string) => {
                      if (name === 'successRate') return [`${value?.toFixed(2) || 0}%`, 'Процент успешности']
                      if (name === 'totalSignals') return [value, 'Всего сигналов']
                      if (name === 'successfulSignals') return [value, 'Успешных сигналов']
                      if (name === 'totalPnl') return [`${value?.toFixed(2) || 0}`, 'Суммарный PnL']
                      return [value, name]
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="successRate"
                    stroke="#82ca9d"
                    name="Процент успешности (%)"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Summary Statistics Table */}
            <div className="border rounded-md p-4">
              <h4 className="text-lg font-semibold mb-4">Сводная статистика</h4>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Час</TableHead>
                      <TableHead>Всего сигналов</TableHead>
                      <TableHead>Оценено</TableHead>
                      <TableHead>Успешных (направление)</TableHead>
                      <TableHead>Успешных (PnL)</TableHead>
                      <TableHead>% успешности (направление)</TableHead>
                      <TableHead>% успешности (PnL)</TableHead>
                      <TableHead>Средняя уверенность</TableHead>
                      <TableHead>Buy/Sell</TableHead>
                      <TableHead>Суммарный PnL</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {successRateData.data.map((item, index) => (
                      <TableRow key={index}>
                        <TableCell className="text-xs">
                          {format(parseISO(item.hour), 'dd.MM.yyyy HH:mm')}
                        </TableCell>
                        <TableCell>{item.total_signals}</TableCell>
                        <TableCell>{item.evaluated_signals}</TableCell>
                        <TableCell>{item.successful_by_direction}</TableCell>
                        <TableCell>{item.successful_by_pnl}</TableCell>
                        <TableCell>
                          {item.success_rate_direction_percent !== null
                            ? `${item.success_rate_direction_percent.toFixed(2)}%`
                            : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {item.success_rate_pnl_percent !== null
                            ? `${item.success_rate_pnl_percent.toFixed(2)}%`
                            : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {item.avg_confidence !== null
                            ? `${(item.avg_confidence * 100).toFixed(2)}%`
                            : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {item.buy_signals}/{item.sell_signals}
                        </TableCell>
                        <TableCell>
                          {item.total_pnl_sum !== null
                            ? item.total_pnl_sum.toFixed(2)
                            : 'N/A'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

