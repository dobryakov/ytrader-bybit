import { useParams, useNavigate, Link } from 'react-router-dom'
import { useDataset } from '@/hooks/useDatasets'
import { useModelsByDataset } from '@/hooks/useModels'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { ArrowLeft, Calendar, Database, FileText, Settings, TrendingUp } from 'lucide-react'
import { MetricCard } from '@/components/metrics/MetricCard'

function getStatusBadgeVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
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

function getStatusLabel(status: string) {
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

function getSplitStrategyLabel(strategy: string) {
  switch (strategy) {
    case 'time_based':
      return 'По времени'
    case 'walk_forward':
      return 'Walk-forward'
    default:
      return strategy
  }
}

export default function DatasetDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { data, isLoading, error } = useDataset(id || '')
  const { data: modelsData, isLoading: isLoadingModels } = useModelsByDataset(id || '')

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="space-y-6">
        <div>
          <Button variant="outline" onClick={() => navigate('/datasets')} className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Назад к датасетам
          </Button>
          <div className="text-center text-muted-foreground py-8">
            Ошибка загрузки данных датасета. ID: {id}
          </div>
        </div>
      </div>
    )
  }

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'N/A'
    try {
      return format(parseISO(dateString), 'dd.MM.yyyy HH:mm:ss')
    } catch {
      return dateString
    }
  }

  const formatDateShort = (dateString: string | null) => {
    if (!dateString) return 'N/A'
    try {
      return format(parseISO(dateString), 'dd.MM.yyyy')
    } catch {
      return dateString
    }
  }

  const formatPercent = (value: number | null) => {
    if (value === null || value === undefined) return 'N/A'
    return `${(value * 100).toFixed(2)}%`
  }

  const formatDecimal = (value: number | null, decimals: number = 4) => {
    if (value === null || value === undefined) return 'N/A'
    return value.toFixed(decimals)
  }

  // Prepare data for split distribution chart
  const splitChartData = [
    { name: 'Train', value: data.train_records || 0, color: '#8884d8' },
    { name: 'Validation', value: data.validation_records || 0, color: '#82ca9d' },
    { name: 'Test', value: data.test_records || 0, color: '#ffc658' },
  ].filter(item => item.value > 0)

  const totalRecords = (data.train_records || 0) + (data.validation_records || 0) + (data.test_records || 0)

  // Prepare data for bar chart
  const barChartData = [
    { split: 'Train', records: data.train_records || 0, percentage: totalRecords > 0 ? ((data.train_records || 0) / totalRecords * 100).toFixed(1) : '0' },
    { split: 'Validation', records: data.validation_records || 0, percentage: totalRecords > 0 ? ((data.validation_records || 0) / totalRecords * 100).toFixed(1) : '0' },
    { split: 'Test', records: data.test_records || 0, percentage: totalRecords > 0 ? ((data.test_records || 0) / totalRecords * 100).toFixed(1) : '0' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Button variant="outline" onClick={() => navigate('/datasets')} className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Назад к датасетам
          </Button>
          <h2 className="text-3xl font-bold tracking-tight">Детальная информация о датасете</h2>
          <p className="text-muted-foreground">ID: {data.id}</p>
        </div>
      </div>

      {/* Basic Information */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Database className="h-4 w-4" />
              Статус
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant={getStatusBadgeVariant(data.status)}>
              {getStatusLabel(data.status)}
            </Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Символ
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.symbol}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Стратегия
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-medium">{data.strategy_id || 'N/A'}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Всего записей
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalRecords.toLocaleString()}</div>
          </CardContent>
        </Card>
      </div>

      {/* Split Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Распределение записей по сплитам</CardTitle>
          <CardDescription>Количество записей в каждом сплите датасета</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Pie Chart */}
            {splitChartData.length > 0 && (
              <div>
                <h4 className="text-lg font-semibold mb-4">Круговая диаграмма</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={splitChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {splitChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => value.toLocaleString()} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Bar Chart */}
            <div>
              <h4 className="text-lg font-semibold mb-4">Столбчатая диаграмма</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={barChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="split" />
                  <YAxis />
                  <Tooltip 
                    formatter={(value: number) => [
                      `${value.toLocaleString()} записей`,
                      'Количество'
                    ]}
                    labelFormatter={(label) => `Сплит: ${label}`}
                  />
                  <Legend />
                  <Bar dataKey="records" fill="#8884d8" name="Количество записей" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Split Statistics Table */}
          <div className="mt-6">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Сплит</TableHead>
                  <TableHead>Количество записей</TableHead>
                  <TableHead>Процент от общего</TableHead>
                  <TableHead>Распределение классов</TableHead>
                  <TableHead>Баланс классов</TableHead>
                  <TableHead>Статистика таргета</TableHead>
                  <TableHead>Период начала</TableHead>
                  <TableHead>Период окончания</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(['train', 'validation', 'test'] as const).map((splitName) => {
                  const splitStats = data.split_statistics?.[splitName]
                  const recordsKey = `${splitName}_records` as 'train_records' | 'validation_records' | 'test_records'
                  const periodStartKey = `${splitName}_period_start` as 'train_period_start' | 'validation_period_start' | 'test_period_start'
                  const periodEndKey = `${splitName}_period_end` as 'train_period_end' | 'validation_period_end' | 'test_period_end'
                  const records = data[recordsKey] || 0
                  const isClassification = data.target_config?.type === 'classification' || data.target_config?.type === 'risk_adjusted'
                  
                  // Format class distribution
                  const classDist = splitStats?.class_distribution
                  const classDistText = classDist 
                    ? Object.entries(classDist)
                        .map(([cls, count]) => `${cls}: ${count.toLocaleString()}`)
                        .join(', ')
                    : 'N/A'
                  
                  // Class balance ratio
                  const balanceRatio = splitStats?.class_balance_ratio
                  const balanceText = balanceRatio !== undefined 
                    ? `${(balanceRatio * 100).toFixed(1)}%`
                    : 'N/A'
                  
                  // Target statistics
                  const targetStats = splitStats?.target_statistics
                  const targetStatsText = targetStats
                    ? `μ=${targetStats.mean.toFixed(4)}, σ=${targetStats.std.toFixed(4)}`
                    : 'N/A'
                  
                  return (
                    <TableRow key={splitName}>
                      <TableCell className="font-medium capitalize">{splitName === 'train' ? 'Train' : splitName === 'validation' ? 'Validation' : 'Test'}</TableCell>
                      <TableCell>{records.toLocaleString()}</TableCell>
                      <TableCell>
                        {totalRecords > 0 
                          ? `${(records / totalRecords * 100).toFixed(2)}%`
                          : '0%'}
                      </TableCell>
                      <TableCell className="text-xs" title={classDistText}>
                        {isClassification && classDist 
                          ? (Object.keys(classDist).length <= 3 
                              ? classDistText 
                              : `${Object.keys(classDist).length} классов`)
                          : 'N/A'}
                      </TableCell>
                      <TableCell className="text-xs">
                        {isClassification && balanceRatio !== undefined ? (
                          <span className={balanceRatio < 0.3 ? 'text-yellow-600 font-medium' : balanceRatio < 0.5 ? 'text-orange-600 font-medium' : ''}>
                            {balanceText}
                            {balanceRatio < 0.3 && ' ⚠️'}
                          </span>
                        ) : 'N/A'}
                      </TableCell>
                      <TableCell className="text-xs font-mono" title={targetStats ? `Mean: ${targetStats.mean.toFixed(6)}, Median: ${targetStats.median.toFixed(6)}, Std: ${targetStats.std.toFixed(6)}, Min: ${targetStats.min.toFixed(6)}, Max: ${targetStats.max.toFixed(6)}` : ''}>
                        {targetStatsText}
                      </TableCell>
                      <TableCell className="text-xs">{formatDateShort(data[periodStartKey])}</TableCell>
                      <TableCell className="text-xs">{formatDateShort(data[periodEndKey])}</TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Active Model Metrics */}
      {data.status === 'ready' && (
        <Card>
          <CardHeader>
            <CardTitle>Активная модель, обученная на этом датасете</CardTitle>
            <CardDescription>Метрики и статистика модели, обученной на данном датасете</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingModels ? (
              <Skeleton className="h-64 w-full" />
            ) : !modelsData || modelsData.models.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                На этом датасете еще не обучено ни одной модели
              </div>
            ) : (
              <div className="space-y-6">
                {(() => {
                  const activeModel = modelsData.models.find(m => m.is_active)
                  const modelToShow = activeModel || modelsData.models[0]
                  
                  if (!modelToShow) return null
                  
                  const metrics = modelToShow.metrics
                  const isClassification = data.target_config?.type === 'classification' || data.target_config?.type === 'risk_adjusted'
                  
                  return (
                    <>
                      {/* Model Info */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-muted rounded-md">
                        <div>
                          <span className="text-sm text-muted-foreground">Версия модели:</span>
                          <div className="font-medium mt-1">
                            <Link 
                              to={`/models/${modelToShow.version}`}
                              className="text-primary hover:underline"
                            >
                              {modelToShow.version}
                            </Link>
                            {modelToShow.is_active && (
                              <Badge variant="default" className="ml-2">Active</Badge>
                            )}
                          </div>
                        </div>
                        <div>
                          <span className="text-sm text-muted-foreground">Тип модели:</span>
                          <div className="font-medium mt-1">{modelToShow.model_type}</div>
                        </div>
                        <div>
                          <span className="text-sm text-muted-foreground">Обучена:</span>
                          <div className="font-medium mt-1">{formatDate(modelToShow.trained_at)}</div>
                        </div>
                      </div>

                      {/* Classification Metrics */}
                      {isClassification && metrics && (
                        <div>
                          <h4 className="font-semibold mb-4">Метрики классификации</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <MetricCard title="Accuracy" value={formatPercent(metrics.accuracy)} />
                            <MetricCard title="Precision" value={formatPercent(metrics.precision)} />
                            <MetricCard title="Recall" value={formatPercent(metrics.recall)} />
                            <MetricCard title="F1 Score" value={formatPercent(metrics.f1_score)} />
                            <MetricCard title="Balanced Accuracy" value={formatPercent(metrics.balanced_accuracy)} />
                            <MetricCard title="ROC AUC" value={formatDecimal(metrics.roc_auc)} />
                            <MetricCard title="PR AUC" value={formatDecimal(metrics.pr_auc)} />
                          </div>
                        </div>
                      )}

                      {/* Regression Metrics */}
                      {!isClassification && metrics && (
                        <div>
                          <h4 className="font-semibold mb-4">Метрики регрессии</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <MetricCard title="MAE" value={formatDecimal(metrics.mae, 6)} />
                            <MetricCard title="RMSE" value={formatDecimal(metrics.rmse, 6)} />
                            <MetricCard title="R² Score" value={formatDecimal(metrics.r2_score, 4)} />
                            <MetricCard title="MSE" value={formatDecimal(metrics.mse, 6)} />
                          </div>
                        </div>
                      )}

                      {/* All Metrics (if both classification and regression metrics exist) */}
                      {metrics && (
                        (isClassification && (metrics.mae !== null || metrics.rmse !== null || metrics.r2_score !== null || metrics.mse !== null)) ||
                        (!isClassification && (metrics.accuracy !== null || metrics.precision !== null || metrics.recall !== null || metrics.f1_score !== null))
                      ) && (
                        <div>
                          <h4 className="font-semibold mb-4">Дополнительные метрики</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            {isClassification && (
                              <>
                                {metrics.mae !== null && <MetricCard title="MAE" value={formatDecimal(metrics.mae, 6)} />}
                                {metrics.rmse !== null && <MetricCard title="RMSE" value={formatDecimal(metrics.rmse, 6)} />}
                                {metrics.r2_score !== null && <MetricCard title="R² Score" value={formatDecimal(metrics.r2_score, 4)} />}
                                {metrics.mse !== null && <MetricCard title="MSE" value={formatDecimal(metrics.mse, 6)} />}
                              </>
                            )}
                            {!isClassification && (
                              <>
                                {metrics.accuracy !== null && <MetricCard title="Accuracy" value={formatPercent(metrics.accuracy)} />}
                                {metrics.precision !== null && <MetricCard title="Precision" value={formatPercent(metrics.precision)} />}
                                {metrics.recall !== null && <MetricCard title="Recall" value={formatPercent(metrics.recall)} />}
                                {metrics.f1_score !== null && <MetricCard title="F1 Score" value={formatPercent(metrics.f1_score)} />}
                                {metrics.balanced_accuracy !== null && <MetricCard title="Balanced Accuracy" value={formatPercent(metrics.balanced_accuracy)} />}
                                {metrics.roc_auc !== null && <MetricCard title="ROC AUC" value={formatDecimal(metrics.roc_auc)} />}
                                {metrics.pr_auc !== null && <MetricCard title="PR AUC" value={formatDecimal(metrics.pr_auc)} />}
                              </>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Trading Performance Metrics */}
                      {metrics && (
                        <div>
                          <h4 className="font-semibold mb-4">Метрики торговой эффективности</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <MetricCard 
                              title="Sharpe Ratio" 
                              value={formatDecimal(metrics.sharpe_ratio)} 
                            />
                            <MetricCard 
                              title="Win Rate" 
                              value={formatPercent(metrics.win_rate)} 
                            />
                            <MetricCard 
                              title="Total PnL" 
                              value={metrics.total_pnl !== null ? metrics.total_pnl.toFixed(2) : 'N/A'}
                              className={metrics.total_pnl !== null ? (metrics.total_pnl >= 0 ? 'border-green-500' : 'border-red-500') : ''}
                            />
                            <MetricCard 
                              title="Profit Factor" 
                              value={formatDecimal(metrics.profit_factor)} 
                            />
                            <MetricCard 
                              title="Avg PnL" 
                              value={metrics.avg_pnl !== null ? metrics.avg_pnl.toFixed(2) : 'N/A'}
                              className={metrics.avg_pnl !== null ? (metrics.avg_pnl >= 0 ? 'border-green-500' : 'border-red-500') : ''}
                            />
                            <MetricCard 
                              title="Max Drawdown" 
                              value={formatDecimal(metrics.max_drawdown)} 
                              className="border-red-500"
                            />
                          </div>
                        </div>
                      )}

                      {/* All Models Link */}
                      {modelsData.models.length > 1 && (
                        <div className="pt-4 border-t">
                          <p className="text-sm text-muted-foreground mb-2">
                            Всего моделей, обученных на этом датасете: {modelsData.models.length}
                          </p>
                          <Button variant="outline" size="sm" asChild>
                            <Link to={`/models`}>
                              Показать все модели
                            </Link>
                          </Button>
                        </div>
                      )}
                    </>
                  )
                })()}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Dataset Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Split Strategy */}
        <Card>
          <CardHeader>
            <CardTitle>Стратегия разбиения</CardTitle>
            <CardDescription>Метод разделения данных на сплиты</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <span className="text-sm text-muted-foreground">Тип стратегии:</span>
                <div className="font-medium mt-1">{getSplitStrategyLabel(data.split_strategy)}</div>
              </div>
              {data.split_strategy === 'walk_forward' && data.walk_forward_config && (
                <div className="mt-4 p-4 bg-muted rounded-md">
                  <h4 className="font-semibold mb-2">Walk-forward конфигурация:</h4>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-muted-foreground">Окно обучения:</span>{' '}
                      <span className="font-medium">{data.walk_forward_config.train_window_days} дней</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Окно валидации:</span>{' '}
                      <span className="font-medium">{data.walk_forward_config.validation_window_days} дней</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Окно теста:</span>{' '}
                      <span className="font-medium">{data.walk_forward_config.test_window_days} дней</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Шаг:</span>{' '}
                      <span className="font-medium">{data.walk_forward_config.step_days} дней</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Период:</span>{' '}
                      <span className="font-medium">
                        {formatDateShort(data.walk_forward_config.start_date)} - {formatDateShort(data.walk_forward_config.end_date)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Target Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Конфигурация таргета</CardTitle>
            <CardDescription>Настройки целевой переменной для обучения</CardDescription>
          </CardHeader>
          <CardContent>
            {data.target_config && data.target_config.type ? (
              <div className="space-y-4">
                <div>
                  <span className="text-sm text-muted-foreground">Тип таргета:</span>
                  <div className="font-medium mt-1">
                    <Badge variant="outline">{data.target_config.type}</Badge>
                  </div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Горизонт предсказания:</span>
                  <div className="font-medium mt-1">{data.target_config.horizon} секунд</div>
                </div>
                {data.target_config.threshold !== null && data.target_config.threshold !== undefined && (
                  <div>
                    <span className="text-sm text-muted-foreground">Порог для классификации:</span>
                    <div className="font-medium mt-1">{data.target_config.threshold}</div>
                  </div>
                )}
                {data.target_config.computation && (
                  <div className="mt-4 p-4 bg-muted rounded-md">
                    <h4 className="font-semibold mb-2">Метод вычисления:</h4>
                    <div className="space-y-2 text-sm">
                      <div>
                        <span className="text-muted-foreground">Preset:</span>{' '}
                        <span className="font-medium">{data.target_config.computation.preset}</span>
                      </div>
                      {data.target_config.computation.options && Object.keys(data.target_config.computation.options).length > 0 && (
                        <div>
                          <span className="text-muted-foreground">Опции:</span>
                          <div className="mt-1 font-mono text-xs">
                            {JSON.stringify(data.target_config.computation.options, null, 2)}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-2">
                {data.target_registry_version ? (
                  <>
                    <div className="text-sm text-muted-foreground">
                      Конфигурация таргета не загружена из Target Registry
                    </div>
                    <div>
                      <span className="text-sm text-muted-foreground">Версия Target Registry:</span>
                      <div className="font-mono text-sm mt-1">{data.target_registry_version}</div>
                    </div>
                    <div className="text-xs text-muted-foreground mt-2">
                      Конфигурация должна быть загружена из Target Registry по версии {data.target_registry_version}
                    </div>
                  </>
                ) : (
                  <div className="text-muted-foreground">Конфигурация таргета не доступна</div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Versions and Metadata */}
      <Card>
        <CardHeader>
          <CardTitle>Версии и метаданные</CardTitle>
          <CardDescription>Версии реестров и дополнительная информация</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-4">Версии реестров</h4>
              <div className="space-y-3">
                <div>
                  <span className="text-sm text-muted-foreground">Версия фич:</span>
                  <div className="font-mono text-sm mt-1">{data.feature_registry_version}</div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Версия таргета:</span>
                  <div className="font-mono text-sm mt-1">{data.target_registry_version || 'N/A'}</div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Формат и хранение</h4>
              <div className="space-y-3">
                <div>
                  <span className="text-sm text-muted-foreground">Формат вывода:</span>
                  <div className="font-medium mt-1">
                    <Badge variant="outline">{data.output_format}</Badge>
                  </div>
                </div>
                {data.storage_path && (
                  <div>
                    <span className="text-sm text-muted-foreground">Путь хранения:</span>
                    <div className="font-mono text-xs mt-1 break-all">{data.storage_path}</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Timeline */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Временная шкала
          </CardTitle>
          <CardDescription>Даты создания и завершения датасета</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Событие</TableHead>
                <TableHead>Дата и время</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="font-medium">Создан</TableCell>
                <TableCell>{formatDate(data.created_at)}</TableCell>
              </TableRow>
              {data.completed_at && (
                <TableRow>
                  <TableCell className="font-medium">Завершен</TableCell>
                  <TableCell>{formatDate(data.completed_at)}</TableCell>
                </TableRow>
              )}
              {data.estimated_completion && (
                <TableRow>
                  <TableCell className="font-medium">Ожидаемое завершение</TableCell>
                  <TableCell>{formatDate(data.estimated_completion)}</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Error Message (if failed) */}
      {data.status === 'failed' && data.error_message && (
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">Ошибка сборки датасета</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="p-4 bg-destructive/10 rounded-md">
              <pre className="text-sm whitespace-pre-wrap">{data.error_message}</pre>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Insights */}
      <Card>
        <CardHeader>
          <CardTitle>Инсайты и рекомендации</CardTitle>
          <CardDescription>Автоматический анализ датасета</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {totalRecords === 0 && (
              <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-md">
                <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                  ⚠️ Датасет не содержит записей. Возможно, он еще собирается или произошла ошибка.
                </p>
              </div>
            )}
            {totalRecords > 0 && (
              <>
                <div>
                  <h4 className="font-semibold mb-2">Распределение сплитов:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {barChartData.map((item) => (
                      <li key={item.split}>
                        <strong>{item.split}</strong>: {item.records.toLocaleString()} записей ({item.percentage}%)
                      </li>
                    ))}
                  </ul>
                </div>
                {data.target_config && (
                  <div>
                    <h4 className="font-semibold mb-2">Информация о таргете:</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                      <li>
                        Тип: <strong>{data.target_config.type}</strong> - 
                        {data.target_config.type === 'classification' && ' задача классификации'}
                        {data.target_config.type === 'regression' && ' задача регрессии'}
                        {data.target_config.type === 'risk_adjusted' && ' задача с учетом риска'}
                      </li>
                      <li>
                        Горизонт предсказания: <strong>{data.target_config.horizon} секунд</strong>
                        {data.target_config.horizon >= 3600 && ' (≥1 час)'}
                        {data.target_config.horizon < 3600 && data.target_config.horizon >= 60 && ' (<1 час)'}
                        {data.target_config.horizon < 60 && ' (<1 минута)'}
                      </li>
                      {data.target_config.computation && (
                        <li>
                          Метод вычисления: <strong>{data.target_config.computation.preset}</strong>
                        </li>
                      )}
                    </ul>
                  </div>
                )}
                {data.split_strategy === 'walk_forward' && (
                  <div>
                    <h4 className="font-semibold mb-2">Walk-forward валидация:</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                      <li>Используется метод walk-forward для временной валидации</li>
                      <li>Это позволяет проверить модель на разных временных окнах</li>
                      <li>Каждое окно содержит train/validation/test сплиты</li>
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

