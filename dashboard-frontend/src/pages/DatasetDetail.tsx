import { useParams, useNavigate, Link } from 'react-router-dom'
import { useDataset, usePreviousDataset } from '@/hooks/useDatasets'
import { useModelsByDataset } from '@/hooks/useModels'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts'
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
  const { data: previousDataset } = usePreviousDataset(data?.symbol, data?.strategy_id, id || '')
  const { data: previousModelsData } = useModelsByDataset(previousDataset?.id || '')

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

      {/* Target Distribution Statistics (for regression) */}
      {data.target_config?.type === 'regression' && data.split_statistics && (
        <Card>
          <CardHeader>
            <CardTitle>Статистика распределения таргета по сплитам</CardTitle>
            <CardDescription>
              Детальная статистика распределения целевой переменной для каждого сплита (для регрессии)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {(['train', 'validation', 'test'] as const).map((splitName) => {
                const splitStats = data.split_statistics?.[splitName]
                const targetStats = splitStats?.target_statistics
                const recordsKey = `${splitName}_records` as 'train_records' | 'validation_records' | 'test_records'
                const records = data[recordsKey] || 0

                if (!targetStats) return null

                // Calculate additional statistics
                const cv = targetStats.std !== 0 && targetStats.mean !== 0
                  ? (targetStats.std / Math.abs(targetStats.mean)) * 100
                  : 0
                const range = targetStats.max - targetStats.min
                const iqr_approx = targetStats.std * 1.35 // Approximate IQR from std (for normal distribution)

                return (
                  <Card key={splitName} className="border-2">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg capitalize">
                        {splitName === 'train' ? 'Train' : splitName === 'validation' ? 'Validation' : 'Test'}
                      </CardTitle>
                      <CardDescription>
                        {records.toLocaleString()} записей
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-muted-foreground">Среднее (μ):</span>
                          <div className="font-mono font-semibold">{targetStats.mean.toFixed(6)}</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Медиана:</span>
                          <div className="font-mono font-semibold">{targetStats.median.toFixed(6)}</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Стд. откл. (σ):</span>
                          <div className="font-mono font-semibold">{targetStats.std.toFixed(6)}</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">CV (%):</span>
                          <div className="font-mono font-semibold">{cv.toFixed(2)}%</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Минимум:</span>
                          <div className="font-mono text-xs">{targetStats.min.toFixed(6)}</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Максимум:</span>
                          <div className="font-mono text-xs">{targetStats.max.toFixed(6)}</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Размах:</span>
                          <div className="font-mono text-xs">{range.toFixed(6)}</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Количество:</span>
                          <div className="font-mono">{targetStats.count.toLocaleString()}</div>
                        </div>
                      </div>

                      {/* Target Quality Metrics */}
                      {(targetStats.zero_targets_count !== undefined || targetStats.positive_count !== undefined) && (
                        <div className="mt-4 pt-4 border-t">
                          <h5 className="font-semibold text-sm mb-2">Качество таргета</h5>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            {targetStats.zero_targets_count !== undefined && (
                              <>
                                <div>
                                  <span className="text-muted-foreground">Нулевых значений:</span>
                                  <div className={`font-mono font-semibold ${targetStats.zero_targets_percentage && targetStats.zero_targets_percentage > 15 ? 'text-yellow-600' : ''}`}>
                                    {targetStats.zero_targets_count.toLocaleString()}
                                    {targetStats.zero_targets_percentage !== undefined && (
                                      <span className="text-xs ml-1">
                                        ({targetStats.zero_targets_percentage.toFixed(2)}%)
                                        {targetStats.zero_targets_percentage > 15 && ' ⚠️'}
                                      </span>
                                    )}
                                  </div>
                                </div>
                                {targetStats.near_zero_count !== undefined && (
                                  <div>
                                    <span className="text-muted-foreground">Near-zero (&lt;1e-6):</span>
                                    <div className="font-mono text-xs">
                                      {targetStats.near_zero_count.toLocaleString()}
                                      {targetStats.near_zero_percentage !== undefined && (
                                        <span className="ml-1">({targetStats.near_zero_percentage.toFixed(2)}%)</span>
                                      )}
                                    </div>
                                  </div>
                                )}
                              </>
                            )}
                            {targetStats.positive_count !== undefined && (
                              <div>
                                <span className="text-muted-foreground">Положительных:</span>
                                <div className="font-mono text-xs text-green-600">
                                  {targetStats.positive_count.toLocaleString()}
                                  {targetStats.count > 0 && (
                                    <span className="ml-1">
                                      ({((targetStats.positive_count / targetStats.count) * 100).toFixed(2)}%)
                                    </span>
                                  )}
                                </div>
                              </div>
                            )}
                            {targetStats.negative_count !== undefined && (
                              <div>
                                <span className="text-muted-foreground">Отрицательных:</span>
                                <div className="font-mono text-xs text-red-600">
                                  {targetStats.negative_count.toLocaleString()}
                                  {targetStats.count > 0 && (
                                    <span className="ml-1">
                                      ({((targetStats.negative_count / targetStats.count) * 100).toFixed(2)}%)
                                    </span>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                          
                          {/* Consecutive zeros warning */}
                          {targetStats.consecutive_zeros && targetStats.consecutive_zeros.sequences_count && targetStats.consecutive_zeros.sequences_count > 0 && (
                            <div className="mt-3 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded-md">
                              <div className="text-xs text-yellow-800 dark:text-yellow-200">
                                <strong>⚠️ Последовательные нули:</strong> {targetStats.consecutive_zeros.sequences_count} последовательностей
                                {targetStats.consecutive_zeros.max_consecutive_length && (
                                  <span>, максимум {targetStats.consecutive_zeros.max_consecutive_length} подряд</span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Percentiles */}
                      {targetStats.percentiles && (
                        <div className="mt-4 pt-4 border-t">
                          <h5 className="font-semibold text-sm mb-2">Процентили</h5>
                          <div className="grid grid-cols-3 gap-2 text-xs">
                            {targetStats.percentiles.p1 !== undefined && (
                              <div>
                                <span className="text-muted-foreground">P1:</span>
                                <div className="font-mono">{targetStats.percentiles.p1.toFixed(6)}</div>
                              </div>
                            )}
                            {targetStats.percentiles.p5 !== undefined && (
                              <div>
                                <span className="text-muted-foreground">P5:</span>
                                <div className="font-mono">{targetStats.percentiles.p5.toFixed(6)}</div>
                              </div>
                            )}
                            {targetStats.percentiles.p10 !== undefined && (
                              <div>
                                <span className="text-muted-foreground">P10:</span>
                                <div className="font-mono">{targetStats.percentiles.p10.toFixed(6)}</div>
                              </div>
                            )}
                            {targetStats.percentiles.p25 !== undefined && (
                              <div>
                                <span className="text-muted-foreground">P25:</span>
                                <div className="font-mono">{targetStats.percentiles.p25.toFixed(6)}</div>
                              </div>
                            )}
                            {targetStats.percentiles.p50 !== undefined && (
                              <div>
                                <span className="text-muted-foreground">P50:</span>
                                <div className="font-mono font-semibold">{targetStats.percentiles.p50.toFixed(6)}</div>
                              </div>
                            )}
                            {targetStats.percentiles.p75 !== undefined && (
                              <div>
                                <span className="text-muted-foreground">P75:</span>
                                <div className="font-mono">{targetStats.percentiles.p75.toFixed(6)}</div>
                              </div>
                            )}
                            {targetStats.percentiles.p90 !== undefined && (
                              <div>
                                <span className="text-muted-foreground">P90:</span>
                                <div className="font-mono">{targetStats.percentiles.p90.toFixed(6)}</div>
                              </div>
                            )}
                            {targetStats.percentiles.p95 !== undefined && (
                              <div>
                                <span className="text-muted-foreground">P95:</span>
                                <div className="font-mono">{targetStats.percentiles.p95.toFixed(6)}</div>
                              </div>
                            )}
                            {targetStats.percentiles.p99 !== undefined && (
                              <div>
                                <span className="text-muted-foreground">P99:</span>
                                <div className="font-mono">{targetStats.percentiles.p99.toFixed(6)}</div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )
              })}
            </div>

            {/* Comparison Chart */}
            {(['train', 'validation', 'test'] as const).some(splitName =>
              data.split_statistics?.[splitName]?.target_statistics
            ) && (
                <div className="mt-6 space-y-6">
                  <div>
                    <h4 className="text-lg font-semibold mb-4">Сравнение статистики между сплитами</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart
                        data={(['train', 'validation', 'test'] as const)
                          .map(splitName => {
                            const stats = data.split_statistics?.[splitName]?.target_statistics
                            if (!stats) return null
                            return {
                              split: splitName === 'train' ? 'Train' : splitName === 'validation' ? 'Validation' : 'Test',
                              mean: stats.mean,
                              median: stats.median,
                              std: stats.std,
                              min: stats.min,
                              max: stats.max,
                            }
                          })
                          .filter(Boolean)
                        }
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="split" />
                        <YAxis />
                        <Tooltip
                          formatter={(value: number) => value.toFixed(6)}
                          labelFormatter={(label) => `Сплит: ${label}`}
                        />
                        <Legend />
                        <Bar dataKey="mean" fill="#8884d8" name="Среднее (μ)" />
                        <Bar dataKey="median" fill="#82ca9d" name="Медиана" />
                        <Bar dataKey="std" fill="#ffc658" name="Стд. откл. (σ)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Target Quality Comparison Chart */}
                  {(['train', 'validation', 'test'] as const).some(splitName =>
                    data.split_statistics?.[splitName]?.target_statistics?.zero_targets_percentage !== undefined
                  ) && (
                      <div>
                        <h4 className="text-lg font-semibold mb-4">Качество таргета: распределение значений</h4>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart
                            data={(['train', 'validation', 'test'] as const)
                              .map(splitName => {
                                const stats = data.split_statistics?.[splitName]?.target_statistics
                                if (!stats) return null
                                const total = stats.count || 0
                                return {
                                  split: splitName === 'train' ? 'Train' : splitName === 'validation' ? 'Validation' : 'Test',
                                  zero: stats.zero_targets_count || 0,
                                  zeroPercent: stats.zero_targets_percentage || 0,
                                  positive: stats.positive_count || 0,
                                  positivePercent: total > 0 ? ((stats.positive_count || 0) / total * 100) : 0,
                                  negative: stats.negative_count || 0,
                                  negativePercent: total > 0 ? ((stats.negative_count || 0) / total * 100) : 0,
                                }
                              })
                              .filter(Boolean)
                            }
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="split" />
                            <YAxis label={{ value: 'Процент', angle: -90, position: 'insideLeft' }} />
                            <Tooltip
                              formatter={(value: number, name: string) => {
                                if (name === 'zeroPercent' || name === 'positivePercent' || name === 'negativePercent') {
                                  return `${value.toFixed(2)}%`
                                }
                                return value.toLocaleString()
                              }}
                              labelFormatter={(label) => `Сплит: ${label}`}
                            />
                            <Legend />
                            <Bar dataKey="zeroPercent" fill="#fbbf24" name="Нулевых (%)" />
                            <Bar dataKey="positivePercent" fill="#10b981" name="Положительных (%)" />
                            <Bar dataKey="negativePercent" fill="#ef4444" name="Отрицательных (%)" />
                          </BarChart>
                        </ResponsiveContainer>
                        <div className="mt-2 text-xs text-muted-foreground text-center">
                          * Процентное распределение значений таргета по типам (нулевые, положительные, отрицательные)
                        </div>
                      </div>
                    )}
                </div>
              )}
          </CardContent>
        </Card>
      )}

      {/* Outlier Detection and Clipping */}
      {data.target_config?.type === 'regression' && data.split_statistics?.outlier_detection && (
        <Card className="border-blue-500 bg-blue-50 dark:bg-blue-900/20">
          <CardHeader>
            <CardTitle className="text-blue-800 dark:text-blue-200">
              🔍 Обработка выбросов (Outlier Detection & Clipping)
            </CardTitle>
            <CardDescription className="text-blue-700 dark:text-blue-300">
              Информация о примененной обработке выбросов: обнаружение и подрезка экстремальных значений таргета
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <span className="text-sm text-muted-foreground">Метод:</span>
                  <div className="font-mono font-semibold text-blue-800 dark:text-blue-200">
                    {data.split_statistics.outlier_detection.method}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Порог (3σ):</span>
                  <div className="font-mono font-semibold text-blue-800 dark:text-blue-200">
                    {data.split_statistics.outlier_detection.threshold.toFixed(6)}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Всего выбросов:</span>
                  <div className="font-mono font-semibold text-blue-800 dark:text-blue-200">
                    {data.split_statistics.outlier_detection.total_outliers_detected.toLocaleString()}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
                <div>
                  <span className="text-sm text-muted-foreground">Train Mean (μ):</span>
                  <div className="font-mono text-sm">
                    {data.split_statistics.outlier_detection.train_mean.toFixed(6)}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Train Std (σ):</span>
                  <div className="font-mono text-sm">
                    {data.split_statistics.outlier_detection.train_std.toFixed(6)}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Нижняя граница:</span>
                  <div className="font-mono text-sm">
                    {data.split_statistics.outlier_detection.lower_bound.toFixed(6)}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-muted-foreground">Верхняя граница:</span>
                  <div className="font-mono text-sm">
                    {data.split_statistics.outlier_detection.upper_bound.toFixed(6)}
                  </div>
                </div>
              </div>

              <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-900/30 rounded-md border border-blue-300 dark:border-blue-700">
                <div className="text-xs text-blue-800 dark:text-blue-200">
                  <strong>ℹ️ Как это работает:</strong> Порог 3σ вычисляется на train сплите (mean ± 3×std).
                  Значения таргета, выходящие за эти границы, помечаются как выбросы (is_outlier = 1) и подрезаются
                  до границ [mean - 3σ, mean + 3σ]. Это помогает стабилизировать обучение модели, сохраняя информацию
                  о выбросах через флаг is_outlier, который может использоваться моделью как фича.
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Data Quality - Problematic Periods */}
      {data.data_quality && data.data_quality.problematic_periods_excluded && data.data_quality.problematic_periods_excluded > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Качество данных</CardTitle>
            <CardDescription>
              Периоды с проблемными данными (идентичные OHLC), которые были исключены из датасета.
              Эти периоды могут указывать на проблемы с качеством исторических данных от биржи.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-muted-foreground">Исключено периодов:</span>
                  <div className="text-2xl font-bold text-yellow-600">
                    {data.data_quality.problematic_periods_excluded}
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  {data.data_quality.problematic_periods && data.data_quality.problematic_periods.length > 0 && (
                    <span>Показано первых {Math.min(data.data_quality.problematic_periods.length, data.data_quality.problematic_periods_excluded)}</span>
                  )}
                </div>
              </div>

              {data.data_quality.problematic_periods && data.data_quality.problematic_periods.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-sm font-semibold mb-3">Детали проблемных периодов:</h4>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {data.data_quality.problematic_periods.map((period, index) => {
                      const startDate = format(parseISO(period.start), 'dd.MM.yyyy HH:mm:ss')
                      const endDate = format(parseISO(period.end), 'dd.MM.yyyy HH:mm:ss')
                      const priceSame = Math.abs(period.current_price - period.future_price) < 1e-6
                      
                      return (
                        <div
                          key={index}
                          className="p-3 border rounded-lg bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800"
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
                                Период #{index + 1}
                              </div>
                              <div className="text-xs text-yellow-700 dark:text-yellow-300 mt-1">
                                <div>Начало: {startDate}</div>
                                <div>Конец: {endDate}</div>
                              </div>
                            </div>
                            <div className="text-right text-xs">
                              <div className="text-yellow-800 dark:text-yellow-200">
                                <div>Цена: {period.current_price.toFixed(2)}</div>
                                {!priceSame && (
                                  <div className="text-yellow-600 dark:text-yellow-400">
                                    → {period.future_price.toFixed(2)}
                                  </div>
                                )}
                                {priceSame && (
                                  <div className="text-red-600 dark:text-red-400 font-semibold">
                                    → {period.future_price.toFixed(2)} (одинаковая)
                                  </div>
                                )}
                              </div>
                              <div className="text-yellow-600 dark:text-yellow-400 mt-1">
                                <div>Объем: {period.current_volume.toFixed(2)}</div>
                                <div>→ {period.future_volume.toFixed(2)}</div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-200 dark:border-blue-800">
                <div className="text-xs text-blue-800 dark:text-blue-200">
                  <strong>ℹ️ Примечание:</strong> Эти периоды были автоматически исключены из датасета,
                  так как данные показывают идентичные OHLC значения (open = high = low = close),
                  что указывает на проблемы с качеством исторических данных от биржи.
                  Исключение таких периодов помогает улучшить качество датасета и снизить количество нулевых таргетов.
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Feature Correlations */}
      {data.feature_correlations && Object.keys(data.feature_correlations).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Важность фичей (Корреляция с таргетом)</CardTitle>
            <CardDescription>
              Топ-20 фичей с наивысшей абсолютной корреляцией с целевой переменной.
              Показывает линейную зависимость между фичей и таргетом.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[500px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={Object.entries(data.feature_correlations)
                    .map(([feature, correlation]) => ({
                      feature,
                      correlation,
                      absCorrelation: Math.abs(correlation)
                    }))
                    .sort((a, b) => b.absCorrelation - a.absCorrelation)
                    .slice(0, 20)}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[-1, 1]} />
                  <YAxis type="category" dataKey="feature" width={200} tick={{ fontSize: 12 }} />
                  <Tooltip
                    formatter={(value: number) => value.toFixed(4)}
                    labelStyle={{ color: 'black' }}
                  />
                  <Legend />
                  <Bar dataKey="correlation" name="Корреляция" fill="#8884d8">
                    {Object.entries(data.feature_correlations)
                      .map(([feature, correlation]) => ({ feature, correlation, abs: Math.abs(correlation) }))
                      .sort((a, b) => b.abs - a.abs)
                      .slice(0, 20)
                      .map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.correlation >= 0 ? '#82ca9d' : '#ef4444'} />
                      ))
                    }
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 text-sm text-muted-foreground text-center">
              * Зеленый цвет: положительная корреляция (цена растет вместе с фичей).
              Красный цвет: отрицательная корреляция (цена падает при росте фичи).
            </div>
          </CardContent>
        </Card>
      )}

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
                  {data.target_config?.type === 'classification' || data.target_config?.type === 'risk_adjusted' ? (
                    <>
                      <TableHead>Распределение классов</TableHead>
                      <TableHead>Баланс классов</TableHead>
                    </>
                  ) : (
                    <>
                      <TableHead>Среднее (μ)</TableHead>
                      <TableHead>Медиана</TableHead>
                      <TableHead>Стд. откл. (σ)</TableHead>
                      <TableHead>Размах</TableHead>
                      <TableHead>Нулевых (%)</TableHead>
                      <TableHead>Положительных</TableHead>
                      <TableHead>Отрицательных</TableHead>
                    </>
                  )}
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
                  const targetStats = splitStats?.target_statistics

                  return (
                    <TableRow key={splitName}>
                      <TableCell className="font-medium capitalize">{splitName === 'train' ? 'Train' : splitName === 'validation' ? 'Validation' : 'Test'}</TableCell>
                      <TableCell>{records.toLocaleString()}</TableCell>
                      <TableCell>
                        {totalRecords > 0
                          ? `${(records / totalRecords * 100).toFixed(2)}%`
                          : '0%'}
                      </TableCell>
                      {isClassification ? (
                        <>
                          {/* Classification columns */}
                          <TableCell className="text-xs" title={splitStats?.class_distribution ? Object.entries(splitStats.class_distribution).map(([cls, count]) => `${cls}: ${count.toLocaleString()}`).join(', ') : ''}>
                            {splitStats?.class_distribution
                              ? (Object.keys(splitStats.class_distribution).length <= 3
                                ? Object.entries(splitStats.class_distribution).map(([cls, count]) => `${cls}: ${count.toLocaleString()}`).join(', ')
                                : `${Object.keys(splitStats.class_distribution).length} классов`)
                              : 'N/A'}
                          </TableCell>
                          <TableCell className="text-xs">
                            {splitStats?.class_balance_ratio !== undefined ? (
                              <span className={splitStats.class_balance_ratio < 0.3 ? 'text-yellow-600 font-medium' : splitStats.class_balance_ratio < 0.5 ? 'text-orange-600 font-medium' : ''}>
                                {(splitStats.class_balance_ratio * 100).toFixed(1)}%
                                {splitStats.class_balance_ratio < 0.3 && ' ⚠️'}
                              </span>
                            ) : 'N/A'}
                          </TableCell>
                        </>
                      ) : (
                        <>
                          {/* Regression columns */}
                          <TableCell className="text-xs font-mono" title={targetStats ? `Mean: ${targetStats.mean.toFixed(6)}` : ''}>
                            {targetStats ? targetStats.mean.toFixed(6) : 'N/A'}
                          </TableCell>
                          <TableCell className="text-xs font-mono" title={targetStats ? `Median: ${targetStats.median.toFixed(6)}` : ''}>
                            {targetStats ? targetStats.median.toFixed(6) : 'N/A'}
                          </TableCell>
                          <TableCell className="text-xs font-mono" title={targetStats ? `Std: ${targetStats.std.toFixed(6)}` : ''}>
                            {targetStats ? targetStats.std.toFixed(6) : 'N/A'}
                          </TableCell>
                          <TableCell className="text-xs font-mono" title={targetStats ? `Range: ${(targetStats.max - targetStats.min).toFixed(6)}` : ''}>
                            {targetStats ? (targetStats.max - targetStats.min).toFixed(6) : 'N/A'}
                          </TableCell>
                          <TableCell className="text-xs" title={targetStats && targetStats.zero_targets_count !== undefined ? `${targetStats.zero_targets_count} нулевых значений` : ''}>
                            {targetStats && targetStats.zero_targets_count !== undefined ? (
                              <span className={targetStats.zero_targets_percentage && targetStats.zero_targets_percentage > 15 ? 'text-yellow-600 font-medium' : ''}>
                                {targetStats.zero_targets_count.toLocaleString()}
                                {targetStats.zero_targets_percentage !== undefined && (
                                  <span className="ml-1">
                                    ({targetStats.zero_targets_percentage.toFixed(1)}%)
                                    {targetStats.zero_targets_percentage > 15 && ' ⚠️'}
                                  </span>
                                )}
                              </span>
                            ) : 'N/A'}
                          </TableCell>
                          <TableCell className="text-xs text-green-600">
                            {targetStats && targetStats.positive_count !== undefined ? (
                              <>
                                {targetStats.positive_count.toLocaleString()}
                                {targetStats.count > 0 && (
                                  <span className="ml-1 text-muted-foreground">
                                    ({((targetStats.positive_count / targetStats.count) * 100).toFixed(1)}%)
                                  </span>
                                )}
                              </>
                            ) : 'N/A'}
                          </TableCell>
                          <TableCell className="text-xs text-red-600">
                            {targetStats && targetStats.negative_count !== undefined ? (
                              <>
                                {targetStats.negative_count.toLocaleString()}
                                {targetStats.count > 0 && (
                                  <span className="ml-1 text-muted-foreground">
                                    ({((targetStats.negative_count / targetStats.count) * 100).toFixed(1)}%)
                                  </span>
                                )}
                              </>
                            ) : 'N/A'}
                          </TableCell>
                        </>
                      )}
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

                  // Get previous model metrics if available
                  const previousModel = previousModelsData?.models?.find(m => m.is_active) || previousModelsData?.models?.[0]
                  const previousMetrics = previousModel?.metrics || null

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
                            <MetricCard title="Accuracy" value={formatPercent(metrics.accuracy)} currentValue={metrics.accuracy} previousValue={previousMetrics?.accuracy} isHigherBetter={true} />
                            <MetricCard title="Precision" value={formatPercent(metrics.precision)} currentValue={metrics.precision} previousValue={previousMetrics?.precision} isHigherBetter={true} />
                            <MetricCard title="Recall" value={formatPercent(metrics.recall)} currentValue={metrics.recall} previousValue={previousMetrics?.recall} isHigherBetter={true} />
                            <MetricCard title="F1 Score" value={formatPercent(metrics.f1_score)} currentValue={metrics.f1_score} previousValue={previousMetrics?.f1_score} isHigherBetter={true} />
                            <MetricCard title="Balanced Accuracy" value={formatPercent(metrics.balanced_accuracy)} currentValue={metrics.balanced_accuracy} previousValue={previousMetrics?.balanced_accuracy} isHigherBetter={true} />
                            <MetricCard title="ROC AUC" value={formatDecimal(metrics.roc_auc)} currentValue={metrics.roc_auc} previousValue={previousMetrics?.roc_auc} isHigherBetter={true} />
                            <MetricCard title="PR AUC" value={formatDecimal(metrics.pr_auc)} currentValue={metrics.pr_auc} previousValue={previousMetrics?.pr_auc} isHigherBetter={true} />
                          </div>
                        </div>
                      )}

                      {/* Regression Metrics */}
                      {!isClassification && metrics && (
                        <div>
                          <h4 className="font-semibold mb-4">Метрики регрессии</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <MetricCard title="MAE" value={formatDecimal(metrics.mae, 6)} currentValue={metrics.mae} previousValue={previousMetrics?.mae} isHigherBetter={false} />
                            <MetricCard title="RMSE" value={formatDecimal(metrics.rmse, 6)} currentValue={metrics.rmse} previousValue={previousMetrics?.rmse} isHigherBetter={false} />
                            <MetricCard title="R² Score" value={formatDecimal(metrics.r2_score, 4)} currentValue={metrics.r2_score} previousValue={previousMetrics?.r2_score} isHigherBetter={true} />
                            <MetricCard title="MSE" value={formatDecimal(metrics.mse, 6)} currentValue={metrics.mse} previousValue={previousMetrics?.mse} isHigherBetter={false} />
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
                                  {metrics.mae !== null && <MetricCard title="MAE" value={formatDecimal(metrics.mae, 6)} currentValue={metrics.mae} previousValue={previousMetrics?.mae} isHigherBetter={false} />}
                                  {metrics.rmse !== null && <MetricCard title="RMSE" value={formatDecimal(metrics.rmse, 6)} currentValue={metrics.rmse} previousValue={previousMetrics?.rmse} isHigherBetter={false} />}
                                  {metrics.r2_score !== null && <MetricCard title="R² Score" value={formatDecimal(metrics.r2_score, 4)} currentValue={metrics.r2_score} previousValue={previousMetrics?.r2_score} isHigherBetter={true} />}
                                  {metrics.mse !== null && <MetricCard title="MSE" value={formatDecimal(metrics.mse, 6)} currentValue={metrics.mse} previousValue={previousMetrics?.mse} isHigherBetter={false} />}
                                </>
                              )}
                              {!isClassification && (
                                <>
                                  {metrics.accuracy !== null && <MetricCard title="Accuracy" value={formatPercent(metrics.accuracy)} currentValue={metrics.accuracy} previousValue={previousMetrics?.accuracy} isHigherBetter={true} />}
                                  {metrics.precision !== null && <MetricCard title="Precision" value={formatPercent(metrics.precision)} currentValue={metrics.precision} previousValue={previousMetrics?.precision} isHigherBetter={true} />}
                                  {metrics.recall !== null && <MetricCard title="Recall" value={formatPercent(metrics.recall)} currentValue={metrics.recall} previousValue={previousMetrics?.recall} isHigherBetter={true} />}
                                  {metrics.f1_score !== null && <MetricCard title="F1 Score" value={formatPercent(metrics.f1_score)} currentValue={metrics.f1_score} previousValue={previousMetrics?.f1_score} isHigherBetter={true} />}
                                  {metrics.balanced_accuracy !== null && <MetricCard title="Balanced Accuracy" value={formatPercent(metrics.balanced_accuracy)} currentValue={metrics.balanced_accuracy} previousValue={previousMetrics?.balanced_accuracy} isHigherBetter={true} />}
                                  {metrics.roc_auc !== null && <MetricCard title="ROC AUC" value={formatDecimal(metrics.roc_auc)} currentValue={metrics.roc_auc} previousValue={previousMetrics?.roc_auc} isHigherBetter={true} />}
                                  {metrics.pr_auc !== null && <MetricCard title="PR AUC" value={formatDecimal(metrics.pr_auc)} currentValue={metrics.pr_auc} previousValue={previousMetrics?.pr_auc} isHigherBetter={true} />}
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
                              currentValue={metrics.sharpe_ratio}
                              previousValue={previousMetrics?.sharpe_ratio}
                              isHigherBetter={true}
                            />
                            <MetricCard
                              title="Win Rate"
                              value={formatPercent(metrics.win_rate)}
                              currentValue={metrics.win_rate}
                              previousValue={previousMetrics?.win_rate}
                              isHigherBetter={true}
                            />
                            <MetricCard
                              title="Total PnL"
                              value={metrics.total_pnl !== null ? metrics.total_pnl.toFixed(2) : 'N/A'}
                              className={metrics.total_pnl !== null ? (metrics.total_pnl >= 0 ? 'border-green-500' : 'border-red-500') : ''}
                              currentValue={metrics.total_pnl}
                              previousValue={previousMetrics?.total_pnl}
                              isHigherBetter={true}
                            />
                            <MetricCard
                              title="Profit Factor"
                              value={formatDecimal(metrics.profit_factor)}
                              currentValue={metrics.profit_factor}
                              previousValue={previousMetrics?.profit_factor}
                              isHigherBetter={true}
                            />
                            <MetricCard
                              title="Avg PnL"
                              value={metrics.avg_pnl !== null ? metrics.avg_pnl.toFixed(2) : 'N/A'}
                              className={metrics.avg_pnl !== null ? (metrics.avg_pnl >= 0 ? 'border-green-500' : 'border-red-500') : ''}
                              currentValue={metrics.avg_pnl}
                              previousValue={previousMetrics?.avg_pnl}
                              isHigherBetter={true}
                            />
                            <MetricCard
                              title="Max Drawdown"
                              value={formatDecimal(metrics.max_drawdown)}
                              className="border-red-500"
                              currentValue={metrics.max_drawdown}
                              previousValue={previousMetrics?.max_drawdown}
                              isHigherBetter={false}
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

