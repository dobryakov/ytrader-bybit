import { useParams, useNavigate, Link } from 'react-router-dom'
import { useModelAnalysis, usePredictionsData } from '@/hooks/useModels'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { MetricCard } from '@/components/metrics/MetricCard'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, ReferenceLine } from 'recharts'
import { ArrowLeft, TrendingUp, TrendingDown, Minus } from 'lucide-react'

export default function ModelDetail() {
  const { version } = useParams<{ version: string }>()
  const navigate = useNavigate()
  const { data, isLoading, error } = useModelAnalysis(version || '')
  const { data: predictionsData, isLoading: predictionsLoading } = usePredictionsData(version || '', 'test', 1000)

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
          <Button variant="outline" onClick={() => navigate('/models')} className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Назад к моделям
          </Button>
          <div className="text-center text-muted-foreground py-8">
            Ошибка загрузки данных модели. Версия: {version}
          </div>
        </div>
      </div>
    )
  }

  const formatPercent = (value: number | null) => {
    if (value === null || value === undefined) return 'N/A'
    return `${(value * 100).toFixed(2)}%`
  }

  const formatDecimal = (value: number | null, decimals: number = 4) => {
    if (value === null || value === undefined) return 'N/A'
    return value.toFixed(decimals)
  }

  const getComparisonIcon = (difference: number | null) => {
    if (difference === null || difference === undefined) return <Minus className="h-4 w-4 text-gray-400" />
    if (difference > 0) return <TrendingUp className="h-4 w-4 text-green-500" />
    if (difference < 0) return <TrendingDown className="h-4 w-4 text-red-500" />
    return <Minus className="h-4 w-4 text-gray-400" />
  }

  const getComparisonColor = (difference: number | null) => {
    if (difference === null || difference === undefined) return 'text-gray-500'
    if (difference > 0) return 'text-green-600'
    if (difference < 0) return 'text-red-600'
    return 'text-gray-500'
  }

  // Determine task type: regression if regression metrics exist, classification otherwise
  const isRegression = data.model_metrics.r2_score !== null && data.model_metrics.r2_score !== undefined ||
                       data.model_metrics.mse !== null && data.model_metrics.mse !== undefined ||
                       data.model_metrics.directional_accuracy !== null && data.model_metrics.directional_accuracy !== undefined

  // Prepare data for top-k chart (only for classification)
  const topKChartData = data.top_k_metrics.map((tk) => ({
    k: `Top-${tk.k}%`,
    pr_auc: tk.pr_auc ? tk.pr_auc * 100 : null,
    roc_auc: tk.roc_auc ? tk.roc_auc * 100 : null,
    accuracy: tk.accuracy ? tk.accuracy * 100 : null,
    lift: tk.lift ? tk.lift : null, // Lift is already a ratio (e.g., 1.2 = 20% improvement)
  }))

  // Prepare scatter plot data for regression
  const scatterPlotData = isRegression && predictionsData ? predictionsData.data_points
    .filter((dp) => dp.y_pred !== null && dp.y_pred !== undefined)
    .map((dp) => ({
      y_true: dp.y_true,
      y_pred: dp.y_pred,
    })) : []

  // Prepare error distribution data for regression
  const errorDistributionData = isRegression && predictionsData ? (() => {
    const errors = predictionsData.data_points
      .filter((dp) => dp.error !== null && dp.error !== undefined)
      .map((dp) => dp.error!)
    
    if (errors.length === 0) return []
    
    // Create histogram bins
    const minError = Math.min(...errors)
    const maxError = Math.max(...errors)
    const binCount = 20
    const binSize = (maxError - minError) / binCount
    
    const bins = Array(binCount).fill(0).map((_, i) => ({
      bin: minError + i * binSize,
      count: 0,
    }))
    
    errors.forEach((error) => {
      const binIndex = Math.min(Math.floor((error - minError) / binSize), binCount - 1)
      bins[binIndex].count++
    })
    
    return bins
  })() : []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Button variant="outline" onClick={() => navigate('/models')} className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Назад к моделям
          </Button>
          <h2 className="text-3xl font-bold tracking-tight">Детальный анализ модели</h2>
          <p className="text-muted-foreground">Версия: {data.model_version} (ID: {data.model_id.slice(0, 8)}...)</p>
        </div>
      </div>

      {/* Optimal Top-K Percentage Info - only for classification */}
      {!isRegression && data.optimal_top_k_percentage !== null && data.optimal_top_k_percentage !== undefined && (
        <Card className="border-green-500">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Оптимальный Top-K процент
              <Badge variant="default" className="bg-green-600">Автоматически подобран</Badge>
            </CardTitle>
            <CardDescription>
              Процент, автоматически выбранный для этой модели на основе метрик качества
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <div className="text-4xl font-bold text-green-600">
                  Top-{data.optimal_top_k_percentage}%
                </div>
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">
                    Этот процент был автоматически выбран при обучении модели
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Используется для определения порога confidence при генерации сигналов
                  </div>
                </div>
              </div>
              {data.top_k_metrics.find(tk => tk.k === data.optimal_top_k_percentage) && (
                <div className="border-t pt-4">
                  <h4 className="font-semibold mb-2 text-sm">Метрики для Top-{data.optimal_top_k_percentage}%:</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <div className="text-xs text-muted-foreground">Accuracy</div>
                      <div className="font-semibold">
                        {formatPercent(data.top_k_metrics.find(tk => tk.k === data.optimal_top_k_percentage)?.accuracy)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Lift</div>
                      <div className="font-semibold">
                        {data.top_k_metrics.find(tk => tk.k === data.optimal_top_k_percentage)?.lift 
                          ? `${data.top_k_metrics.find(tk => tk.k === data.optimal_top_k_percentage)!.lift!.toFixed(2)}x`
                          : 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">PR AUC</div>
                      <div className="font-semibold">
                        {formatDecimal(data.top_k_metrics.find(tk => tk.k === data.optimal_top_k_percentage)?.pr_auc)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Coverage</div>
                      <div className="font-semibold">
                        {formatPercent(data.top_k_metrics.find(tk => tk.k === data.optimal_top_k_percentage)?.coverage)}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Confidence Threshold Info - only for classification */}
      {!isRegression && data.confidence_threshold_info && (
        <Card>
          <CardHeader>
            <CardTitle>Порог уверенности (Confidence Threshold)</CardTitle>
            <CardDescription>Информация о пороге, используемом при генерации сигналов</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Значение порога:</span>
                    <span className="font-bold text-lg">{data.confidence_threshold_info.threshold_value.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Источник:</span>
                    <Badge variant={data.confidence_threshold_info.threshold_source === 'top_k' ? 'default' : 'secondary'}>
                      {data.confidence_threshold_info.threshold_source === 'top_k' ? 'Top-K анализ' : 'Статический'}
                    </Badge>
                  </div>
                </div>
                <div className="space-y-2">
                  {data.confidence_threshold_info.threshold_source === 'top_k' && (
                    <>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Используемый Top-K процент:</span>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">Top-{data.confidence_threshold_info.top_k_percentage}%</span>
                          {data.optimal_top_k_percentage === data.confidence_threshold_info.top_k_percentage ? (
                            <Badge variant="outline" className="border-green-500 text-green-600">
                              Оптимальный
                            </Badge>
                          ) : data.optimal_top_k_percentage !== null && data.optimal_top_k_percentage !== undefined ? (
                            <Badge variant="outline" className="border-yellow-500 text-yellow-600">
                              Из настроек
                            </Badge>
                          ) : null}
                        </div>
                      </div>
                      {data.optimal_top_k_percentage !== null && 
                       data.optimal_top_k_percentage !== undefined && 
                       data.optimal_top_k_percentage !== data.confidence_threshold_info.top_k_percentage && (
                        <div className="flex justify-between items-center text-xs text-muted-foreground">
                          <span>Оптимальный для модели:</span>
                          <span className="font-medium">Top-{data.optimal_top_k_percentage}%</span>
                        </div>
                      )}
                      {data.confidence_threshold_info.metric_name && (
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Метрика в БД:</span>
                          <span className="font-mono text-xs">{data.confidence_threshold_info.metric_name}</span>
                        </div>
                      )}
                    </>
                  )}
                  {data.confidence_threshold_info.threshold_source === 'static' && (
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Статический порог:</span>
                      <span className="font-medium">{data.confidence_threshold_info.static_threshold?.toFixed(4)}</span>
                    </div>
                  )}
                </div>
              </div>
              <div className="border-t pt-4">
                <h4 className="font-semibold mb-2 text-sm">Как это работает:</h4>
                <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                  <li>При генерации сигнала система получает активную модель из БД</li>
                  {data.confidence_threshold_info.threshold_source === 'top_k' ? (
                    <>
                      <li>
                        Система определяет, какой Top-K процент использовать:
                        {data.optimal_top_k_percentage !== null && data.optimal_top_k_percentage !== undefined
                          ? ` сначала проверяет optimal_top_k_percentage из training_config модели (${data.optimal_top_k_percentage}%),`
                          : ''}
                        {' '}если не найден — использует значение из настроек ({data.confidence_threshold_info.top_k_percentage}%)
                      </li>
                      <li>
                        Использует метрику {data.confidence_threshold_info.metric_name} из model_quality_metrics для получения порога confidence
                      </li>
                      <li>Порог найден и используется для фильтрации сигналов (confidence должен быть ≥ порога)</li>
                    </>
                  ) : (
                    <>
                      <li>
                        Ищет метрику top_k_{data.optimal_top_k_percentage || 'X'}_confidence_threshold в model_quality_metrics
                      </li>
                      <li>Метрика не найдена, используется статический порог из настроек</li>
                    </>
                  )}
                  <li>В логах видно: какой порог используется (top_k или static), значение порога, источник порога</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Predictions Info */}
      <Card>
        <CardHeader>
          <CardTitle>Сохранённые предсказания</CardTitle>
          <CardDescription>
            {isRegression 
              ? "Raw predictions (y_true и y_pred) для анализа регрессии"
              : "Raw probabilities и y_true для анализа классификации"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Split</TableHead>
                <TableHead>Количество</TableHead>
                <TableHead>Dataset ID</TableHead>
                <TableHead>Дата создания</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.predictions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} className="text-center text-muted-foreground">
                    Нет сохранённых предсказаний
                  </TableCell>
                </TableRow>
              ) : (
                data.predictions.map((pred, idx) => (
                  <TableRow key={idx}>
                    <TableCell>
                      <Badge variant="outline">{pred.split}</Badge>
                    </TableCell>
                    <TableCell className="font-medium">{pred.count.toLocaleString()}</TableCell>
                    <TableCell className="font-mono text-xs">
                      {pred.dataset_id ? (
                        <Link 
                          to={`/datasets/${pred.dataset_id}`}
                          className="text-primary hover:underline"
                        >
                          {pred.dataset_id.slice(0, 8)}...
                        </Link>
                      ) : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {pred.created_at ? format(parseISO(pred.created_at), 'dd.MM.yyyy HH:mm:ss') : 'N/A'}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Model Metrics vs Baseline */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Метрики модели</CardTitle>
            <CardDescription>Основные метрики качества на test split</CardDescription>
          </CardHeader>
          <CardContent>
            {isRegression ? (
              <div className="grid grid-cols-2 gap-4">
                <MetricCard title="R² Score" value={formatDecimal(data.model_metrics.r2_score)} />
                <MetricCard title="RMSE" value={formatDecimal(data.model_metrics.rmse)} />
                <MetricCard title="MAE" value={formatDecimal(data.model_metrics.mae)} />
                <MetricCard title="MSE" value={formatDecimal(data.model_metrics.mse)} />
                <MetricCard title="Directional Accuracy" value={formatPercent(data.model_metrics.directional_accuracy)} />
                <MetricCard title="Sharpe Ratio" value={formatDecimal(data.model_metrics.sharpe_ratio)} />
                <MetricCard title="Information Coefficient" value={formatDecimal(data.model_metrics.information_coefficient)} />
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4">
                <MetricCard title="Accuracy" value={formatPercent(data.model_metrics.accuracy)} />
                <MetricCard title="Precision" value={formatPercent(data.model_metrics.precision)} />
                <MetricCard title="Recall" value={formatPercent(data.model_metrics.recall)} />
                <MetricCard title="F1 Score" value={formatPercent(data.model_metrics.f1_score)} />
                <MetricCard title="Balanced Accuracy" value={formatPercent(data.model_metrics.balanced_accuracy)} />
                <MetricCard title="ROC AUC" value={formatDecimal(data.model_metrics.roc_auc)} />
                <MetricCard title="PR AUC" value={formatDecimal(data.model_metrics.pr_auc)} />
              </div>
            )}
          </CardContent>
        </Card>

        {!isRegression && (
          <Card>
            <CardHeader>
              <CardTitle>Baseline метрики</CardTitle>
              <CardDescription>Majority class strategy (всегда предсказывать большинство)</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <MetricCard title="Accuracy" value={formatPercent(data.baseline_metrics.accuracy)} />
                <MetricCard title="Precision" value={formatPercent(data.baseline_metrics.precision)} />
                <MetricCard title="Recall" value={formatPercent(data.baseline_metrics.recall)} />
                <MetricCard title="F1 Score" value={formatPercent(data.baseline_metrics.f1_score)} />
                <MetricCard title="Balanced Accuracy" value={formatPercent(data.baseline_metrics.balanced_accuracy)} />
                <MetricCard title="ROC AUC" value={formatDecimal(data.baseline_metrics.roc_auc)} />
                <MetricCard title="PR AUC" value={formatDecimal(data.baseline_metrics.pr_auc)} />
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Comparison - only for classification */}
      {!isRegression && data.comparison && Object.keys(data.comparison).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Сравнение с Baseline</CardTitle>
            <CardDescription>Разница между моделью и baseline стратегией</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(data.comparison).map(([metric, comp]) => (
                <Card key={metric}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium capitalize">{metric.replace('_', ' ')}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Модель:</span>
                        <span className="font-medium">{formatDecimal(comp.model)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Baseline:</span>
                        <span className="font-medium">{formatDecimal(comp.baseline)}</span>
                      </div>
                      <div className="flex justify-between items-center pt-2 border-t">
                        <span className="text-sm font-medium">Разница:</span>
                        <span className={`font-bold flex items-center gap-1 ${getComparisonColor(comp.difference)}`}>
                          {getComparisonIcon(comp.difference)}
                          {comp.difference !== null ? (comp.difference > 0 ? '+' : '') + formatDecimal(comp.difference) : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Top-K Metrics - only for classification */}
      {!isRegression && data.top_k_metrics && data.top_k_metrics.length > 0 && (
      <Card>
        <CardHeader>
          <CardTitle>Top-K% анализ</CardTitle>
          <CardDescription>
            Метрики для top-k% предсказаний, отсортированных по уверенности (без применения фильтров)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Top-K Chart */}
            <div className="border rounded-md p-4">
              <h4 className="text-lg font-semibold mb-4">График метрик по Top-K%</h4>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={topKChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="k" />
                  <YAxis />
                  <Tooltip formatter={(value: any) => value !== null ? `${value.toFixed(2)}%` : 'N/A'} />
                  <Legend />
                  <Bar dataKey="pr_auc" fill="#8884d8" name="PR AUC (%)" />
                  <Bar dataKey="roc_auc" fill="#82ca9d" name="ROC AUC (%)" />
                  <Bar dataKey="accuracy" fill="#ffc658" name="Accuracy (%)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Top-K Table */}
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>K</TableHead>
                    <TableHead>Coverage</TableHead>
                    <TableHead>Accuracy</TableHead>
                    <TableHead>Precision</TableHead>
                    <TableHead>Recall</TableHead>
                    <TableHead>F1 Score</TableHead>
                    <TableHead>Balanced Acc</TableHead>
                    <TableHead>ROC AUC</TableHead>
                    <TableHead>PR AUC</TableHead>
                    <TableHead>Lift</TableHead>
                    <TableHead>Precision (class 1)</TableHead>
                    <TableHead>Recall (class 1)</TableHead>
                    <TableHead>F1 (class 1)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.top_k_metrics.map((tk) => (
                    <TableRow key={tk.k}>
                      <TableCell className="font-medium">Top-{tk.k}%</TableCell>
                      <TableCell>{formatPercent(tk.coverage)}</TableCell>
                      <TableCell>{formatPercent(tk.accuracy)}</TableCell>
                      <TableCell>{formatPercent(tk.precision)}</TableCell>
                      <TableCell>{formatPercent(tk.recall)}</TableCell>
                      <TableCell>{formatPercent(tk.f1_score)}</TableCell>
                      <TableCell>{formatPercent(tk.balanced_accuracy)}</TableCell>
                      <TableCell>{formatDecimal(tk.roc_auc)}</TableCell>
                      <TableCell className="font-semibold">{formatDecimal(tk.pr_auc)}</TableCell>
                      <TableCell>{tk.lift !== null ? `${tk.lift.toFixed(2)}x` : 'N/A'}</TableCell>
                      <TableCell>{formatPercent(tk.precision_class_1)}</TableCell>
                      <TableCell>{formatPercent(tk.recall_class_1)}</TableCell>
                      <TableCell>{formatPercent(tk.f1_class_1)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {/* Top-K Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {data.top_k_metrics.map((tk) => (
                <Card key={tk.k} className="border-2">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Top-{tk.k}%</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">PR AUC:</span>
                        <span className="font-bold text-lg">{formatDecimal(tk.pr_auc)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">ROC AUC:</span>
                        <span className="font-medium">{formatDecimal(tk.roc_auc)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Accuracy:</span>
                        <span className="font-medium">{formatPercent(tk.accuracy)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Lift:</span>
                        <span className="font-medium">{tk.lift !== null ? `${tk.lift.toFixed(2)}x` : 'N/A'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Coverage:</span>
                        <span className="font-medium">{formatPercent(tk.coverage)}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
      )}

      {/* Detailed Metrics Comparison Table - only for classification */}
      {!isRegression && data.comparison && Object.keys(data.comparison).length > 0 && (
      <Card>
        <CardHeader>
          <CardTitle>Детальное сравнение метрик</CardTitle>
          <CardDescription>Полное сравнение всех метрик модели и baseline</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Метрика</TableHead>
                <TableHead>Модель</TableHead>
                <TableHead>Baseline</TableHead>
                <TableHead>Разница</TableHead>
                <TableHead>Улучшение</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(data.comparison).map(([metric, comp]) => (
                <TableRow key={metric}>
                  <TableCell className="font-medium capitalize">{metric.replace('_', ' ')}</TableCell>
                  <TableCell>{formatDecimal(comp.model)}</TableCell>
                  <TableCell>{formatDecimal(comp.baseline)}</TableCell>
                  <TableCell>
                    <span className={`flex items-center gap-1 ${getComparisonColor(comp.difference)}`}>
                      {getComparisonIcon(comp.difference)}
                      {comp.difference !== null ? (comp.difference > 0 ? '+' : '') + formatDecimal(comp.difference) : 'N/A'}
                    </span>
                  </TableCell>
                  <TableCell>
                    {comp.baseline !== null && comp.baseline !== 0 && comp.difference !== null ? (
                      <span className={getComparisonColor(comp.difference)}>
                        {((comp.difference / comp.baseline) * 100).toFixed(2)}%
                      </span>
                    ) : (
                      comp.difference !== null && comp.difference > 0 ? (
                        <span className="text-green-600">Модель лучше</span>
                      ) : comp.difference !== null && comp.difference < 0 ? (
                        <span className="text-red-600">Baseline лучше</span>
                      ) : (
                        'N/A'
                      )
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      )}

      {/* Regression Thresholds Info - only for regression */}
      {isRegression && data.regression_thresholds && (
        <Card>
          <CardHeader>
            <CardTitle>Пороги регрессии (Regression Thresholds)</CardTitle>
            <CardDescription>Пороги для конвертации предсказанного возврата в сигналы BUY/SELL/HOLD</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                <div>
                  <div className="text-sm font-medium text-muted-foreground">Метод</div>
                  <Badge variant={data.regression_thresholds.method === 'quantile' ? 'default' : 'secondary'}>
                    {data.regression_thresholds.method === 'quantile' ? 'Квантильный' : 'Фиксированный'}
                  </Badge>
                </div>
              </div>

              {data.regression_thresholds.method === 'quantile' && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 border rounded-lg">
                      <div className="text-sm font-medium text-muted-foreground mb-2">BUY порог</div>
                      <div className="space-y-1">
                        <div className="text-lg font-bold">
                          {data.regression_thresholds.buy_threshold_value !== null && data.regression_thresholds.buy_threshold_value !== undefined
                            ? formatDecimal(data.regression_thresholds.buy_threshold_value, 6)
                            : 'N/A'}
                        </div>
                        {data.regression_thresholds.buy_quantile !== null && (
                          <div className="text-xs text-muted-foreground">
                            Квантиль: {formatPercent(data.regression_thresholds.buy_quantile)} (Top {formatPercent(1 - (data.regression_thresholds.buy_quantile || 0))})
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <div className="text-sm font-medium text-muted-foreground mb-2">SELL порог</div>
                      <div className="space-y-1">
                        <div className="text-lg font-bold">
                          {data.regression_thresholds.sell_threshold_value !== null && data.regression_thresholds.sell_threshold_value !== undefined
                            ? formatDecimal(data.regression_thresholds.sell_threshold_value, 6)
                            : 'N/A'}
                        </div>
                        {data.regression_thresholds.sell_quantile !== null && (
                          <div className="text-xs text-muted-foreground">
                            Квантиль: {formatPercent(data.regression_thresholds.sell_quantile)} (Bottom {formatPercent(data.regression_thresholds.sell_quantile || 0)})
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="pt-4 border-t">
                    <h4 className="font-semibold mb-2 text-sm">Как работает квантильный подход:</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                      <li>Пороги вычисляются на валидационном наборе после обучения модели</li>
                      <li>Top 20% предсказаний → BUY сигнал (≥ {formatDecimal(data.regression_thresholds.buy_threshold_value || 0, 6)})</li>
                      <li>Bottom 20% предсказаний → SELL сигнал (≤ {formatDecimal(data.regression_thresholds.sell_threshold_value || 0, 6)})</li>
                      <li>Остальные 60% → HOLD (нет сигнала)</li>
                      <li>Это обеспечивает адаптивные пороги, которые подстраиваются под распределение предсказаний модели</li>
                    </ul>
                  </div>
                </>
              )}

              {data.regression_thresholds.method === 'fixed' && (
                <div className="pt-4 border-t">
                  <p className="text-sm text-muted-foreground">
                    Используется фиксированный порог из настроек (MODEL_REGRESSION_THRESHOLD).
                    Квантильные пороги недоступны для этой модели.
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Regression Analysis - only for regression */}
      {isRegression && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Error Statistics */}
          <Card>
            <CardHeader>
              <CardTitle>Статистика ошибок</CardTitle>
              <CardDescription>Анализ распределения ошибок предсказаний</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <MetricCard 
                    title="MAE" 
                    value={data.model_metrics.mae !== null && data.model_metrics.mae !== undefined ? formatDecimal(data.model_metrics.mae) : 'N/A'} 
                  />
                  <MetricCard 
                    title="RMSE" 
                    value={data.model_metrics.rmse !== null && data.model_metrics.rmse !== undefined ? formatDecimal(data.model_metrics.rmse) : 'N/A'} 
                  />
                  <MetricCard 
                    title="MSE" 
                    value={data.model_metrics.mse !== null && data.model_metrics.mse !== undefined ? formatDecimal(data.model_metrics.mse) : 'N/A'} 
                  />
                  <MetricCard 
                    title="R² Score" 
                    value={data.model_metrics.r2_score !== null && data.model_metrics.r2_score !== undefined ? formatDecimal(data.model_metrics.r2_score) : 'N/A'} 
                  />
                </div>
                <div className="pt-4 border-t">
                  <h4 className="font-semibold mb-2 text-sm">Интерпретация:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {data.model_metrics.r2_score !== null && data.model_metrics.r2_score < 0 && (
                      <li className="text-red-600">R² отрицательный - модель хуже, чем предсказание среднего значения</li>
                    )}
                    {data.model_metrics.r2_score !== null && data.model_metrics.r2_score >= 0 && data.model_metrics.r2_score < 0.3 && (
                      <li className="text-yellow-600">R² низкий (0-0.3) - слабая объясняющая способность</li>
                    )}
                    {data.model_metrics.r2_score !== null && data.model_metrics.r2_score >= 0.3 && data.model_metrics.r2_score < 0.7 && (
                      <li className="text-blue-600">R² средний (0.3-0.7) - умеренная объясняющая способность</li>
                    )}
                    {data.model_metrics.r2_score !== null && data.model_metrics.r2_score >= 0.7 && (
                      <li className="text-green-600">R² высокий (≥0.7) - хорошая объясняющая способность</li>
                    )}
                    {data.model_metrics.directional_accuracy !== null && data.model_metrics.directional_accuracy >= 0.5 && data.model_metrics.directional_accuracy < 0.6 && (
                      <li>Directional Accuracy {formatPercent(data.model_metrics.directional_accuracy)} - лучше случайного, но низкий</li>
                    )}
                    {data.model_metrics.directional_accuracy !== null && data.model_metrics.directional_accuracy >= 0.6 && (
                      <li className="text-green-600">Directional Accuracy {formatPercent(data.model_metrics.directional_accuracy)} - хороший показатель направления</li>
                    )}
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Regression Metrics Analysis */}
          <Card>
            <CardHeader>
              <CardTitle>Анализ метрик регрессии</CardTitle>
              <CardDescription>Дополнительные метрики качества модели</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <MetricCard 
                    title="Directional Accuracy" 
                    value={data.model_metrics.directional_accuracy !== null && data.model_metrics.directional_accuracy !== undefined ? formatPercent(data.model_metrics.directional_accuracy) : 'N/A'} 
                  />
                  <MetricCard 
                    title="Sharpe Ratio" 
                    value={data.model_metrics.sharpe_ratio !== null && data.model_metrics.sharpe_ratio !== undefined ? formatDecimal(data.model_metrics.sharpe_ratio) : 'N/A'} 
                  />
                  <MetricCard 
                    title="Information Coefficient" 
                    value={data.model_metrics.information_coefficient !== null && data.model_metrics.information_coefficient !== undefined ? formatDecimal(data.model_metrics.information_coefficient) : 'N/A'} 
                  />
                </div>
                <div className="pt-4 border-t">
                  <h4 className="font-semibold mb-2 text-sm">Оценка качества:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {data.model_metrics.sharpe_ratio !== null && data.model_metrics.sharpe_ratio < 0 && (
                      <li className="text-red-600">Sharpe Ratio отрицательный - риск не оправдан доходностью</li>
                    )}
                    {data.model_metrics.sharpe_ratio !== null && data.model_metrics.sharpe_ratio >= 0 && data.model_metrics.sharpe_ratio < 1 && (
                      <li>Sharpe Ratio {formatDecimal(data.model_metrics.sharpe_ratio)} - низкий риск-скорректированный доход</li>
                    )}
                    {data.model_metrics.sharpe_ratio !== null && data.model_metrics.sharpe_ratio >= 1 && (
                      <li className="text-green-600">Sharpe Ratio {formatDecimal(data.model_metrics.sharpe_ratio)} - хороший риск-скорректированный доход</li>
                    )}
                    {data.model_metrics.information_coefficient !== null && data.model_metrics.information_coefficient >= 0.1 && data.model_metrics.information_coefficient < 0.3 && (
                      <li>IC {formatDecimal(data.model_metrics.information_coefficient)} - слабая корреляция</li>
                    )}
                    {data.model_metrics.information_coefficient !== null && data.model_metrics.information_coefficient >= 0.3 && (
                      <li className="text-green-600">IC {formatDecimal(data.model_metrics.information_coefficient)} - умеренная/сильная корреляция</li>
                    )}
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Regression Visualizations - only for regression */}
      {isRegression && predictionsData && predictionsData.data_points.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Scatter Plot: y_true vs y_pred */}
          {scatterPlotData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Scatter Plot: Предсказания vs Фактические значения</CardTitle>
                <CardDescription>Визуализация корреляции между предсказаниями и реальными значениями</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <ScatterChart data={scatterPlotData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="y_true" 
                      name="Фактическое значение"
                      label={{ value: 'y_true', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      dataKey="y_pred" 
                      name="Предсказание"
                      label={{ value: 'y_pred', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Предсказания" data={scatterPlotData} fill="#8884d8" />
                    {/* Perfect prediction line (y = x) */}
                    {(() => {
                      const minVal = Math.min(...scatterPlotData.map(d => Math.min(d.y_true, d.y_pred || 0)))
                      const maxVal = Math.max(...scatterPlotData.map(d => Math.max(d.y_true, d.y_pred || 0)))
                      return (
                        <ReferenceLine 
                          segment={[{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]}
                          stroke="#82ca9d"
                          strokeDasharray="5 5"
                          strokeWidth={2}
                        />
                      )
                    })()}
                  </ScatterChart>
                </ResponsiveContainer>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p>Зелёная пунктирная линия: идеальное предсказание (y_pred = y_true)</p>
                  <p>Точки выше линии: переоценка, ниже линии: недооценка</p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Error Distribution */}
          {errorDistributionData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Распределение ошибок</CardTitle>
                <CardDescription>Гистограмма ошибок предсказаний (y_true - y_pred)</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={errorDistributionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="bin" 
                      name="Ошибка"
                      label={{ value: 'Ошибка (y_true - y_pred)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: 'Количество', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8884d8" name="Количество предсказаний" />
                  </BarChart>
                </ResponsiveContainer>
                <div className="mt-4 text-sm text-muted-foreground">
                  <p>Распределение ошибок показывает, насколько равномерно модель ошибается</p>
                  <p>Идеальное распределение: нормальное распределение с центром около 0</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Insights */}
      <Card>
        <CardHeader>
          <CardTitle>Инсайты и выводы</CardTitle>
          <CardDescription>Автоматический анализ результатов</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {isRegression ? (
              <>
                <div>
                  <h4 className="font-semibold mb-2">Анализ метрик регрессии:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {data.model_metrics.r2_score !== null && data.model_metrics.r2_score < 0 && (
                      <li className="text-red-600">
                        <strong>R² Score</strong> = {formatDecimal(data.model_metrics.r2_score)} - модель хуже, чем предсказание среднего значения. 
                        Рекомендуется улучшить признаки или пересмотреть подход.
                      </li>
                    )}
                    {data.model_metrics.directional_accuracy !== null && data.model_metrics.directional_accuracy >= 0.5 && (
                      <li className="text-green-600">
                        <strong>Directional Accuracy</strong> = {formatPercent(data.model_metrics.directional_accuracy)} - модель правильно определяет направление движения
                        {data.model_metrics.directional_accuracy >= 0.6 && ' ⭐ Хороший результат!'}
                      </li>
                    )}
                    {data.model_metrics.information_coefficient !== null && data.model_metrics.information_coefficient >= 0.1 && (
                      <li>
                        <strong>Information Coefficient</strong> = {formatDecimal(data.model_metrics.information_coefficient)} - 
                        {data.model_metrics.information_coefficient >= 0.3 ? ' умеренная/сильная' : ' слабая'} корреляция между предсказаниями и фактическими значениями
                      </li>
                    )}
                    {data.model_metrics.sharpe_ratio !== null && data.model_metrics.sharpe_ratio >= 1 && (
                      <li className="text-green-600">
                        <strong>Sharpe Ratio</strong> = {formatDecimal(data.model_metrics.sharpe_ratio)} - хороший риск-скорректированный доход ⭐
                      </li>
                    )}
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Рекомендации:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {data.model_metrics.r2_score !== null && data.model_metrics.r2_score < 0 && (
                      <li>R² отрицательный - рекомендуется пересмотреть признаки, увеличить объём данных или изменить гиперпараметры</li>
                    )}
                    {data.model_metrics.directional_accuracy !== null && data.model_metrics.directional_accuracy >= 0.6 && (
                      <li>Высокая Directional Accuracy позволяет использовать модель для определения направления сделок</li>
                    )}
                    {data.model_metrics.information_coefficient !== null && data.model_metrics.information_coefficient >= 0.3 && (
                      <li>Хорошая корреляция (IC ≥ 0.3) показывает, что модель улавливает закономерности в данных</li>
                    )}
                    {data.model_metrics.sharpe_ratio !== null && data.model_metrics.sharpe_ratio < 0 && (
                      <li>Отрицательный Sharpe Ratio указывает на высокий риск относительно доходности - требуется оптимизация</li>
                    )}
                  </ul>
                </div>
              </>
            ) : (
              <>
                <div>
                  <h4 className="font-semibold mb-2">Edge в Top-K%:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {data.top_k_metrics.map((tk) => (
                      <li key={tk.k}>
                        <strong>Top-{tk.k}%</strong>: PR-AUC = {formatDecimal(tk.pr_auc)}, 
                        Lift = {tk.lift !== null ? `${tk.lift.toFixed(2)}x` : 'N/A'}, 
                        Coverage = {formatPercent(tk.coverage)}
                        {tk.pr_auc && tk.pr_auc > 0.9 && ' ⭐ Отличный результат!'}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Сравнение с Baseline:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {Object.entries(data.comparison).map(([metric, comp]) => (
                      <li key={metric}>
                        <strong className="capitalize">{metric.replace('_', ' ')}</strong>: 
                        {comp.difference !== null && comp.difference > 0 ? (
                          <span className="text-green-600"> Модель лучше на {formatDecimal(comp.difference)}</span>
                        ) : comp.difference !== null && comp.difference < 0 ? (
                          <span className="text-red-600"> Baseline лучше на {formatDecimal(Math.abs(comp.difference))}</span>
                        ) : (
                          ' Нет разницы'
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Рекомендации:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                    {data.top_k_metrics.find(tk => tk.k === 10 && tk.pr_auc && tk.pr_auc > 0.9) && (
                      <li>Top-10% показывает очень высокий PR-AUC - можно использовать для высокоточных сигналов</li>
                    )}
                    {data.comparison.pr_auc && data.comparison.pr_auc.difference && data.comparison.pr_auc.difference > 0.5 && (
                      <li>Модель значительно превосходит baseline по PR-AUC - хороший знак для ранжирования</li>
                    )}
                    {data.top_k_metrics.find(tk => tk.lift && tk.lift > 0.8) && (
                      <li>Высокий Lift в top-k% показывает хорошее ранжирование предсказаний</li>
                    )}
                  </ul>
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

