import { useParams, useNavigate } from 'react-router-dom'
import { useModelAnalysis } from '@/hooks/useModels'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { MetricCard } from '@/components/metrics/MetricCard'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts'
import { ArrowLeft, TrendingUp, TrendingDown, Minus } from 'lucide-react'

export default function ModelDetail() {
  const { version } = useParams<{ version: string }>()
  const navigate = useNavigate()
  const { data, isLoading, error } = useModelAnalysis(version || '')

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

  // Prepare data for top-k chart
  const topKChartData = data.top_k_metrics.map((tk) => ({
    k: `Top-${tk.k}%`,
    pr_auc: tk.pr_auc ? tk.pr_auc * 100 : null,
    roc_auc: tk.roc_auc ? tk.roc_auc * 100 : null,
    accuracy: tk.accuracy ? tk.accuracy * 100 : null,
    lift: tk.lift ? tk.lift : null, // Lift is already a ratio (e.g., 1.2 = 20% improvement)
  }))

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

      {/* Confidence Threshold Info */}
      {data.confidence_threshold_info && (
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
                        <span className="text-sm text-muted-foreground">Top-K процент:</span>
                        <span className="font-medium">Top-{data.confidence_threshold_info.top_k_percentage}%</span>
                      </div>
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
                  <li>
                    {data.confidence_threshold_info.threshold_source === 'top_k' 
                      ? `Использует метрику ${data.confidence_threshold_info.metric_name} из model_quality_metrics`
                      : 'Ищет метрику top_k_{percentage}_confidence_threshold в model_quality_metrics'}
                  </li>
                  <li>
                    {data.confidence_threshold_info.threshold_source === 'top_k'
                      ? 'Порог найден и используется для фильтрации сигналов'
                      : 'Метрика не найдена, используется статический порог из настроек'}
                  </li>
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
          <CardDescription>Raw probabilities и y_true для анализа</CardDescription>
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
                      {pred.dataset_id ? `${pred.dataset_id.slice(0, 8)}...` : 'N/A'}
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
            <div className="grid grid-cols-2 gap-4">
              <MetricCard title="Accuracy" value={formatPercent(data.model_metrics.accuracy)} />
              <MetricCard title="Precision" value={formatPercent(data.model_metrics.precision)} />
              <MetricCard title="Recall" value={formatPercent(data.model_metrics.recall)} />
              <MetricCard title="F1 Score" value={formatPercent(data.model_metrics.f1_score)} />
              <MetricCard title="Balanced Accuracy" value={formatPercent(data.model_metrics.balanced_accuracy)} />
              <MetricCard title="ROC AUC" value={formatDecimal(data.model_metrics.roc_auc)} />
              <MetricCard title="PR AUC" value={formatDecimal(data.model_metrics.pr_auc)} />
            </div>
          </CardContent>
        </Card>

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
      </div>

      {/* Comparison */}
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

      {/* Top-K Metrics */}
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

      {/* Detailed Metrics Comparison Table */}
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

      {/* Insights */}
      <Card>
        <CardHeader>
          <CardTitle>Инсайты и выводы</CardTitle>
          <CardDescription>Автоматический анализ результатов</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
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
                {data.comparison.pr_auc.difference && data.comparison.pr_auc.difference > 0.5 && (
                  <li>Модель значительно превосходит baseline по PR-AUC - хороший знак для ранжирования</li>
                )}
                {data.top_k_metrics.find(tk => tk.lift && tk.lift > 0.8) && (
                  <li>Высокий Lift в top-k% показывает хорошее ранжирование предсказаний</li>
                )}
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

