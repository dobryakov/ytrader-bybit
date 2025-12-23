import { useModels } from '@/hooks/useModels'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'

export default function Models() {
  const { data, isLoading } = useModels({ is_active: true })

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
              <TableHead>F1 Score</TableHead>
              <TableHead>Precision</TableHead>
              <TableHead>Recall</TableHead>
              <TableHead>Accuracy</TableHead>
              <TableHead>Trained At</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data?.models.length === 0 ? (
              <TableRow>
                <TableCell colSpan={10} className="text-center text-muted-foreground">
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
                    {model.metrics?.f1_score ? (model.metrics.f1_score * 100).toFixed(2) + '%' : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {model.metrics?.precision_score ? (model.metrics.precision_score * 100).toFixed(2) + '%' : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {model.metrics?.recall_score ? (model.metrics.recall_score * 100).toFixed(2) + '%' : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {model.metrics?.accuracy_score ? (model.metrics.accuracy_score * 100).toFixed(2) + '%' : 'N/A'}
                  </TableCell>
                  <TableCell>{format(parseISO(model.trained_at), 'dd.MM.yyyy HH:mm:ss')}</TableCell>
                  <TableCell>
                    <Badge variant={model.is_active ? 'default' : 'outline'}>
                      {model.is_active ? 'Active' : 'Inactive'}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      )}
    </div>
  )
}

