import { useContainers } from '@/hooks'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'

export default function Containers() {
  const { data, isLoading } = useContainers()

  const getHealthBadgeVariant = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'default'
      case 'unhealthy':
        return 'destructive'
      case 'starting':
      case 'restarting':
        return 'outline'
      case 'running':
        return 'default'
      case 'exited':
        return 'destructive'
      default:
        return 'outline'
    }
  }

  const getHealthBadgeText = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'Healthy'
      case 'unhealthy':
        return 'Unhealthy'
      case 'starting':
        return 'Starting'
      case 'running':
        return 'Running'
      case 'restarting':
        return 'Restarting'
      case 'exited':
        return 'Exited'
      default:
        return 'Unknown'
    }
  }

  const getHealthBadgeColor = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'bg-green-500'
      case 'unhealthy':
        return 'bg-red-500'
      case 'starting':
      case 'restarting':
        return 'bg-yellow-500'
      case 'running':
        return 'bg-blue-500'
      case 'exited':
        return 'bg-gray-500'
      default:
        return 'bg-gray-400'
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Контейнеры</h2>
        <p className="text-muted-foreground">Статус Docker контейнеров системы</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Статус контейнеров</CardTitle>
          <CardDescription>
            {data ? `Всего контейнеров: ${data.count}` : 'Загрузка...'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-64 w-full" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Имя</TableHead>
                  <TableHead>Сервис</TableHead>
                  <TableHead>Образ</TableHead>
                  <TableHead>Статус</TableHead>
                  <TableHead>Health Status</TableHead>
                  <TableHead>Создан</TableHead>
                  <TableHead>Порты</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {!data || data.containers.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center text-muted-foreground">
                      Нет данных о контейнерах
                    </TableCell>
                  </TableRow>
                ) : (
                  data.containers.map((container) => (
                    <TableRow key={container.name}>
                      <TableCell className="font-mono text-sm">{container.name}</TableCell>
                      <TableCell>{container.service || 'N/A'}</TableCell>
                      <TableCell className="font-mono text-xs max-w-xs truncate" title={container.image}>
                        {container.image}
                      </TableCell>
                      <TableCell>
                        <span className="text-sm">{container.status}</span>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full ${getHealthBadgeColor(container.health_status)}`} />
                          <Badge variant={getHealthBadgeVariant(container.health_status)}>
                            {getHealthBadgeText(container.health_status)}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {container.created}
                      </TableCell>
                      <TableCell>
                        {container.ports && container.ports.length > 0 ? (
                          <div className="flex flex-col gap-1">
                            {container.ports.map((port, idx) => (
                              <span key={idx} className="font-mono text-xs">
                                {port}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-muted-foreground text-sm">—</span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

