import { useOverviewMetrics, usePortfolioMetrics, useSignals, useBalances } from '@/hooks'
import { MetricCard } from '@/components/metrics/MetricCard'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { PnLChart } from '@/components/charts/PnLChart'
import { SignalsTable } from '@/components/tables/SignalsTable'
import { format } from 'date-fns'

export default function Overview() {
  const { data: metrics, isLoading: metricsLoading } = useOverviewMetrics()
  const { data: portfolio, isLoading: portfolioLoading } = usePortfolioMetrics()
  const { data: signals, isLoading: signalsLoading } = useSignals({ page_size: 10 })
  const { data: balances, isLoading: balancesLoading } = useBalances()

  const formatCurrency = (value: string | null) => {
    if (!value) return 'N/A'
    const num = parseFloat(value)
    return new Intl.NumberFormat('ru-RU', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(num)
  }

  const formatBalance = (value: string | null, coin: string) => {
    if (!value) return 'N/A'
    const num = parseFloat(value)
    // For USDT/USDC, format as currency. For crypto (BTC, ETH), format as number with appropriate decimals
    if (coin === 'USDT' || coin === 'USDC') {
      return new Intl.NumberFormat('ru-RU', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(num)
    } else {
      // For crypto: use more decimals for small amounts, fewer for large
      const decimals = num < 1 ? 8 : (num < 100 ? 4 : 2)
      return new Intl.NumberFormat('ru-RU', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      }).format(num) + ` ${coin}`
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Обзор</h2>
        <p className="text-muted-foreground">Ключевые метрики торговой системы</p>
      </div>

      {/* Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {metricsLoading ? (
          <>
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
          </>
        ) : (
          <>
            <MetricCard
              title="Баланс"
              value={formatCurrency(metrics?.balance)}
              description="Доступный баланс"
            />
            <MetricCard
              title="Unrealized PnL"
              value={formatCurrency(metrics?.total_unrealized_pnl)}
              description="Нереализованная прибыль/убыток"
            />
            <MetricCard
              title="Realized PnL"
              value={formatCurrency(metrics?.total_realized_pnl)}
              description="Реализованная прибыль/убыток"
            />
            <MetricCard
              title="Открытые позиции"
              value={metrics?.open_positions_count.toString() || '0'}
              description={`Всего позиций: ${metrics?.total_positions_count || 0}`}
            />
          </>
        )}
      </div>

      {/* PnL Chart */}
      <Card>
        <CardHeader>
          <CardTitle>График PnL</CardTitle>
          <CardDescription>История прибыли и убытков</CardDescription>
        </CardHeader>
        <CardContent>
          <PnLChart />
        </CardContent>
      </Card>

      {/* Recent Signals */}
      <Card>
        <CardHeader>
          <CardTitle>Последние сигналы</CardTitle>
          <CardDescription>10 последних торговых сигналов</CardDescription>
        </CardHeader>
        <CardContent>
          {signalsLoading ? (
            <Skeleton className="h-64" />
          ) : (
            <SignalsTable signals={signals?.signals || []} />
          )}
        </CardContent>
      </Card>

      {/* Balances by Asset */}
      <Card>
        <CardHeader>
          <CardTitle>Балансы по активам</CardTitle>
          <CardDescription>Доступный баланс для торговли</CardDescription>
        </CardHeader>
        <CardContent>
          {balancesLoading ? (
            <Skeleton className="h-32" />
          ) : balances && balances.balances.length > 0 ? (
            <div className="space-y-2">
              {balances.balances.map((balance) => (
                <div key={balance.coin} className="flex justify-between items-center py-2 border-b last:border-b-0">
                  <span className="font-medium">{balance.coin}</span>
                  <div className="flex gap-4">
                    <div className="text-right">
                      <div className="text-sm font-semibold text-green-600">
                        {formatBalance(balance.available_balance, balance.coin)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Доступно для торговли
                      </div>
                    </div>
                    {parseFloat(balance.frozen) > 0 && (
                      <div className="text-right">
                        <div className="text-sm text-muted-foreground">
                          {formatBalance(balance.frozen, balance.coin)}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Заморожено
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-4">
              Нет данных о балансах
            </div>
          )}
        </CardContent>
      </Card>

      {/* Portfolio Distribution */}
      {!portfolioLoading && portfolio && portfolio.portfolio.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Распределение портфолио</CardTitle>
            <CardDescription>Exposure по активам</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {portfolio.portfolio.slice(0, 5).map((item) => (
                <div key={item.asset} className="flex justify-between items-center">
                  <span className="font-medium">{item.asset}</span>
                  <div className="flex gap-4">
                    <span className="text-sm text-muted-foreground">
                      Exposure: {formatCurrency(item.exposure)}
                    </span>
                    <span className="text-sm">
                      PnL: {formatCurrency(item.unrealized_pnl)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

