import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Signal } from '@/hooks/useSignals'
import { format } from 'date-fns'
import { parseISO } from 'date-fns'

interface SignalsTableProps {
  signals: Signal[]
}

export function SignalsTable({ signals }: SignalsTableProps) {
  if (signals.length === 0) {
    return <div className="text-center text-muted-foreground py-4">Нет сигналов</div>
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Asset</TableHead>
          <TableHead>Type</TableHead>
          <TableHead>Amount</TableHead>
          <TableHead>Confidence</TableHead>
          <TableHead>Strategy</TableHead>
          <TableHead>Time</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {signals.map((signal) => (
          <TableRow key={signal.signal_id}>
            <TableCell className="font-medium">{signal.asset}</TableCell>
            <TableCell>
              <Badge variant={signal.signal_type === 'buy' ? 'default' : 'destructive'}>
                {signal.signal_type.toUpperCase()}
              </Badge>
            </TableCell>
            <TableCell>{parseFloat(signal.amount).toFixed(2)} USDT</TableCell>
            <TableCell>
              {signal.confidence ? (signal.confidence * 100).toFixed(2) + '%' : 'N/A'}
            </TableCell>
            <TableCell>{signal.strategy_id || 'N/A'}</TableCell>
            <TableCell>{format(parseISO(signal.timestamp), 'dd.MM.yyyy HH:mm:ss')}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

