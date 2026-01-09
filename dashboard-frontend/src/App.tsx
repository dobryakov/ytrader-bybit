import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import Overview from './pages/Overview'
import Positions from './pages/Positions'
import PositionDetail from './pages/PositionDetail'
import Orders from './pages/Orders'
import Signals from './pages/Signals'
import Models from './pages/Models'
import ModelDetail from './pages/ModelDetail'
import Datasets from './pages/Datasets'
import DatasetDetail from './pages/DatasetDetail'
import Containers from './pages/Containers'
import Registries from './pages/Registries'
import Modes from './pages/Modes'
import { cn } from './lib/utils'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background">
        <nav className="border-b">
          <div className="container mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-8">
                <h1 className="text-xl font-bold">YTrader Dashboard</h1>
                <div className="flex space-x-4">
                  <NavLink to="/">Обзор</NavLink>
                  <NavLink to="/positions">Позиции</NavLink>
                  <NavLink to="/orders">Ордера</NavLink>
                  <NavLink to="/signals">Сигналы</NavLink>
                  <NavLink to="/models">Модели</NavLink>
                  <NavLink to="/datasets">Датасеты</NavLink>
                  <NavLink to="/modes">Режимы</NavLink>
                  <NavLink to="/containers">Контейнеры</NavLink>
                  <NavLink to="/registries">Registry</NavLink>
                </div>
              </div>
            </div>
          </div>
        </nav>

        <main className="container mx-auto px-4 py-6">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/positions" element={<Positions />} />
            <Route path="/positions/:positionId" element={<PositionDetail />} />
            <Route path="/orders" element={<Orders />} />
            <Route path="/signals" element={<Signals />} />
            <Route path="/models" element={<Models />} />
            <Route path="/models/:version" element={<ModelDetail />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/datasets/:id" element={<DatasetDetail />} />
            <Route path="/modes" element={<Modes />} />
            <Route path="/containers" element={<Containers />} />
            <Route path="/registries" element={<Registries />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

function NavLink({ to, children }: { to: string; children: React.ReactNode }) {
  const location = useLocation()
  const isActive = location.pathname === to
  
  return (
    <Link
      to={to}
      className={cn(
        'text-sm font-medium transition-colors',
        isActive ? 'text-primary' : 'text-muted-foreground hover:text-primary'
      )}
    >
      {children}
    </Link>
  )
}

export default App
