import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import Overview from './pages/Overview'
import Positions from './pages/Positions'
import Orders from './pages/Orders'
import Signals from './pages/Signals'
import Models from './pages/Models'
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
                </div>
              </div>
            </div>
          </div>
        </nav>

        <main className="container mx-auto px-4 py-6">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/positions" element={<Positions />} />
            <Route path="/orders" element={<Orders />} />
            <Route path="/signals" element={<Signals />} />
            <Route path="/models" element={<Models />} />
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
