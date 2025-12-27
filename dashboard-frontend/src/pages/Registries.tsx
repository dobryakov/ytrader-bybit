import { useState, useEffect } from 'react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import api from '@/lib/api'

interface RegistryVersion {
  version: string
  is_active: boolean
  created_at?: string
  description?: string
  config?: {
    type?: string
    horizon?: number
    threshold?: number | null
    computation?: {
      preset?: string
    }
  }
}

export default function Registries() {
  const [featureRegistryVersions, setFeatureRegistryVersions] = useState<RegistryVersion[]>([])
  const [targetRegistryVersions, setTargetRegistryVersions] = useState<RegistryVersion[]>([])
  const [isLoadingFeature, setIsLoadingFeature] = useState(true)
  const [isLoadingTarget, setIsLoadingTarget] = useState(true)
  const [activatingFeature, setActivatingFeature] = useState<string | null>(null)
  const [activatingTarget, setActivatingTarget] = useState<string | null>(null)

  const loadFeatureRegistryVersions = async () => {
    try {
      setIsLoadingFeature(true)
      const response = await api.get('/v1/feature-registry/versions')
      setFeatureRegistryVersions(response.data || [])
    } catch (error: any) {
      console.error('Failed to load feature registry versions:', error)
      console.error('Error details:', error.response?.data?.detail || error.message)
    } finally {
      setIsLoadingFeature(false)
    }
  }

  const loadTargetRegistryVersions = async () => {
    try {
      setIsLoadingTarget(true)
      const response = await api.get('/v1/target-registry/versions')
      // Load config for each version to get type, preset, etc.
      const versionsWithConfig = await Promise.all(
        (response.data || []).map(async (version: any) => {
          try {
            const versionResponse = await api.get(`/v1/target-registry/versions/${version.version}`)
            return {
              ...version,
              config: versionResponse.data.config,
            }
          } catch {
            return version
          }
        })
      )
      setTargetRegistryVersions(versionsWithConfig)
    } catch (error: any) {
      console.error('Failed to load target registry versions:', error)
      console.error('Error details:', error.response?.data?.detail || error.message)
    } finally {
      setIsLoadingTarget(false)
    }
  }

  useEffect(() => {
    loadFeatureRegistryVersions()
    loadTargetRegistryVersions()
  }, [])

  const handleActivateFeatureRegistry = async (version: string) => {
    console.log(`Activating Feature Registry version: ${version}`)
    
    try {
      setActivatingFeature(version)
      await api.post(`/v1/feature-registry/versions/${version}/activate`, {
        activated_by: 'dashboard-user',
        activation_reason: 'Manual activation from dashboard',
        acknowledge_breaking_changes: false,
      })
      console.log(`Feature Registry version ${version} activated successfully`)
      await loadFeatureRegistryVersions()
    } catch (error: any) {
      console.error('Failed to activate feature registry version:', error)
      console.error('Error details:', error.response?.data?.detail || error.message)
    } finally {
      setActivatingFeature(null)
    }
  }

  const handleActivateTargetRegistry = async (version: string) => {
    console.log(`Activating Target Registry version: ${version}`)
    
    try {
      setActivatingTarget(version)
      await api.post(`/v1/target-registry/versions/${version}/activate`, {
        activated_by: 'dashboard-user',
        activation_reason: 'Manual activation from dashboard',
      })
      console.log(`Target Registry version ${version} activated successfully`)
      await loadTargetRegistryVersions()
    } catch (error: any) {
      console.error('Failed to activate target registry version:', error)
      console.error('Error details:', error.response?.data?.detail || error.message)
    } finally {
      setActivatingTarget(null)
    }
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A'
    try {
      return new Date(dateString).toLocaleString('ru-RU')
    } catch {
      return dateString
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold">Feature & Target Registry</h2>
        <p className="text-muted-foreground">Управление версиями Feature Registry и Target Registry</p>
      </div>

      {/* Feature Registry Table */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold">Feature Registry Versions</h3>
          <Button onClick={loadFeatureRegistryVersions} variant="outline" size="sm">
            Обновить
          </Button>
        </div>
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Версия</TableHead>
                <TableHead>Статус</TableHead>
                <TableHead>Дата создания</TableHead>
                <TableHead>Описание</TableHead>
                <TableHead>Действия</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoadingFeature ? (
                <TableRow>
                  <TableCell colSpan={5}>
                    <Skeleton className="h-4 w-full" />
                  </TableCell>
                </TableRow>
              ) : featureRegistryVersions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center text-muted-foreground">
                    Нет версий
                  </TableCell>
                </TableRow>
              ) : (
                featureRegistryVersions.map((version) => (
                  <TableRow key={version.version}>
                    <TableCell className="font-mono font-semibold">{version.version}</TableCell>
                    <TableCell>
                      {version.is_active ? (
                        <Badge variant="default">Активна</Badge>
                      ) : (
                        <Badge variant="secondary">Неактивна</Badge>
                      )}
                    </TableCell>
                    <TableCell>{formatDate(version.created_at)}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {version.description || version.config?.description || 'N/A'}
                    </TableCell>
                    <TableCell>
                      {!version.is_active && (
                        <Button
                          onClick={() => handleActivateFeatureRegistry(version.version)}
                          disabled={activatingFeature === version.version}
                          size="sm"
                          variant="outline"
                        >
                          {activatingFeature === version.version ? 'Активация...' : 'Активировать'}
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      {/* Target Registry Table */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold">Target Registry Versions</h3>
          <Button onClick={loadTargetRegistryVersions} variant="outline" size="sm">
            Обновить
          </Button>
        </div>
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Версия</TableHead>
                <TableHead>Статус</TableHead>
                <TableHead>Тип</TableHead>
                <TableHead>Preset</TableHead>
                <TableHead>Horizon</TableHead>
                <TableHead>Дата создания</TableHead>
                <TableHead>Действия</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoadingTarget ? (
                <TableRow>
                  <TableCell colSpan={7}>
                    <Skeleton className="h-4 w-full" />
                  </TableCell>
                </TableRow>
              ) : targetRegistryVersions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-muted-foreground">
                    Нет версий
                  </TableCell>
                </TableRow>
              ) : (
                targetRegistryVersions.map((version) => (
                  <TableRow key={version.version}>
                    <TableCell className="font-mono font-semibold">{version.version}</TableCell>
                    <TableCell>
                      {version.is_active ? (
                        <Badge variant="default">Активна</Badge>
                      ) : (
                        <Badge variant="secondary">Неактивна</Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge variant={version.config?.type === 'regression' ? 'default' : 'secondary'}>
                        {version.config?.type || 'N/A'}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm">
                      {version.config?.computation?.preset || 'N/A'}
                    </TableCell>
                    <TableCell className="text-sm">
                      {version.config?.horizon ? `${version.config.horizon}s` : 'N/A'}
                    </TableCell>
                    <TableCell>{formatDate(version.created_at)}</TableCell>
                    <TableCell>
                      {!version.is_active && (
                        <Button
                          onClick={() => handleActivateTargetRegistry(version.version)}
                          disabled={activatingTarget === version.version}
                          size="sm"
                          variant="outline"
                        >
                          {activatingTarget === version.version ? 'Активация...' : 'Активировать'}
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}

