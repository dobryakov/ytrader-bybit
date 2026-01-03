import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import api from '@/lib/api'
import type { Mode } from '@/hooks/useModes'

interface ModeFormProps {
  mode: Mode | null
  onSubmit: (modeData: Omit<Mode, 'id' | 'created_at' | 'updated_at' | 'created_by'>) => void
  onCancel: () => void
}

interface RegistryVersion {
  version: string
  is_active: boolean
}

export default function ModeForm({ mode, onSubmit, onCancel }: ModeFormProps) {
  const [name, setName] = useState(mode?.name || '')
  const [asset, setAsset] = useState(mode?.asset || '')
  const [strategyId, setStrategyId] = useState(mode?.strategy_id || '')
  const [featureRegistryVersion, setFeatureRegistryVersion] = useState(mode?.feature_registry_version || '')
  const [targetRegistryVersion, setTargetRegistryVersion] = useState(mode?.target_registry_version || '')
  const [trainDurationDays, setTrainDurationDays] = useState(mode?.train_duration_days || 60)
  const [validationDurationDays, setValidationDurationDays] = useState(mode?.validation_duration_days || 7)
  const [testDurationDays, setTestDurationDays] = useState(mode?.test_duration_days || 1)
  const [description, setDescription] = useState(mode?.description || '')
  const [isActive, setIsActive] = useState(mode?.is_active ?? true)

  const [featureRegistryVersions, setFeatureRegistryVersions] = useState<RegistryVersion[]>([])
  const [targetRegistryVersions, setTargetRegistryVersions] = useState<RegistryVersion[]>([])
  const [isLoadingVersions, setIsLoadingVersions] = useState(true)

  useEffect(() => {
    const loadVersions = async () => {
      try {
        setIsLoadingVersions(true)
        const [featureResponse, targetResponse] = await Promise.all([
          api.get('/v1/feature-registry/versions'),
          api.get('/v1/target-registry/versions'),
        ])
        setFeatureRegistryVersions(featureResponse.data || [])
        setTargetRegistryVersions(targetResponse.data || [])
      } catch (error) {
        console.error('Failed to load registry versions:', error)
      } finally {
        setIsLoadingVersions(false)
      }
    }
    loadVersions()
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (!name || !asset || !strategyId || !featureRegistryVersion || !targetRegistryVersion) {
      alert('Пожалуйста, заполните все обязательные поля')
      return
    }

    if (trainDurationDays <= 0 || validationDurationDays <= 0 || testDurationDays <= 0) {
      alert('Длительности периодов должны быть больше 0')
      return
    }

    onSubmit({
      name,
      asset,
      strategy_id: strategyId,
      feature_registry_version: featureRegistryVersion,
      target_registry_version: targetRegistryVersion,
      train_duration_days: trainDurationDays,
      validation_duration_days: validationDurationDays,
      test_duration_days: testDurationDays,
      description: description || null,
      is_active: isActive,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h3 className="text-xl font-semibold mb-4">
        {mode ? 'Редактировать режим' : 'Создать режим'}
      </h3>

      <div>
        <label htmlFor="name" className="block text-sm font-medium mb-1">
          Название <span className="text-red-500">*</span>
        </label>
        <input
          id="name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          placeholder="ETHUSDT-60days-v1.7.2"
        />
      </div>

      <div>
        <label htmlFor="asset" className="block text-sm font-medium mb-1">
          Ассет <span className="text-red-500">*</span>
        </label>
        <input
          id="asset"
          type="text"
          value={asset}
          onChange={(e) => setAsset(e.target.value.toUpperCase())}
          required
          className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          placeholder="BTCUSDT"
        />
      </div>

      <div>
        <label htmlFor="strategy_id" className="block text-sm font-medium mb-1">
          Стратегия <span className="text-red-500">*</span>
        </label>
        <input
          id="strategy_id"
          type="text"
          value={strategyId}
          onChange={(e) => setStrategyId(e.target.value)}
          required
          className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          placeholder="test-strategy"
        />
      </div>

      <div>
        <label htmlFor="feature_registry_version" className="block text-sm font-medium mb-1">
          Feature Registry версия <span className="text-red-500">*</span>
        </label>
        {isLoadingVersions ? (
          <div className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm">
            Загрузка...
          </div>
        ) : (
          <select
            id="feature_registry_version"
            value={featureRegistryVersion}
            onChange={(e) => setFeatureRegistryVersion(e.target.value)}
            required
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          >
            <option value="">Выберите версию</option>
            {featureRegistryVersions.map((version) => (
              <option key={version.version} value={version.version}>
                {version.version} {version.is_active ? '(активна)' : ''}
              </option>
            ))}
          </select>
        )}
      </div>

      <div>
        <label htmlFor="target_registry_version" className="block text-sm font-medium mb-1">
          Target Registry версия <span className="text-red-500">*</span>
        </label>
        {isLoadingVersions ? (
          <div className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm">
            Загрузка...
          </div>
        ) : (
          <select
            id="target_registry_version"
            value={targetRegistryVersion}
            onChange={(e) => setTargetRegistryVersion(e.target.value)}
            required
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          >
            <option value="">Выберите версию</option>
            {targetRegistryVersions.map((version) => (
              <option key={version.version} value={version.version}>
                {version.version} {version.is_active ? '(активна)' : ''}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <label htmlFor="train_duration_days" className="block text-sm font-medium mb-1">
            Train (дни) <span className="text-red-500">*</span>
          </label>
          <input
            id="train_duration_days"
            type="number"
            min="1"
            value={trainDurationDays}
            onChange={(e) => setTrainDurationDays(parseInt(e.target.value) || 0)}
            required
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          />
        </div>
        <div>
          <label htmlFor="validation_duration_days" className="block text-sm font-medium mb-1">
            Validation (дни) <span className="text-red-500">*</span>
          </label>
          <input
            id="validation_duration_days"
            type="number"
            min="1"
            value={validationDurationDays}
            onChange={(e) => setValidationDurationDays(parseInt(e.target.value) || 0)}
            required
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          />
        </div>
        <div>
          <label htmlFor="test_duration_days" className="block text-sm font-medium mb-1">
            Test (дни) <span className="text-red-500">*</span>
          </label>
          <input
            id="test_duration_days"
            type="number"
            min="1"
            value={testDurationDays}
            onChange={(e) => setTestDurationDays(parseInt(e.target.value) || 0)}
            required
            className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          />
        </div>
      </div>

      <div>
        <label htmlFor="description" className="block text-sm font-medium mb-1">
          Описание
        </label>
        <textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
          className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
          placeholder="Описание режима (опционально)"
        />
      </div>

      {mode && (
        <div>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={isActive}
              onChange={(e) => setIsActive(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm font-medium">Активен</span>
          </label>
        </div>
      )}

      <div className="flex justify-end gap-2 pt-4">
        <Button type="button" variant="outline" onClick={onCancel}>
          Отмена
        </Button>
        <Button type="submit">
          {mode ? 'Сохранить' : 'Создать'}
        </Button>
      </div>
    </form>
  )
}

