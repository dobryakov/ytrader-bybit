#!/usr/bin/env python3
"""
Скрипт для анализа данных свечей и рекомендации параметров классификации.

Анализирует изменения цены за разные горизонты и пороги,
рекомендует оптимальные значения threshold и horizon для классификации.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# Добавляем путь к src для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.parquet_storage import ParquetStorage
from src.config import config


class ThresholdHorizonAnalyzer:
    """Анализатор для рекомендации threshold и horizon."""
    
    def __init__(self, data_path: str = None):
        """
        Инициализация анализатора.
        
        Args:
            data_path: Путь к данным (по умолчанию из конфига)
        """
        if data_path is None:
            data_path = config.feature_service_raw_data_path
        self._parquet_storage = ParquetStorage(data_path)
    
    async def load_klines_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Загружает данные свечей за указанный период.
        
        Args:
            symbol: Торговая пара (например, 'BTCUSDT')
            start_date: Начальная дата
            end_date: Конечная дата
            
        Returns:
            DataFrame с колонками: timestamp, open, high, low, close
        """
        print(f"Загрузка данных для {symbol} с {start_date.date()} по {end_date.date()}...")
        
        # Убеждаемся, что start_date и end_date в timezone-aware UTC
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        else:
            start_date = start_date.astimezone(timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        else:
            end_date = end_date.astimezone(timezone.utc)
        
        all_data = []
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            date_str = current_date.strftime("%Y-%m-%d")
            try:
                df = await self._parquet_storage.read_klines(symbol, date_str)
                if df is not None and not df.empty:
                    # Убеждаемся, что timestamp в timezone-aware UTC
                    if hasattr(df["timestamp"].dtype, 'tz') and df["timestamp"].dtype.tz is None:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    elif hasattr(df["timestamp"].dtype, 'tz') and df["timestamp"].dtype.tz is not None:
                        df["timestamp"] = df["timestamp"].dt.tz_convert(timezone.utc)
                    else:
                        # Для numpy datetime64 без timezone
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    
                    # Фильтруем по timestamp для точного периода
                    df = df[
                        (df["timestamp"] >= start_date) &
                        (df["timestamp"] <= end_date)
                    ]
                    if not df.empty:
                        all_data.append(df)
            except Exception as e:
                print(f"  Предупреждение: не удалось загрузить данные за {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        if not all_data:
            raise ValueError(f"Не найдено данных для {symbol} за указанный период")
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Удаляем дубликаты по timestamp
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        
        print(f"  Загружено {len(df)} свечей")
        return df[["timestamp", "open", "high", "low", "close"]]
    
    def calculate_price_changes(
        self,
        df: pd.DataFrame,
        horizons_minutes: List[int],
    ) -> pd.DataFrame:
        """
        Вычисляет изменения цены за разные горизонты.
        
        Args:
            df: DataFrame с данными свечей
            horizons_minutes: Список горизонтов в минутах
            
        Returns:
            DataFrame с добавленными колонками для каждого горизонта
        """
        df = df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        for horizon_min in horizons_minutes:
            # Вычисляем изменение цены через horizon_min минут
            # Используем shift для получения будущей цены
            future_close = df["close"].shift(-horizon_min)
            current_open = df["open"]
            
            # Вычисляем относительное изменение
            # ΔP = (close_future - open_current) / open_current
            price_change = (future_close - current_open) / current_open
            
            # Сохраняем как процент
            df[f"price_change_{horizon_min}m"] = price_change * 100
        
        # Удаляем строки, где нет будущих данных (последние horizon_min строк)
        max_horizon = max(horizons_minutes)
        df = df.iloc[:-max_horizon].copy()
        
        return df
    
    def calculate_threshold_percentages(
        self,
        df: pd.DataFrame,
        horizons_minutes: List[int],
        thresholds_pct: List[float],
    ) -> pd.DataFrame:
        """
        Вычисляет долю свечей, превышающих пороги.
        
        Args:
            df: DataFrame с изменениями цены
            horizons_minutes: Список горизонтов
            thresholds_pct: Список порогов в процентах
            
        Returns:
            DataFrame с результатами анализа
        """
        results = []
        
        for horizon_min in horizons_minutes:
            col_name = f"price_change_{horizon_min}m"
            if col_name not in df.columns:
                continue
            
            price_changes = df[col_name].dropna()
            total_count = len(price_changes)
            
            if total_count == 0:
                continue
            
            for threshold_pct in thresholds_pct:
                # Считаем долю свечей, где |ΔP| > threshold
                exceeded = (price_changes.abs() > threshold_pct).sum()
                percentage = (exceeded / total_count) * 100
                
                results.append({
                    "horizon_minutes": horizon_min,
                    "threshold_percent": threshold_pct,
                    "exceeded_count": exceeded,
                    "total_count": total_count,
                    "percentage": percentage,
                })
        
        return pd.DataFrame(results)
    
    def recommend_parameters(
        self,
        results_df: pd.DataFrame,
        target_percentage_range: Tuple[float, float] = (20.0, 30.0),
    ) -> Dict:
        """
        Рекомендует threshold и horizon на основе анализа.
        
        Args:
            results_df: DataFrame с результатами анализа
            target_percentage_range: Целевой диапазон процента (min, max)
            
        Returns:
            Словарь с рекомендациями
        """
        recommendations = {
            "threshold_recommendations": [],
            "horizon_recommendations": [],
        }
        
        # Вычисляем середину целевого диапазона
        target_mid = (target_percentage_range[0] + target_percentage_range[1]) / 2
        
        # Ищем горизонты и пороги, где процент попадает в целевой диапазон
        target_results = results_df[
            (results_df["percentage"] >= target_percentage_range[0]) &
            (results_df["percentage"] <= target_percentage_range[1])
        ].copy()
        
        if target_results.empty:
            # Если нет точных попаданий, ищем ближайшие
            print(f"\n⚠️  Нет результатов в целевом диапазоне {target_percentage_range[0]}-{target_percentage_range[1]}%")
            print("   Ищем ближайшие значения...")
            
            # Находим результаты, максимально близкие к середине диапазона
            target_results = results_df.copy()
            target_results["distance"] = (target_results["percentage"] - target_mid).abs()
            target_results = target_results.nsmallest(10, "distance")
        else:
            # Всегда вычисляем расстояние до середины диапазона для выбора лучшего варианта
            target_results["distance"] = (target_results["percentage"] - target_mid).abs()
        
        # Группируем по горизонтам и порогам
        for horizon_min in sorted(results_df["horizon_minutes"].unique()):
            horizon_data = target_results[target_results["horizon_minutes"] == horizon_min]
            if not horizon_data.empty:
                # Берем порог с процентом, ближайшим к середине диапазона
                best = horizon_data.nsmallest(1, "distance")
                if not best.empty:
                    row = best.iloc[0]
                    recommendations["threshold_recommendations"].append({
                        "horizon_minutes": int(row["horizon_minutes"]),
                        "threshold_percent": float(row["threshold_percent"]),
                        "percentage": float(row["percentage"]),
                        "exceeded_count": int(row["exceeded_count"]),
                        "total_count": int(row["total_count"]),
                    })
        
        # Группируем по порогам
        for threshold_pct in sorted(results_df["threshold_percent"].unique()):
            threshold_data = target_results[target_results["threshold_percent"] == threshold_pct]
            if not threshold_data.empty:
                # Берем горизонт с процентом, ближайшим к середине диапазона
                best = threshold_data.nsmallest(1, "distance")
                if not best.empty:
                    row = best.iloc[0]
                    recommendations["horizon_recommendations"].append({
                        "horizon_minutes": int(row["horizon_minutes"]),
                        "threshold_percent": float(row["threshold_percent"]),
                        "percentage": float(row["percentage"]),
                        "exceeded_count": int(row["exceeded_count"]),
                        "total_count": int(row["total_count"]),
                    })
        
        return recommendations
    
    def print_results_table(self, results_df: pd.DataFrame):
        """Выводит таблицу результатов."""
        print("\n" + "=" * 80)
        print("ТАБЛИЦА: Доля свечей с |ΔP| > порога по горизонтам")
        print("=" * 80)
        
        if results_df.empty:
            print("\n⚠️  Нет данных для отображения таблицы")
            print("=" * 80)
            return
        
        # Создаем сводную таблицу
        pivot = results_df.pivot_table(
            index="horizon_minutes",
            columns="threshold_percent",
            values="percentage",
            aggfunc="first",
        )
        
        print(f"\n{'Горизонт (мин)':<15}", end="")
        for threshold in sorted(results_df["threshold_percent"].unique()):
            print(f"{threshold:>8.2f}%", end="")
        print()
        print("-" * 80)
        
        for horizon in sorted(pivot.index):
            print(f"{horizon:<15}", end="")
            for threshold in sorted(results_df["threshold_percent"].unique()):
                value = pivot.loc[horizon, threshold]
                if pd.notna(value):
                    print(f"{value:>8.2f}%", end="")
                else:
                    print(f"{'N/A':>8}", end="")
            print()
        
        print("=" * 80)
    
    def print_recommendations(self, recommendations: Dict, target_percentage_range: Tuple[float, float] = (20.0, 30.0)):
        """Выводит рекомендации."""
        print("\n" + "=" * 80)
        print("РЕКОМЕНДАЦИИ")
        print("=" * 80)
        
        if recommendations["threshold_recommendations"]:
            print("\n📊 Рекомендуемые комбинации (по горизонтам):")
            print("-" * 80)
            for rec in recommendations["threshold_recommendations"]:
                print(
                    f"  Горизонт: {rec['horizon_minutes']} мин | "
                    f"Порог: {rec['threshold_percent']:.3f}% | "
                    f"Доля up/down: {rec['percentage']:.2f}% "
                    f"({rec['exceeded_count']}/{rec['total_count']} свечей)"
                )
        
        if recommendations["horizon_recommendations"]:
            print("\n📈 Рекомендуемые комбинации (по порогам):")
            print("-" * 80)
            for rec in recommendations["horizon_recommendations"]:
                print(
                    f"  Порог: {rec['threshold_percent']:.3f}% | "
                    f"Горизонт: {rec['horizon_minutes']} мин | "
                    f"Доля up/down: {rec['percentage']:.2f}% "
                    f"({rec['exceeded_count']}/{rec['total_count']} свечей)"
                )
        
        # Лучшая рекомендация - выбираем вариант, ближайший к середине целевого диапазона
        all_recommendations = recommendations["threshold_recommendations"] + recommendations["horizon_recommendations"]
        if all_recommendations:
            # Вычисляем середину целевого диапазона
            target_mid = (target_percentage_range[0] + target_percentage_range[1]) / 2
            # Вычисляем расстояние до середины для каждого варианта
            for rec in all_recommendations:
                rec["distance_to_mid"] = abs(rec["percentage"] - target_mid)
            # Выбираем вариант с минимальным расстоянием
            best = min(all_recommendations, key=lambda x: x["distance_to_mid"])
            
            print("\n" + "=" * 80)
            print("🎯 ЛУЧШАЯ РЕКОМЕНДАЦИЯ:")
            print("=" * 80)
            print(f"  MODEL_PREDICTION_HORIZON_SECONDS = {best['horizon_minutes'] * 60}")
            print(f"  MODEL_CLASSIFICATION_THRESHOLD = {best['threshold_percent'] / 100:.4f}")
            print(f"\n  Ожидаемая доля up/down классов: {best['percentage']:.2f}%")
            print(f"  Ожидаемая доля flat класса: {100 - best['percentage']:.2f}%")
            print("=" * 80)


async def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Анализ данных свечей для рекомендации threshold и horizon"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Торговая пара (по умолчанию: BTCUSDT)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Начальная дата (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Конечная дата (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Горизонты в минутах через запятую (опционально, по умолчанию: автоматический поиск от 1 до 60 минут)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Пороги в процентах через запятую (опционально, по умолчанию: автоматический поиск от 0.1% до 5%)",
    )
    parser.add_argument(
        "--target-percentage",
        type=str,
        default="20,30",
        help="Целевой диапазон процента up/down (min,max) (по умолчанию: 20,30)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Путь к данным (по умолчанию из конфига)",
    )
    
    args = parser.parse_args()
    
    # Парсим аргументы
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    end_date = end_date.replace(hour=23, minute=59, second=59)
    
    # Определяем горизонты: если не указаны, используем автоматический поиск
    if args.horizons:
        horizons_minutes = [int(x.strip()) for x in args.horizons.split(",")]
    else:
        # Автоматический поиск: от 1 до 60 минут с разными шагами для разных диапазонов
        horizons_minutes = (
            list(range(1, 6)) +      # 1-5 мин (шаг 1)
            list(range(5, 16, 2)) +  # 5-15 мин (шаг 2)
            list(range(15, 31, 5)) + # 15-30 мин (шаг 5)
            list(range(30, 61, 10))  # 30-60 мин (шаг 10)
        )
        # Удаляем дубликаты и сортируем
        horizons_minutes = sorted(list(set(horizons_minutes)))
    
    # Определяем пороги: если не указаны, используем автоматический поиск
    if args.thresholds:
        thresholds_pct = [float(x.strip()) for x in args.thresholds.split(",")]
    else:
        # Автоматический поиск: от 0.1% до 5% с разными шагами
        thresholds_pct = (
            [0.1, 0.2, 0.3] +                    # 0.1-0.3% (шаг 0.1)
            [0.5, 0.7, 1.0] +                    # 0.5-1.0% (шаг 0.2-0.3)
            [1.5, 2.0, 2.5, 3.0] +               # 1.5-3.0% (шаг 0.5)
            [4.0, 5.0]                           # 4.0-5.0% (шаг 1.0)
        )
    target_percentage = tuple(float(x.strip()) for x in args.target_percentage.split(","))
    
    print("=" * 80)
    print("АНАЛИЗ ДАННЫХ СВЕЧЕЙ ДЛЯ РЕКОМЕНДАЦИИ ПАРАМЕТРОВ КЛАССИФИКАЦИИ")
    print("=" * 80)
    print(f"\nПараметры:")
    print(f"  Символ: {args.symbol}")
    print(f"  Период: {start_date.date()} - {end_date.date()}")
    if args.horizons:
        print(f"  Горизонты (заданы вручную): {horizons_minutes} минут")
    else:
        print(f"  Горизонты (автоматический поиск): {len(horizons_minutes)} значений от {min(horizons_minutes)} до {max(horizons_minutes)} минут")
    if args.thresholds:
        print(f"  Пороги (заданы вручную): {thresholds_pct}%")
    else:
        print(f"  Пороги (автоматический поиск): {len(thresholds_pct)} значений от {min(thresholds_pct)}% до {max(thresholds_pct)}%")
    print(f"  Целевой диапазон up/down: {target_percentage[0]}-{target_percentage[1]}%")
    print()
    
    # Создаем анализатор
    analyzer = ThresholdHorizonAnalyzer(data_path=args.data_path)
    
    try:
        # Загружаем данные
        df = await analyzer.load_klines_data(args.symbol, start_date, end_date)
        
        # Вычисляем изменения цены
        print("\nВычисление изменений цены за разные горизонты...")
        df = analyzer.calculate_price_changes(df, horizons_minutes)
        
        # Вычисляем доли превышения порогов
        print("Вычисление долей превышения порогов...")
        results_df = analyzer.calculate_threshold_percentages(
            df, horizons_minutes, thresholds_pct
        )
        
        # Выводим таблицу (если есть данные)
        if not results_df.empty:
            analyzer.print_results_table(results_df)
            
            # Получаем рекомендации
            recommendations = analyzer.recommend_parameters(
                results_df, target_percentage_range=target_percentage
            )
            
            # Выводим рекомендации
            analyzer.print_recommendations(recommendations, target_percentage_range=target_percentage)
        else:
            print("\n⚠️  Не удалось вычислить результаты. Возможные причины:")
            print("   - Недостаточно данных за указанный период")
            print("   - Данные не содержат необходимые колонки (timestamp, open, close)")
            print("   - Проблемы с форматом данных")
            print(f"   - Загружено свечей: {len(df)}")
            if not df.empty:
                print(f"   - После вычисления изменений цены: {len(df)} строк")
        
        print("\n✅ Анализ завершен успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

