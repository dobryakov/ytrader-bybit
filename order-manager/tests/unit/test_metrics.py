"""Unit tests for metrics utilities."""

import pytest
from src.utils.metrics import (
    record_latency,
    increment_counter,
    get_metrics_summary,
    reset_metrics,
    measure_latency,
)


def test_record_latency():
    """Test latency recording."""
    reset_metrics()
    record_latency("test_metric", 100.0, tag1="value1")
    summary = get_metrics_summary()
    assert "test_metric" in summary
    assert summary["test_metric"]["count"] == 1


def test_increment_counter():
    """Test counter increment."""
    reset_metrics()
    increment_counter("test_counter", value=5, tag="test")
    summary = get_metrics_summary()
    assert "counters" in summary
    assert len(summary["counters"]) > 0


def test_get_metrics_summary():
    """Test metrics summary retrieval."""
    reset_metrics()
    record_latency("test", 50.0)
    record_latency("test", 100.0)
    summary = get_metrics_summary()
    assert "test" in summary
    assert summary["test"]["min"] == 50.0
    assert summary["test"]["max"] == 100.0
    assert summary["test"]["avg"] == 75.0


@pytest.mark.asyncio
async def test_measure_latency_context_manager():
    """Test latency measurement context manager."""
    reset_metrics()
    async with measure_latency("async_test"):
        pass
    summary = get_metrics_summary()
    assert "async_test" in summary

