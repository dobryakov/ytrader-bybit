"""
Shared test fixtures for Feature Service tests.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

# Import all fixtures from fixtures modules so pytest can discover them
from tests.fixtures.market_data import (
    sample_orderbook_snapshot,
    sample_orderbook_delta,
    sample_trade,
    sample_kline,
    sample_ticker,
    sample_funding_rate,
    sample_market_data_sequence,
)
from tests.fixtures.orderbook import (
    sample_orderbook_state,
    sample_orderbook_deltas,
    sample_orderbook_desynchronized,
    sample_orderbook_empty,
)
from tests.fixtures.rolling_windows import (
    sample_rolling_windows,
    sample_rolling_windows_empty,
    sample_rolling_windows_trades,
    sample_rolling_windows_klines,
)
from tests.fixtures.feature_vectors import (
    sample_feature_vector,
    sample_feature_vector_minimal,
    sample_feature_vector_sequence,
    sample_feature_vector_multiple_symbols,
)
from tests.fixtures.market_data_streams import (
    sample_orderbook_stream,
    sample_trades_stream,
    sample_klines_stream,
    sample_ticker_stream,
    sample_funding_stream,
    sample_mixed_market_data_stream,
)
from tests.fixtures.datasets import (
    sample_dataset_metadata,
    sample_dataset_metadata_walk_forward,
    sample_dataset_metadata_building,
    sample_dataset_metadata_failed,
    sample_dataset_list,
)
from tests.fixtures.historical_data import (
    sample_historical_orderbook_snapshots,
    sample_historical_orderbook_deltas,
    sample_historical_trades,
    sample_historical_klines,
    sample_historical_ticker,
    sample_historical_funding,
    sample_parquet_file_orderbook,
    sample_parquet_file_trades,
    sample_parquet_directory_structure,
)
from tests.fixtures.targets import (
    sample_targets_regression,
    sample_targets_classification,
    sample_targets_risk_adjusted,
    sample_targets_with_leakage,
    sample_targets_no_leakage,
)


@pytest.fixture
def mock_logger():
    """Mock structured logger."""
    return MagicMock()


@pytest.fixture
def mock_db_pool():
    """Mock database connection pool."""
    pool = AsyncMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    return pool


@pytest.fixture
def mock_rabbitmq_connection():
    """Mock RabbitMQ connection."""
    connection = AsyncMock()
    channel = AsyncMock()
    connection.channel = AsyncMock(return_value=channel)
    return connection, channel


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for ws-gateway API."""
    client = AsyncMock()
    return client

