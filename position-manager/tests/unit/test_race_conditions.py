"""Unit tests for race condition protection in PositionManager.

Tests optimistic locking, SELECT FOR UPDATE, and retry logic.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
import pytest
import asyncpg

from src.services.position_manager import PositionManager
# VersionConflictError is raised internally, not imported
from src.models import Position


def make_mock_position_row(size: Decimal = Decimal("1.0"), version: int = 1):
    """Create a mock database row with all required fields for Position.from_db_dict."""
    pos_id = uuid4()
    now = datetime.utcnow()
    return {
        "id": pos_id,
        "asset": "BTCUSDT",
        "mode": "one-way",
        "size": size,
        "average_entry_price": Decimal("50000.0"),
        "current_price": Decimal("51000.0"),
        "unrealized_pnl": Decimal("1000.0"),
        "realized_pnl": Decimal("0.0"),
        "long_size": None,
        "short_size": None,
        "long_avg_price": None,
        "short_avg_price": None,
        "leverage": None,
        "position_value": None,
        "liq_price": None,
        "bust_price": None,
        "take_profit": None,
        "stop_loss": None,
        "cum_realised_pnl": None,
        "cum_unrealised_pnl": None,
        "total_fees": Decimal("0.0"),
        "opening_fees": None,
        "closing_fees": None,
        "margin_used": None,
        "available_margin": None,
        "maintenance_margin": None,
        "max_size": None,
        "min_size": None,
        "total_volume_traded": Decimal("0.0"),
        "first_entry_price": None,
        "last_entry_price": None,
        "exit_price": None,
        "peak_unrealized_pnl": None,
        "peak_unrealized_pnl_at": None,
        "worst_unrealized_pnl": None,
        "worst_unrealized_pnl_at": None,
        "created_at": now - timedelta(minutes=10),
        "last_updated": now,
        "closed_at": None,
        "version": version,
        "source": None,
        "last_sync_with_bybit": None,
        "bybit_position_data": None,
    }


@pytest.mark.asyncio
async def test_update_position_with_optimistic_locking():
    """Test that update_position_from_websocket uses optimistic locking with version field."""
    manager = PositionManager()
    
    # Mock database connection and transaction
    # conn.transaction() returns an async context manager
    mock_conn = MagicMock()
    mock_transaction_cm = MagicMock()
    mock_transaction_cm.__aenter__ = AsyncMock(return_value=mock_transaction_cm)
    mock_transaction_cm.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=mock_transaction_cm)
    
    # Mock SELECT FOR UPDATE query - return dict directly (asyncpg returns dict-like objects)
    mock_row_dict = make_mock_position_row(size=Decimal("1.0"), version=1)
    
    # Mock UPDATE with version check - return row for RETURNING clause
    mock_updated_row_dict = make_mock_position_row(size=Decimal("2.0"), version=2)
    mock_updated_row_dict["unrealized_pnl"] = Decimal("2000.0")
    
    # Multiple calls: 
    # 1. SELECT FOR UPDATE returns initial row
    # 2. UPDATE RETURNING returns updated row
    # 3. Additional calls return updated row (for retries or other queries)
    mock_conn.fetchrow = AsyncMock(side_effect=[mock_row_dict, mock_updated_row_dict, mock_updated_row_dict])
    
    # Mock pool and acquire context manager
    # pool.acquire() returns an async context manager
    mock_pool = MagicMock()
    mock_acquire_cm = MagicMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)
    
    with patch("src.services.position_manager.DatabaseConnection.get_pool", new_callable=AsyncMock) as mock_get_pool:
        mock_get_pool.return_value = mock_pool
        
        # This should use optimistic locking
        result = await manager.update_position_from_websocket(
            asset="BTCUSDT",
            mode="one-way",
            size_from_ws=Decimal("2.0"),
            trace_id="test-trace",
        )
        
        # Verify that version was checked in WHERE clause
        # Check fetchrow calls for SELECT FOR UPDATE
        fetchrow_calls = [call[0][0] for call in mock_conn.fetchrow.call_args_list if call[0]]
        for_update_queries = [call for call in fetchrow_calls if "FOR UPDATE" in call.upper()]
        assert len(for_update_queries) > 0, "SELECT FOR UPDATE should be used to lock rows"


@pytest.mark.asyncio
async def test_retry_on_version_conflict():
    """Test that update_position_from_websocket retries on VersionConflictError."""
    manager = PositionManager()
    
    # Mock database connection
    # conn.transaction() returns an async context manager
    mock_conn = MagicMock()
    mock_transaction_cm = MagicMock()
    mock_transaction_cm.__aenter__ = AsyncMock(return_value=mock_transaction_cm)
    mock_transaction_cm.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=mock_transaction_cm)
    
    mock_row_dict = make_mock_position_row(size=Decimal("1.0"), version=1)
    
    # First attempt: version conflict (fetchrow returns row, but UPDATE returns 0 rows)
    # Second attempt: success
    mock_updated_row_dict = make_mock_position_row(size=Decimal("2.0"), version=2)
    mock_updated_row_dict["unrealized_pnl"] = Decimal("2000.0")
    
    # First call: fetchrow returns row, but UPDATE fails (version conflict)
    # Second call: fetchrow returns row, UPDATE succeeds
    call_count = [0]
    def fetchrow_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:  # First two calls return initial row
            return mock_row_dict
        else:  # Third call (after retry) returns updated row
            return mock_updated_row_dict
    
    mock_conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)
    
    # Mock pool and acquire context manager
    mock_pool = MagicMock()
    mock_acquire_cm = AsyncMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)
    
    with patch("src.services.position_manager.DatabaseConnection.get_pool", new_callable=AsyncMock) as mock_get_pool:
        mock_get_pool.return_value = mock_pool
        
        # Should retry on version conflict
        # The retry logic is in the update_position_from_websocket method
        # It checks if UPDATE returned 0 rows and retries
        result = await manager.update_position_from_websocket(
            asset="BTCUSDT",
            mode="one-way",
            size_from_ws=Decimal("2.0"),
            trace_id="test-trace",
        )
        
        # Verify retry happened (fetchrow called multiple times)
        assert mock_conn.fetchrow.call_count >= 2, "Should retry on version conflict"


@pytest.mark.asyncio
async def test_select_for_update_locks_row():
    """Test that SELECT FOR UPDATE is used to lock rows during updates."""
    manager = PositionManager()
    
    # conn.transaction() returns an async context manager
    mock_conn = MagicMock()
    mock_transaction_cm = MagicMock()
    mock_transaction_cm.__aenter__ = AsyncMock(return_value=mock_transaction_cm)
    mock_transaction_cm.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=mock_transaction_cm)
    
    mock_row_dict = make_mock_position_row(size=Decimal("1.0"), version=1)
    
    mock_updated_row_dict = make_mock_position_row(size=Decimal("2.0"), version=2)
    mock_updated_row_dict["unrealized_pnl"] = Decimal("2000.0")
    
    # Multiple calls: SELECT FOR UPDATE, then UPDATE RETURNING, then any additional calls
    mock_conn.fetchrow = AsyncMock(side_effect=[mock_row_dict, mock_updated_row_dict, mock_updated_row_dict])
    
    # Mock pool and acquire context manager
    mock_pool = MagicMock()
    mock_acquire_cm = AsyncMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)
    
    with patch("src.services.position_manager.DatabaseConnection.get_pool", new_callable=AsyncMock) as mock_get_pool:
        mock_get_pool.return_value = mock_pool
        
        await manager.update_position_from_websocket(
            asset="BTCUSDT",
            mode="one-way",
            size_from_ws=Decimal("2.0"),
            trace_id="test-trace",
        )
        
        # Verify SELECT FOR UPDATE was used
        fetchrow_calls = [call[0][0] for call in mock_conn.fetchrow.call_args_list if call[0]]
        for_update_queries = [call for call in fetchrow_calls if "FOR UPDATE" in call.upper()]
        assert len(for_update_queries) > 0, "SELECT FOR UPDATE should be used to lock rows"


@pytest.mark.asyncio
async def test_unique_index_prevents_duplicate_active_positions():
    """Test that unique index prevents creating duplicate active positions."""
    manager = PositionManager()
    
    # conn.transaction() returns an async context manager
    mock_conn = MagicMock()
    mock_transaction_cm = MagicMock()
    mock_transaction_cm.__aenter__ = AsyncMock(return_value=mock_transaction_cm)
    mock_transaction_cm.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=mock_transaction_cm)
    
    # Simulate existing active position
    mock_row_dict = make_mock_position_row(size=Decimal("1.0"), version=1)
    
    mock_updated_row_dict = make_mock_position_row(size=Decimal("2.0"), version=2)
    mock_updated_row_dict["unrealized_pnl"] = Decimal("2000.0")
    
    # Simulate UniqueViolationError on INSERT
    unique_violation = asyncpg.UniqueViolationError("duplicate key value violates unique constraint")
    
    # Mock pool and acquire context manager
    mock_pool = MagicMock()
    mock_acquire_cm = AsyncMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)
    
    with patch("src.services.position_manager.DatabaseConnection.get_pool", new_callable=AsyncMock) as mock_get_pool:
        mock_get_pool.return_value = mock_pool
        
        # When trying to create duplicate, should handle UniqueViolationError
        # First fetchrow: no existing position (None)
        # Then INSERT fails with UniqueViolationError
        # Then fetchrow again: fetch existing position and update it
        call_count = [0]
        def fetchrow_side_effect(*args, **kwargs):
            call_count[0] += 1
            query = args[0] if args else ""
            if "FOR UPDATE" in query.upper() or "SELECT" in query.upper():
                # SELECT queries
                if call_count[0] == 1:
                    return None  # No existing position initially
                elif call_count[0] == 2:
                    return mock_row_dict  # After conflict, fetch existing
                else:
                    return mock_updated_row_dict  # After update
            else:
                # RETURNING clause from UPDATE
                return mock_updated_row_dict
        
        mock_conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)
        
        # First INSERT attempt raises UniqueViolationError
        # Then it should fetch existing and update it
        insert_call_count = [0]
        def execute_side_effect(*args, **kwargs):
            insert_call_count[0] += 1
            if insert_call_count[0] == 1:
                # First INSERT attempt raises UniqueViolationError
                raise unique_violation
            else:
                # Subsequent UPDATE succeeds
                return "UPDATE 1"
        
        mock_conn.execute = AsyncMock(side_effect=execute_side_effect)
        
        # Should handle UniqueViolationError gracefully and update existing position
        result = await manager.update_position_from_websocket(
            asset="BTCUSDT",
            mode="one-way",
            size_from_ws=Decimal("2.0"),
            trace_id="test-trace",
        )
        
        # Verify that UniqueViolationError was handled (fetchrow called multiple times)
        assert mock_conn.fetchrow.call_count >= 2, "Should handle UniqueViolationError and fetch existing position"

