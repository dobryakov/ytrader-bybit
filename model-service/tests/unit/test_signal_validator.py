"""
Unit tests for SignalValidator.
"""

import pytest
from datetime import datetime, timedelta
from src.services.signal_validator import SignalValidator, SignalValidationError
from src.models.signal import TradingSignal, MarketDataSnapshot


@pytest.fixture
def validator():
    """Create SignalValidator instance for testing."""
    return SignalValidator(
        min_amount=10.0,
        max_amount=100000.0,
        max_timestamp_age_seconds=300,
    )


@pytest.fixture
def market_snapshot():
    """Create market data snapshot for testing."""
    return MarketDataSnapshot(
        price=3000.0,
        spread=1.0,
        volume_24h=1000000.0,
        volatility=0.02,
    )


def test_validate_buy_signal_valid(validator, market_snapshot):
    """Test validation of valid BUY signal."""
    signal = TradingSignal(
        signal_type="buy",
        asset="ETHUSDT",
        amount=1000.0,  # In USDT
        confidence=0.8,
        strategy_id="test-strategy",
        market_data_snapshot=market_snapshot,
    )
    
    is_valid, errors = validator.validate(signal)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_sell_signal_valid(validator, market_snapshot):
    """Test validation of valid SELL signal - amount is in USDT, not base currency."""
    signal = TradingSignal(
        signal_type="sell",
        asset="ETHUSDT",
        amount=1000.0,  # In USDT (not ETH)
        confidence=0.8,
        strategy_id="test-strategy",
        market_data_snapshot=market_snapshot,
    )
    
    is_valid, errors = validator.validate(signal)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_amount_below_minimum(validator, market_snapshot):
    """Test validation fails when amount is below minimum."""
    signal = TradingSignal(
        signal_type="buy",
        asset="ETHUSDT",
        amount=5.0,  # Below minimum 10.0
        confidence=0.8,
        strategy_id="test-strategy",
        market_data_snapshot=market_snapshot,
    )
    
    is_valid, errors = validator.validate(signal)
    assert is_valid is False
    assert any("below minimum" in error for error in errors)


def test_validate_amount_above_maximum(validator, market_snapshot):
    """Test validation fails when amount is above maximum."""
    signal = TradingSignal(
        signal_type="sell",
        asset="ETHUSDT",
        amount=200000.0,  # Above maximum 100000.0
        confidence=0.8,
        strategy_id="test-strategy",
        market_data_snapshot=market_snapshot,
    )
    
    is_valid, errors = validator.validate(signal)
    assert is_valid is False
    assert any("above maximum" in error for error in errors)


def test_validate_sell_amount_no_conversion(validator, market_snapshot):
    """
    Test that SELL signal amount is validated directly in USDT without conversion.
    
    This is important because the old code incorrectly multiplied SELL amount by price,
    assuming amount was in base currency. But amount is always in quote currency (USDT).
    """
    # Amount is 79.19 USDT (not ETH)
    # Old code would multiply by price: 79.19 * 3000 = 237570 USDT (incorrect)
    # New code validates directly: 79.19 USDT (correct)
    signal = TradingSignal(
        signal_type="sell",
        asset="ETHUSDT",
        amount=79.19,  # In USDT
        confidence=0.8,
        strategy_id="test-strategy",
        market_data_snapshot=market_snapshot,
    )
    
    is_valid, errors = validator.validate(signal)
    # Should be valid because 79.19 < 100000.0 (max_amount)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_sell_amount_large_usdt(validator, market_snapshot):
    """Test that large SELL amount in USDT is correctly validated."""
    # Large amount that would fail if incorrectly multiplied by price
    signal = TradingSignal(
        signal_type="sell",
        asset="ETHUSDT",
        amount=50000.0,  # In USDT
        confidence=0.8,
        strategy_id="test-strategy",
        market_data_snapshot=market_snapshot,
    )
    
    is_valid, errors = validator.validate(signal)
    # Should be valid because 50000.0 < 100000.0 (max_amount)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_invalid_signal_type(validator, market_snapshot):
    """Test validation fails for invalid signal type."""
    # Use model_construct to bypass Pydantic validation for testing
    signal = TradingSignal.model_construct(
        signal_type="hold",  # Invalid
        asset="ETHUSDT",
        amount=1000.0,
        confidence=0.8,
        strategy_id="test-strategy",
        market_data_snapshot=market_snapshot,
    )
    
    is_valid, errors = validator.validate(signal)
    assert is_valid is False
    assert any("Invalid signal_type" in error for error in errors)


def test_validate_invalid_confidence(validator, market_snapshot):
    """Test validation fails for invalid confidence."""
    # Use model_construct to bypass Pydantic validation for testing
    signal = TradingSignal.model_construct(
        signal_type="buy",
        asset="ETHUSDT",
        amount=1000.0,
        confidence=1.5,  # Invalid (> 1.0)
        strategy_id="test-strategy",
        market_data_snapshot=market_snapshot,
    )
    
    is_valid, errors = validator.validate(signal)
    assert is_valid is False
    assert any("Invalid confidence" in error for error in errors)


def test_validate_and_raise(validator, market_snapshot):
    """Test validate_and_raise raises exception for invalid signal."""
    signal = TradingSignal(
        signal_type="buy",
        asset="ETHUSDT",
        amount=5.0,  # Below minimum
        confidence=0.8,
        strategy_id="test-strategy",
        market_data_snapshot=market_snapshot,
    )
    
    with pytest.raises(SignalValidationError):
        validator.validate_and_raise(signal)

