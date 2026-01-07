"""Tests for intelligent_signal_generator with instrument_info_client integration."""

import pytest
from decimal import Decimal

from src.services.intelligent_signal_generator import IntelligentSignalGenerator


@pytest.mark.asyncio
async def test_calculate_amount_with_custom_min_amount():
    """Test that _calculate_amount uses provided min_amount parameter."""
    generator = IntelligentSignalGenerator(
        min_confidence_threshold=0.6,
        min_amount=100.0,  # Default
        max_amount=1000.0,
    )
    
    # Test with custom min_amount from instrument_info
    amount = generator._calculate_amount(
        current_price=3000.0,
        confidence=0.8,
        prediction_result=None,
        available_resource=None,
        min_amount=50.0,  # Custom min_amount from instrument_info
    )
    
    # Should use 50.0 as minimum, not 100.0
    assert amount is not None
    assert amount >= 50.0
    assert amount <= 1000.0
    
    # Test that it respects the custom min_amount
    amount2 = generator._calculate_amount(
        current_price=3000.0,
        confidence=0.8,
        prediction_result=None,
        available_resource=75.0,  # Between 50.0 and calculated amount
        min_amount=50.0,
    )
    
    # Should be limited to 75.0 (available_resource), which is >= 50.0
    assert amount2 is not None
    assert amount2 == 75.0
    
    # Test that it rejects if available_resource < min_amount
    amount3 = generator._calculate_amount(
        current_price=3000.0,
        confidence=0.8,
        prediction_result=None,
        available_resource=30.0,  # Less than min_amount (50.0)
        min_amount=50.0,
    )
    
    # Should return None because available_resource < min_amount
    assert amount3 is None

