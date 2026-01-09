from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes.positions import get_position_manager
from src.config.settings import settings as app_settings
from src.models import Position
from src.services.position_manager import PositionManager


# NOTE: This test file is deprecated. 
# The update_position_from_order_fill method has been removed.
# Positions are now updated only from WebSocket events, and position_orders
# are created by position_order_linker_consumer based on order events.
# 
# New E2E tests for unified positions architecture are in:
# - test_unified_positions_architecture.py


