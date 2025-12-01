"""
Intelligent signal generation service.

Uses model inference to generate trading signals with confidence scores,
applies quality thresholds, and integrates with rate limiting and validation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal

from ..models.signal import TradingSignal, MarketDataSnapshot
from ..models.position_state import OrderPositionState
from ..services.model_loader import model_loader
from ..services.model_inference import model_inference
from ..services.signal_validator import signal_validator
from ..services.rate_limiter import rate_limiter
from ..services.balance_calculator import balance_calculator
from ..services.position_manager_client import position_manager_client
from ..database.repositories.position_state_repo import PositionStateRepository
from ..config.settings import settings
from ..config.logging import get_logger, bind_context
from ..config.exceptions import ModelInferenceError, SignalValidationError
from ..services.signal_skip_metrics import signal_skip_metrics

logger = get_logger(__name__)


class IntelligentSignalGenerator:
    """Generates trading signals using trained ML models."""

    def __init__(
        self,
        min_confidence_threshold: float = 0.6,
        min_amount: float = 100.0,
        max_amount: float = 1000.0,
    ):
        """
        Initialize intelligent signal generator.

        Args:
            min_confidence_threshold: Minimum confidence score to generate signal (0-1)
            min_amount: Minimum order amount in quote currency
            max_amount: Maximum order amount in quote currency
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.position_state_repo = PositionStateRepository()

    async def generate_signal(
        self,
        asset: str,
        strategy_id: str,
        model_version: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal using model inference.

        Args:
            asset: Trading pair symbol (e.g., 'BTCUSDT')
            strategy_id: Trading strategy identifier
            model_version: Model version to use (None for active model)
            trace_id: Trace ID for request flow tracking

        Returns:
            TradingSignal or None if signal generation fails or confidence too low
        """
        bind_context(strategy_id=strategy_id, asset=asset, trace_id=trace_id)

        logger.info("Generating intelligent signal", asset=asset, strategy_id=strategy_id, model_version=model_version)

        # Check rate limit
        allowed, reason = rate_limiter.check_rate_limit()
        if not allowed:
            logger.warning("Signal generation rate limited", asset=asset, strategy_id=strategy_id, reason=reason)
            return None

        # Get order/position state early (needed for both open order check and feature engineering)
        order_position_state = await self.position_state_repo.get_order_position_state(
            strategy_id=strategy_id,
            asset=asset,
        )

        # Risk Management: Check take profit rule (before model inference)
        take_profit_signal = await self._check_take_profit_rule(asset, strategy_id, trace_id)
        if take_profit_signal:
            logger.info(
                "Take profit rule triggered, generating SELL signal",
                asset=asset,
                strategy_id=strategy_id,
                trace_id=trace_id,
            )
            return take_profit_signal

        # Check for open orders if configured (before inference)
        # If check_opposite_orders_only is true, we'll check again after determining signal_type
        if settings.signal_generation_skip_if_open_order and not settings.signal_generation_check_opposite_orders_only:
            open_order_check = await self._check_open_orders_for_state(
                order_position_state=order_position_state,
                asset=asset,
                strategy_id=strategy_id,
            )
            if open_order_check["should_skip"]:
                reason = open_order_check.get("reason", "open_order_exists")
                # Record skip metrics
                signal_skip_metrics.record_skip(
                    asset=asset,
                    strategy_id=strategy_id,
                    reason=reason,
                )
                logger.info(
                    "Skipping signal generation due to open order",
                    asset=asset,
                    strategy_id=strategy_id,
                    existing_order_id=open_order_check.get("existing_order_id"),
                    order_status=open_order_check.get("order_status"),
                    reason=reason,
                    trace_id=trace_id,
                )
                return None

        try:
            # Load model
            if model_version:
                model = await model_loader.load_model_by_version(model_version)
            else:
                model = await model_loader.load_active_model(strategy_id=strategy_id)

            if not model:
                logger.warning("No active model available", asset=asset, strategy_id=strategy_id, model_version=model_version)
                return None

            # Get market data snapshot at inference time
            market_data_snapshot = model_inference.get_market_data_snapshot(asset)
            if not market_data_snapshot:
                logger.warning("Market data unavailable, skipping signal generation", asset=asset, strategy_id=strategy_id)
                return None

            # Prepare features
            features_df = model_inference.prepare_features(
                asset=asset,
                order_position_state=order_position_state,
                market_data_snapshot=market_data_snapshot,
            )

            # Run model prediction
            prediction_result = model_inference.predict(model, features_df)

            # Check confidence threshold
            confidence = prediction_result.get("confidence", 0.0)
            if confidence < self.min_confidence_threshold:
                logger.debug(
                    "Signal confidence below threshold",
                    asset=asset,
                    strategy_id=strategy_id,
                    confidence=confidence,
                    threshold=self.min_confidence_threshold,
                )
                return None

            # Determine signal type from prediction
            signal_type = self._determine_signal_type(prediction_result)

            # Risk Management: Check position size limit for BUY signals (before generating signal)
            if signal_type.lower() == "buy":
                position_size_check = await self._check_position_size_limit(asset, strategy_id, trace_id)
                if position_size_check["should_skip"]:
                    reason = position_size_check.get("reason", "position_size_limit")
                    signal_skip_metrics.record_skip(
                        asset=asset,
                        strategy_id=strategy_id,
                        reason=reason,
                    )
                    logger.info(
                        "Skipping BUY signal due to position size limit",
                        asset=asset,
                        strategy_id=strategy_id,
                        position_size_norm=position_size_check.get("position_size_norm"),
                        max_ratio_threshold=position_size_check.get("max_ratio_threshold"),
                        reason=reason,
                        trace_id=trace_id,
                    )
                    return None

            # Check for opposite orders if configured (after determining signal_type)
            if settings.signal_generation_skip_if_open_order and settings.signal_generation_check_opposite_orders_only:
                open_order_check = await self._check_open_orders_for_state(
                    order_position_state=order_position_state,
                    asset=asset,
                    strategy_id=strategy_id,
                    signal_type=signal_type,
                )
                if open_order_check["should_skip"]:
                    reason = open_order_check.get("reason", "opposite_order_exists")
                    # Record skip metrics
                    signal_skip_metrics.record_skip(
                        asset=asset,
                        strategy_id=strategy_id,
                        reason=reason,
                    )
                    logger.info(
                        "Skipping signal generation due to opposite open order",
                        asset=asset,
                        strategy_id=strategy_id,
                        signal_type=signal_type,
                        existing_order_id=open_order_check.get("existing_order_id"),
                        order_status=open_order_check.get("order_status"),
                        reason=reason,
                        trace_id=trace_id,
                    )
                    return None

            # Calculate order amount from model
            model_amount = self._calculate_amount(asset, order_position_state, market_data_snapshot, confidence)
            
            # Check available balance and adapt amount
            adapted_amount = await balance_calculator.calculate_affordable_amount(
                trading_pair=asset,
                signal_type=signal_type,
                requested_amount=model_amount,
            )
            
            if adapted_amount is None:
                logger.warning(
                    "Insufficient balance, skipping signal generation",
                    asset=asset,
                    strategy_id=strategy_id,
                    signal_type=signal_type,
                    original_model_amount=model_amount,
                    trace_id=trace_id,
                )
                # Record skip metrics
                signal_skip_metrics.record_skip(
                    asset=asset,
                    strategy_id=strategy_id,
                    reason="insufficient_balance",
                )
                return None
            
            # Use adapted amount
            amount = adapted_amount
            if amount != model_amount:
                logger.info(
                    "Adapted signal amount to available balance",
                    asset=asset,
                    strategy_id=strategy_id,
                    original_model_amount=model_amount,
                    adapted_amount=amount,
                    trace_id=trace_id,
                )

            # Get model version string for signal
            if not model_version:
                # Get active model version from database
                from ..database.repositories.model_version_repo import ModelVersionRepository
                model_version_repo = ModelVersionRepository()
                active_model = await model_version_repo.get_active_by_strategy(strategy_id)
                model_version = active_model["version"] if active_model else None

            # Create signal
            signal = TradingSignal(
                signal_type=signal_type,
                asset=asset,
                amount=amount,
                confidence=confidence,
                strategy_id=strategy_id,
                model_version=model_version,
                is_warmup=False,
                market_data_snapshot=market_data_snapshot,
                metadata={
                    "reasoning": f"Model prediction: {prediction_result.get('prediction')}",
                    "prediction_result": {
                        "prediction": prediction_result.get("prediction"),
                        "buy_probability": prediction_result.get("buy_probability"),
                        "sell_probability": prediction_result.get("sell_probability"),
                    },
                    "model_version": model_version,
                    "inference_timestamp": datetime.utcnow().isoformat() + "Z",
                },
                trace_id=trace_id,
            )

            # Validate signal
            is_valid, errors = signal_validator.validate(signal)
            if not is_valid:
                logger.warning(
                    "Generated signal failed validation",
                    asset=asset,
                    strategy_id=strategy_id,
                    errors=errors,
                )
                return None

            logger.info(
                "Generated intelligent signal",
                asset=asset,
                strategy_id=strategy_id,
                signal_type=signal_type,
                amount=amount,
                confidence=confidence,
                model_version=model_version,
                trace_id=trace_id,
            )

            return signal

        except ModelInferenceError as e:
            logger.error("Model inference error during signal generation", asset=asset, strategy_id=strategy_id, error=str(e), exc_info=True)
            return None
        except Exception as e:
            logger.error("Error generating intelligent signal", asset=asset, strategy_id=strategy_id, error=str(e), exc_info=True)
            return None

    async def _check_open_orders_for_state(
        self,
        order_position_state: OrderPositionState,
        asset: str,
        strategy_id: str,
        signal_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check if there are open orders that should prevent signal generation.

        Args:
            order_position_state: Order position state (already fetched)
            asset: Trading pair symbol
            strategy_id: Trading strategy identifier
            signal_type: Signal type ('buy' or 'sell') - None means check all orders

        Returns:
            Dictionary with:
                - should_skip: bool - whether to skip signal generation
                - reason: str - reason for skipping (if should_skip is True)
                - existing_order_id: str - ID of existing order (if found)
                - order_status: str - status of existing order (if found)
        """
        # Filter open orders (pending or partially filled) for the asset
        open_orders = [
            order
            for order in order_position_state.orders
            if order.status in ("pending", "partially_filled") and order.asset == asset
        ]

        if not open_orders:
            return {"should_skip": False}

        # If checking opposite orders only, filter by signal direction
        if settings.signal_generation_check_opposite_orders_only and signal_type:
            opposite_side = "SELL" if signal_type.lower() == "buy" else "BUY"
            matching_orders = [order for order in open_orders if order.side.upper() == opposite_side]
            if not matching_orders:
                return {"should_skip": False}
            # Use the first matching order for logging
            existing_order = matching_orders[0]
            return {
                "should_skip": True,
                "reason": f"Open {opposite_side.lower()} order exists for asset {asset}",
                "existing_order_id": existing_order.order_id,
                "order_status": existing_order.status,
            }

        # Check all orders (default behavior)
        existing_order = open_orders[0]
        return {
            "should_skip": True,
            "reason": f"Open order exists for asset {asset}",
            "existing_order_id": existing_order.order_id,
            "order_status": existing_order.status,
        }

    def _determine_signal_type(self, prediction_result: Dict[str, Any]) -> str:
        """
        Determine signal type from model prediction.

        Args:
            prediction_result: Prediction result from model inference

        Returns:
            'buy' or 'sell'
        """
        # For classification models, use buy/sell probabilities
        buy_probability = prediction_result.get("buy_probability", 0.0)
        sell_probability = prediction_result.get("sell_probability", 0.0)

        if buy_probability > sell_probability:
            return "buy"
        else:
            return "sell"

    def _calculate_amount(
        self,
        asset: str,
        order_position_state: OrderPositionState,
        market_data_snapshot: MarketDataSnapshot,
        confidence: float,
    ) -> float:
        """
        Calculate order amount based on confidence and position state.

        Args:
            asset: Trading pair symbol
            order_position_state: Current order and position state
            market_data_snapshot: Market data snapshot
            confidence: Signal confidence score

        Returns:
            Order amount in quote currency
        """
        # Base amount
        base_amount = (self.min_amount + self.max_amount) / 2

        # Adjust based on confidence (higher confidence = larger amount, up to max)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
        amount = base_amount * confidence_multiplier

        # Adjust based on position state (reduce if already have large position)
        position = order_position_state.get_position(asset)
        if position and position.size != 0:
            # Reduce amount if we already have a position
            position_size_abs = abs(float(position.size))
            if position_size_abs > 0:
                # Reduce amount proportionally to existing position
                reduction_factor = min(1.0, 0.5 / (position_size_abs / base_amount + 0.5))
                amount = amount * reduction_factor

        # Clamp to valid range
        amount = max(self.min_amount, min(self.max_amount, amount))

        return round(amount, 2)

    async def _check_take_profit_rule(
        self,
        asset: str,
        strategy_id: str,
        trace_id: Optional[str] = None,
    ) -> Optional[TradingSignal]:
        """
        Check take profit rule and generate SELL signal if threshold exceeded.

        Args:
            asset: Trading pair symbol
            strategy_id: Trading strategy identifier
            trace_id: Trace ID for request flow tracking

        Returns:
            TradingSignal (SELL) if take profit triggered, None otherwise
        """
        try:
            unrealized_pnl_pct = await position_manager_client.get_unrealized_pnl_pct(asset)
            if unrealized_pnl_pct is None:
                # No position or error - continue with normal flow
                return None

            take_profit_threshold = settings.model_service_take_profit_pct

            if unrealized_pnl_pct > take_profit_threshold:
                # Get position size to close
                position_size = await position_manager_client.get_position_size(asset)
                if position_size is None or position_size == 0:
                    logger.warning(
                        "Take profit triggered but position size unavailable",
                        asset=asset,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                    )
                    return None

                # Get market data snapshot
                market_data_snapshot = model_inference.get_market_data_snapshot(asset)
                if not market_data_snapshot:
                    logger.warning("Market data unavailable for take profit signal", asset=asset)
                    return None

                # Force generate SELL signal to close position
                signal = TradingSignal(
                    signal_type="sell",
                    asset=asset,
                    amount=position_size,  # Close entire position
                    confidence=1.0,  # Maximum confidence for take profit
                    strategy_id=strategy_id,
                    model_version=None,  # Not from model
                    is_warmup=False,
                    market_data_snapshot=market_data_snapshot,
                    metadata={
                        "reasoning": "take_profit_triggered",
                        "unrealized_pnl_pct": unrealized_pnl_pct,
                        "take_profit_threshold": take_profit_threshold,
                        "position_size": position_size,
                    },
                    trace_id=trace_id,
                )

                logger.info(
                    "Take profit rule triggered",
                    asset=asset,
                    strategy_id=strategy_id,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    take_profit_threshold=take_profit_threshold,
                    position_size=position_size,
                    trace_id=trace_id,
                )

                return signal

            return None

        except Exception as e:
            logger.error(
                "Error checking take profit rule",
                asset=asset,
                strategy_id=strategy_id,
                error=str(e),
                exc_info=True,
            )
            # Continue with normal flow on error
            return None

    async def _check_position_size_limit(
        self,
        asset: str,
        strategy_id: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check position size limit before generating BUY signal.

        Args:
            asset: Trading pair symbol
            strategy_id: Trading strategy identifier
            trace_id: Trace ID for request flow tracking

        Returns:
            Dictionary with:
                - should_skip: bool - whether to skip signal generation
                - reason: str - reason for skipping (if should_skip is True)
                - position_size_norm: float - current position size normalized
                - max_ratio_threshold: float - maximum allowed ratio
        """
        try:
            position_size_norm = await position_manager_client.get_position_size_norm(asset)
            if position_size_norm is None:
                # No position or error - allow signal generation
                return {"should_skip": False}

            max_ratio = settings.model_service_max_position_size_ratio

            if position_size_norm > max_ratio:
                return {
                    "should_skip": True,
                    "reason": "position_size_limit",
                    "position_size_norm": position_size_norm,
                    "max_ratio_threshold": max_ratio,
                }

            return {"should_skip": False}

        except Exception as e:
            logger.error(
                "Error checking position size limit",
                asset=asset,
                strategy_id=strategy_id,
                error=str(e),
                exc_info=True,
            )
            # On error, allow signal generation (fail open)
            return {"should_skip": False}

    async def generate_signals_for_strategies(
        self,
        assets: List[str],
        strategy_ids: List[str],
        trace_id: Optional[str] = None,
    ) -> List[TradingSignal]:
        """
        Generate intelligent signals for multiple strategies and assets.

        Args:
            assets: List of trading pair symbols
            strategy_ids: List of strategy identifiers
            trace_id: Trace ID for request flow tracking

        Returns:
            List of generated signals
        """
        signals = []
        for strategy_id in strategy_ids:
            for asset in assets:
                signal = await self.generate_signal(asset, strategy_id, trace_id=trace_id)
                if signal:
                    signals.append(signal)
        return signals


# Initialize intelligent signal generator with settings
def get_intelligent_generator():
    """Get intelligent signal generator instance with current settings."""
    return IntelligentSignalGenerator(
        min_confidence_threshold=settings.model_quality_threshold_accuracy,  # Use quality threshold as confidence threshold
        min_amount=settings.warmup_min_amount,
        max_amount=settings.warmup_max_amount,
    )


# Global intelligent signal generator instance
intelligent_signal_generator = get_intelligent_generator()

