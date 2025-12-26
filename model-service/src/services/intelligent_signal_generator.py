"""
Intelligent signal generation service.

Uses model inference to generate trading signals with confidence scores,
applies quality thresholds, and integrates with rate limiting and validation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import asyncio

from ..models.signal import TradingSignal, MarketDataSnapshot
from ..models.position_state import OrderPositionState
from ..models.feature_vector import FeatureVector
from ..services.model_loader import model_loader
from ..services.model_inference import model_inference
from ..services.signal_validator import signal_validator
from ..services.rate_limiter import rate_limiter
from ..services.balance_calculator import balance_calculator
from ..services.feature_service_client import feature_service_client
from ..services.feature_cache import feature_cache
from ..consumers.feature_consumer import feature_consumer
from ..database.repositories.position_state_repo import PositionStateRepository
from ..database.repositories.model_version_repo import ModelVersionRepository
from ..database.repositories.quality_metrics_repo import ModelQualityMetricsRepository
from ..config.settings import settings
from ..config.logging import get_logger, bind_context
from ..config.exceptions import ModelInferenceError, SignalValidationError
from ..services.signal_skip_metrics import signal_skip_metrics
from ..services.target_registry_client import target_registry_client
from ..database.repositories.prediction_target_repo import PredictionTargetRepository
from ..services.version_mismatch_handler import version_mismatch_handler
from uuid import UUID

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
        self.quality_metrics_repo = ModelQualityMetricsRepository()

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

        # Get order/position state early (needed for open order check)
        order_position_state = await self.position_state_repo.get_order_position_state(
            strategy_id=strategy_id,
            asset=asset,
        )

        # Check for open orders if configured (before inference)
        # If check_opposite_orders_only is true, we'll check again after determining signal_type
        if settings.signal_generation_skip_if_open_order and not settings.signal_generation_check_opposite_orders_only:
            logger.info("Checking for open orders", asset=asset, strategy_id=strategy_id, trace_id=trace_id)
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
            logger.info("No open orders blocking signal generation", asset=asset, strategy_id=strategy_id, trace_id=trace_id)

        try:
            # Get active model version from database first (needed for training_config)
            active_model = None
            if not model_version:
                # Get active model version from database (by strategy_id and symbol/asset)
                model_version_repo = ModelVersionRepository()
                active_model = await model_version_repo.get_active_by_strategy_and_symbol(strategy_id, asset)
                model_version = active_model["version"] if active_model else None

            # Load model (with symbol binding - asset is the symbol)
            logger.info("Loading model", asset=asset, strategy_id=strategy_id, model_version=model_version, trace_id=trace_id)
            if model_version:
                model = await model_loader.load_model_by_version(model_version)
            else:
                # Load active model for strategy_id and symbol (asset)
                model = await model_loader.load_active_model(strategy_id=strategy_id, symbol=asset)

            if not model:
                logger.warning("No active model available", asset=asset, strategy_id=strategy_id, model_version=model_version, trace_id=trace_id)
                return None
            logger.info("Model loaded successfully", asset=asset, strategy_id=strategy_id, model_version=model_version, trace_id=trace_id)

            # Get feature vector from Feature Service (via cache or REST API)
            logger.info("Getting feature vector", asset=asset, strategy_id=strategy_id, trace_id=trace_id)
            feature_vector = await self._get_feature_vector(asset, trace_id)
            if not feature_vector:
                logger.warning("Features unavailable, skipping signal generation", asset=asset, strategy_id=strategy_id, trace_id=trace_id)
                return None
            logger.info("Feature vector retrieved", asset=asset, strategy_id=strategy_id, feature_count=len(feature_vector.features) if feature_vector else 0, trace_id=trace_id)

            # Validate feature registry version compatibility
            # Get model's training config to check versions
            model_feature_registry_version = None
            model_target_registry_version = None
            training_config = None
            if active_model and active_model.get("training_config"):
                training_config = active_model["training_config"]
                if isinstance(training_config, str):
                    training_config = json.loads(training_config)
                model_feature_registry_version = training_config.get("feature_registry_version")
                model_target_registry_version = training_config.get("target_registry_version")

            if model_feature_registry_version and model_feature_registry_version != feature_vector.feature_registry_version:
                logger.warning(
                    "Feature registry version mismatch - model was trained on different version",
                    asset=asset,
                    strategy_id=strategy_id,
                    model_feature_registry_version=model_feature_registry_version,
                    feature_vector_registry_version=feature_vector.feature_registry_version,
                    recommendation="Model should be retrained on current feature registry version for optimal performance",
                    trace_id=trace_id,
                )
            elif model_feature_registry_version:
                logger.debug(
                    "Feature registry versions match",
                    asset=asset,
                    strategy_id=strategy_id,
                    feature_registry_version=model_feature_registry_version,
                    trace_id=trace_id,
                )

            # Prepare features from FeatureVector
            logger.info("Preparing features for model", asset=asset, strategy_id=strategy_id, trace_id=trace_id)
            features_df = model_inference.prepare_features(
                feature_vector=feature_vector,
                asset=asset,
            )
            logger.info("Features prepared", asset=asset, strategy_id=strategy_id, features_shape=features_df.shape if features_df is not None else None, trace_id=trace_id)

            # Run model prediction
            logger.info("Running model prediction", asset=asset, strategy_id=strategy_id, trace_id=trace_id)
            prediction_result = model_inference.predict(model, features_df)
            logger.info("Model prediction completed", asset=asset, strategy_id=strategy_id, prediction_result_keys=list(prediction_result.keys()) if prediction_result else None, trace_id=trace_id)

            # Get effective confidence threshold (from top-k analysis or fallback to static)
            effective_threshold = await self._get_effective_confidence_threshold(
                active_model=active_model,
                asset=asset,
                strategy_id=strategy_id,
                trace_id=trace_id,
            )

            # Check confidence threshold
            confidence = prediction_result.get("confidence", 0.0)
            threshold_source = "top_k" if effective_threshold != self.min_confidence_threshold else "static"
            logger.info(
                "Checking confidence threshold",
                asset=asset,
                strategy_id=strategy_id,
                confidence=confidence,
                threshold=effective_threshold,
                threshold_source=threshold_source,
                trace_id=trace_id,
            )
            
            # Determine signal type from prediction (before threshold check to have signal_type for rejected signals)
            signal_type = self._determine_signal_type(prediction_result)
            
            # Prepare raw prediction data for metadata (for all signals - valid and rejected)
            raw_prediction_metadata = {
                "prediction_result": {
                    "prediction": prediction_result.get("prediction"),
                    "buy_probability": prediction_result.get("buy_probability"),
                    "sell_probability": prediction_result.get("sell_probability"),
                    "confidence": confidence,
                    "probabilities": prediction_result.get("probabilities"),  # Raw probabilities array if available
                },
                "effective_threshold": effective_threshold,
                "threshold_source": threshold_source,
            }
            
            # Check if confidence is below threshold
            if confidence < effective_threshold:
                logger.info(
                    "Signal confidence below threshold - creating rejected signal",
                    asset=asset,
                    strategy_id=strategy_id,
                    confidence=confidence,
                    threshold=effective_threshold,
                    threshold_source=threshold_source,
                    trace_id=trace_id,
                )
                # Create rejected signal instead of returning None
                return await self._create_rejected_signal(
                    asset=asset,
                    strategy_id=strategy_id,
                    model_version=model_version,
                    confidence=confidence,
                    effective_threshold=effective_threshold,
                    rejection_reason="confidence_below_threshold",
                    prediction_result=prediction_result,
                    feature_vector=feature_vector,
                    active_model=active_model,
                    raw_prediction_metadata=raw_prediction_metadata,
                    trace_id=trace_id,
                )
            
            # If regression model returned None (HOLD), create rejected signal
            if signal_type is None:
                logger.debug(
                    "Regression model predicted HOLD (return within threshold range) - creating rejected signal",
                    asset=asset,
                    strategy_id=strategy_id,
                    predicted_return=prediction_result.get("prediction"),
                    threshold=settings.model_regression_threshold,
                    trace_id=trace_id,
                )
                return await self._create_rejected_signal(
                    asset=asset,
                    strategy_id=strategy_id,
                    model_version=model_version,
                    confidence=confidence,
                    effective_threshold=effective_threshold,
                    rejection_reason="regression_hold_prediction",
                    prediction_result=prediction_result,
                    feature_vector=feature_vector,
                    active_model=active_model,
                    raw_prediction_metadata=raw_prediction_metadata,
                    trace_id=trace_id,
                )
            
            # Log prediction details for debugging
            prediction = prediction_result.get("prediction")
            if isinstance(prediction, float):
                # Regression model
                logger.info(
                    "Model prediction (regression)",
                    asset=asset,
                    strategy_id=strategy_id,
                    predicted_return=prediction,
                    predicted_return_pct=prediction * 100,
                    determined_signal_type=signal_type,
                    confidence=confidence,
                    threshold=settings.model_regression_threshold,
                    trace_id=trace_id,
                )
            else:
                # Classification model
                buy_probability = prediction_result.get("buy_probability", 0.0)
                sell_probability = prediction_result.get("sell_probability", 0.0)
                logger.info(
                    "Model prediction probabilities",
                    asset=asset,
                    strategy_id=strategy_id,
                    buy_probability=buy_probability,
                    sell_probability=sell_probability,
                    determined_signal_type=signal_type,
                    confidence=confidence,
                    signal_direction_check=f"buy_prob({buy_probability:.4f}) {'>' if buy_probability > sell_probability else '<='} sell_prob({sell_probability:.4f})",
                    trace_id=trace_id,
                )

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

            # Extract price from feature vector for calculations
            current_price = float(feature_vector.features.get("mid_price", feature_vector.features.get("price", 0.0)))
            if current_price <= 0:
                logger.warning("Invalid price in feature vector, skipping signal generation", asset=asset, strategy_id=strategy_id, price=current_price)
                return None

            # Calculate order amount from model
            # Pass prediction_result for regression models to use predicted_return for position sizing
            model_amount = self._calculate_amount(
                current_price, 
                confidence,
                prediction_result=prediction_result
            )
            
            # Check available balance and adapt amount
            # For SELL signals, pass current_price to convert quote currency to base currency
            adapted_amount = await balance_calculator.calculate_affordable_amount(
                trading_pair=asset,
                signal_type=signal_type,
                requested_amount=model_amount,
                current_price=current_price,
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

            # active_model and model_version are already defined above
            # Create market data snapshot from feature vector for signal metadata
            market_data_snapshot = self._create_market_data_snapshot_from_features(feature_vector)

            # Get target registry info for metadata
            # Use target_registry_version from model's training_config if available (model was trained on specific target)
            # Otherwise fallback to active target registry version
            target_registry_version = None
            target_config = None
            prediction_horizon_seconds = None
            target_timestamp = None

            if active_model and active_model.get("training_config"):
                training_config = active_model["training_config"]
                if isinstance(training_config, str):
                    training_config = json.loads(training_config)
                target_registry_version = training_config.get("target_registry_version")
                logger.debug(
                    "Using target_registry_version from model training_config",
                    model_version=model_version,
                    target_registry_version=target_registry_version,
                )

            # Fallback to active target registry version if not in model config
            if not target_registry_version:
                active_target_registry_version = await target_registry_client.get_target_registry_version()
                if active_target_registry_version:
                    logger.warning(
                        "Using active target_registry_version (not found in model training_config) - version mismatch possible",
                        asset=asset,
                        strategy_id=strategy_id,
                        model_version=model_version,
                        active_target_registry_version=active_target_registry_version,
                        recommendation="Model training_config should include target_registry_version for consistency",
                        trace_id=trace_id,
                    )
                    target_registry_version = active_target_registry_version
                else:
                    logger.debug(
                        "No target_registry_version available (not in model config and no active version)",
                        asset=asset,
                        strategy_id=strategy_id,
                        trace_id=trace_id,
                    )

            if target_registry_version:
                target_config = await target_registry_client.get_target_config(target_registry_version)
                if target_config:
                    prediction_horizon_seconds = target_config.get("horizon", 0)
                    if prediction_horizon_seconds > 0:
                        target_timestamp = datetime.utcnow() + timedelta(seconds=prediction_horizon_seconds)

            # Check for version mismatches and trigger automatic retraining if needed
            # Do this after we have both model versions and current versions
            if model_feature_registry_version or model_target_registry_version:
                # Trigger automatic retraining in background (non-blocking)
                asyncio.create_task(
                    version_mismatch_handler.handle_version_mismatch(
                        strategy_id=strategy_id,
                        asset=asset,
                        model_feature_registry_version=model_feature_registry_version,
                        model_target_registry_version=model_target_registry_version,
                        current_feature_registry_version=feature_vector.feature_registry_version,
                        current_target_registry_version=target_registry_version,
                        trace_id=trace_id,
                    )
                )

            # Create signal with raw prediction data in metadata
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
                    # Include full raw prediction data in metadata
                    **raw_prediction_metadata,
                    "model_version": model_version,
                    "feature_registry_version": feature_vector.feature_registry_version,
                    "target_registry_version": target_registry_version,
                    "prediction_horizon_seconds": prediction_horizon_seconds,
                    "target_timestamp": target_timestamp.isoformat() + "Z" if target_timestamp else None,
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

            # Store prediction data in signal metadata for later saving after signal is persisted
            # This avoids race condition where prediction_targets is saved before trading_signals
            if not hasattr(signal, '_prediction_data'):
                signal._prediction_data = {
                    'prediction_result': prediction_result,
                    'feature_vector': feature_vector,
                    'model_version': model_version,
                    'trace_id': trace_id,
                }

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

    def _determine_signal_type(self, prediction_result: Dict[str, Any]) -> Optional[str]:
        """
        Determine signal type from model prediction.

        Supports both classification and regression models:
        - Classification: Uses buy/sell probabilities with hysteresis (min_probability_diff)
        - Regression: Converts predicted return to signal using threshold

        INVARIANT: For classification models without calibrated thresholds:
        - If |buy_probability - sell_probability| < min_probability_diff → HOLD (None)
        - This ensures hysteresis and prevents signals when model is uncertain
        - Confidence threshold is an additional filter, not a replacement for hysteresis

        Args:
            prediction_result: Prediction result from model inference
                - For classification: contains "buy_probability", "sell_probability"
                - For regression: contains "prediction" (float, predicted return value)

        Returns:
            'buy', 'sell', or None (for HOLD)
        """
        from ..config.settings import settings
        
        # Check if this is a regression model (prediction is a float, not int)
        prediction = prediction_result.get("prediction")
        
        if prediction is not None and isinstance(prediction, float):
            # Regression model: convert predicted return to signal
            predicted_return = float(prediction)
            threshold = settings.model_regression_threshold
            
            if predicted_return > threshold:
                return "buy"
            elif predicted_return < -threshold:
                return "sell"
            else:
                # HOLD: predicted return is within threshold range (hysteresis)
                return None
        
        # Classification model: use buy/sell probabilities
        buy_probability = prediction_result.get("buy_probability", 0.0)
        sell_probability = prediction_result.get("sell_probability", 0.0)
        
        # INVARIANT: Check if model has calibrated thresholds (provides its own hysteresis)
        # Models with calibrated thresholds already handle uncertainty in model_inference.py:
        # - If no threshold passed → semantic_prediction = 0 → buy_probability/sell_probability reflect this
        # - We check metadata to see if thresholds were used
        # For models without calibrated thresholds, apply min_probability_diff hysteresis
        
        # Check if model used calibrated thresholds (check metadata or prediction structure)
        # If model has calibrated thresholds and none passed, buy_probability and sell_probability
        # will be set, but the model already decided on HOLD (semantic_prediction = 0)
        # We can detect this by checking if prediction_result indicates threshold-based decision
        has_calibrated_thresholds = prediction_result.get("_used_calibrated_thresholds", False)
        
        # Apply min_probability_diff hysteresis for models without calibrated thresholds
        # INVARIANT: For classification without calibrated thresholds:
        # - If |buy_probability - sell_probability| < min_probability_diff → HOLD (None)
        # - This check applies BEFORE confidence threshold
        # - If difference is too small, signal is HOLD regardless of confidence
        if not has_calibrated_thresholds:
            min_probability_diff = settings.model_min_probability_diff
            probability_diff = abs(buy_probability - sell_probability)
            
            if probability_diff < min_probability_diff:
                # INVARIANT: HOLD when difference is too small (hysteresis)
                # This applies regardless of confidence threshold
                logger.debug(
                    "Signal type HOLD due to insufficient probability difference (hysteresis)",
                    buy_probability=buy_probability,
                    sell_probability=sell_probability,
                    probability_diff=probability_diff,
                    min_probability_diff=min_probability_diff,
                    note="This check applies before confidence threshold - signal is HOLD regardless of confidence",
                )
                return None
        
        # Determine signal direction when difference is sufficient
        if buy_probability > sell_probability:
            return "buy"
        else:
            return "sell"

    def _calculate_amount(
        self,
        current_price: float,
        confidence: float,
        prediction_result: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate order amount based on confidence.

        For regression models, can use predicted_return for more sophisticated position sizing.

        Args:
            current_price: Current market price
            confidence: Signal confidence score
            prediction_result: Optional prediction result (for regression models)

        Returns:
            Order amount in quote currency
        """
        from ..config.settings import settings
        
        # Base amount
        base_amount = (self.min_amount + self.max_amount) / 2

        # For regression models, can use predicted_return for position sizing
        # This allows larger positions for stronger predicted movements
        if prediction_result is not None:
            prediction = prediction_result.get("prediction")
            if isinstance(prediction, float):
                # Regression model: use predicted_return for position sizing
                predicted_return = float(prediction)
                max_expected_return = settings.model_regression_max_expected_return
                
                # Calculate confidence multiplier based on predicted return magnitude
                # Larger predicted returns get larger position sizes (up to max)
                return_magnitude = abs(predicted_return)
                return_based_multiplier = min(1.0, return_magnitude / max_expected_return)
                
                # Combine return-based multiplier with confidence
                # confidence_multiplier ranges from 0.5 to 1.0
                # return_based_multiplier also ranges from 0 to 1.0
                # Use average or max of both for more aggressive sizing based on predicted return
                confidence_multiplier = 0.5 + (max(confidence, return_based_multiplier) * 0.5)
                
                logger.debug(
                    "Calculating amount for regression model",
                    predicted_return=predicted_return,
                    predicted_return_pct=predicted_return * 100,
                    return_magnitude=return_magnitude,
                    return_based_multiplier=return_based_multiplier,
                    confidence=confidence,
                    confidence_multiplier=confidence_multiplier,
                )
            else:
                # Classification model: use confidence only
                confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
        else:
            # Fallback: use confidence only
            confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0

        amount = base_amount * confidence_multiplier

        # Clamp to valid range
        amount = max(self.min_amount, min(self.max_amount, amount))

        return round(amount, 2)

    async def _get_feature_vector(self, asset: str, trace_id: Optional[str] = None) -> Optional[FeatureVector]:
        """
        Get feature vector from Feature Service (via cache or REST API with fallback).

        Always checks local cache first, regardless of FEATURE_SERVICE_USE_QUEUE setting.
        Cache can be populated either from queue (if enabled) or from previous REST API calls.

        Args:
            asset: Trading pair symbol
            trace_id: Optional trace ID for request flow tracking

        Returns:
            FeatureVector or None if unavailable
        """
        # Always try cache first (cache can be populated from queue or previous REST API calls)
        cached_feature = await feature_cache.get(asset, max_age_seconds=settings.feature_service_feature_cache_ttl_seconds)
        if cached_feature:
            logger.debug("Using cached feature vector", asset=asset, trace_id=trace_id)
            return cached_feature

        # Cache miss - fallback to REST API
        logger.debug("Cache miss, fetching from REST API", asset=asset, trace_id=trace_id)
        feature_vector = await feature_service_client.get_latest_features(asset, trace_id=trace_id)
        
        if feature_vector:
            # Always cache the result for future use (regardless of queue setting)
            # This allows cache to work even if queue is disabled
            await feature_cache.set(asset, feature_vector)
            logger.debug("Retrieved feature vector from REST API and cached", asset=asset, trace_id=trace_id)
        
        return feature_vector

    def _create_market_data_snapshot_from_features(self, feature_vector: FeatureVector) -> MarketDataSnapshot:
        """
        Create MarketDataSnapshot from FeatureVector for signal metadata.

        Args:
            feature_vector: FeatureVector from Feature Service

        Returns:
            MarketDataSnapshot created from feature vector
        """
        features = feature_vector.features
        
        # Extract price (mid_price is the standard name in Feature Service)
        price = features.get("mid_price", features.get("price", 0.0))
        
        # Extract spread
        spread = features.get("spread_abs", features.get("spread", 0.0))
        
        # Extract volume
        volume_24h = features.get("volume_1m", features.get("volume_24h", 0.0))
        
        # Extract volatility
        volatility = features.get("volatility_1m", features.get("volatility", 0.0))
        
        # Extract orderbook depth if available
        orderbook_depth = None
        if "depth_bid_top5" in features or "depth_ask_top5" in features:
            orderbook_depth = {
                "bid_depth": features.get("depth_bid_top5", 0.0),
                "ask_depth": features.get("depth_ask_top5", 0.0),
            }
        
        # Extract technical indicators if available
        technical_indicators = None
        # Note: Feature Service may not include all technical indicators in feature vector
        # This is a simplified extraction - can be enhanced based on actual feature names
        
        return MarketDataSnapshot(
            price=float(price),
            spread=float(spread),
            volume_24h=float(volume_24h),
            volatility=float(volatility),
            orderbook_depth=orderbook_depth,
            technical_indicators=technical_indicators,
        )

    async def _save_prediction_target(
        self,
        signal: TradingSignal,
        prediction_result: Dict[str, Any],
        feature_vector: FeatureVector,
        model_version: Optional[str],
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Save prediction target to database.

        Args:
            signal: Generated trading signal
            prediction_result: Model prediction result
            feature_vector: Feature vector used for prediction
            model_version: Model version used
            trace_id: Optional trace ID
        """
        try:
            # Get target registry version from model's training_config if available
            # This ensures we use the same target config that was used during training
            target_registry_version = None
            
            # Try to get from model metadata if model_version is available
            if model_version:
                from ..database.repositories.model_version_repo import ModelVersionRepository
                model_version_repo = ModelVersionRepository()
                model_record = await model_version_repo.get_by_version(model_version)
                if model_record and model_record.get("training_config"):
                    training_config = model_record["training_config"]
                    if isinstance(training_config, str):
                        training_config = json.loads(training_config)
                    target_registry_version = training_config.get("target_registry_version")
                    logger.debug(
                        "Using target_registry_version from model training_config",
                        model_version=model_version,
                        target_registry_version=target_registry_version,
                    )
            
            # Fallback to active target registry version if not in model config
            if not target_registry_version:
                target_registry_version = await target_registry_client.get_target_registry_version()
                logger.debug(
                    "Using active target_registry_version (not found in model training_config)",
                    target_registry_version=target_registry_version,
                )
            
            # Get target config (with retry and caching)
            target_config = None
            if target_registry_version:
                target_config = await target_registry_client.get_target_config(target_registry_version)
            
            # Fallback to settings if target_config is unavailable (e.g., Feature Service timeout)
            # This ensures prediction_target is saved even during temporary service issues
            if not target_config:
                logger.warning(
                    "Target registry config not available, using fallback from settings",
                    signal_id=signal.signal_id,
                    target_registry_version=target_registry_version,
                    trace_id=trace_id,
                )
                # Use fallback config from settings
                target_config = {
                    "type": "classification",  # Default type
                    "horizon": settings.model_prediction_horizon_seconds,
                    "computation": {
                        "preset": "returns",
                        "threshold": settings.model_classification_threshold,
                    },
                }
                # Use default target_registry_version if not available
                if not target_registry_version:
                    target_registry_version = "fallback"
                    logger.info(
                        "Using fallback target registry version",
                        signal_id=signal.signal_id,
                        trace_id=trace_id,
                    )

            # Extract horizon from target_config (should always be available now)
            horizon_seconds = target_config.get("horizon", 0)
            if horizon_seconds <= 0:
                # Fallback to settings if horizon is invalid
                horizon_seconds = settings.model_prediction_horizon_seconds
                logger.warning(
                    "Invalid horizon in target config, using fallback from settings",
                    signal_id=signal.signal_id,
                    original_horizon=target_config.get("horizon"),
                    fallback_horizon=horizon_seconds,
                    trace_id=trace_id,
                )
                target_config["horizon"] = horizon_seconds

            # Calculate timestamps
            prediction_timestamp = signal.timestamp
            target_timestamp = prediction_timestamp + timedelta(seconds=horizon_seconds)

            # Format predicted values based on target type
            predicted_values = self._format_predicted_values(
                prediction_result=prediction_result,
                target_config=target_config,
            )

            # Save to database
            # Note: This is now called AFTER signal is persisted in signal_publisher.publish()
            # so we don't need retry logic anymore - signal is guaranteed to exist in DB
            prediction_target_repo = PredictionTargetRepository()
            result = await prediction_target_repo.create(
                signal_id=signal.signal_id,
                prediction_timestamp=prediction_timestamp,
                target_timestamp=target_timestamp,
                model_version=model_version or "unknown",
                feature_registry_version=feature_vector.feature_registry_version,
                target_registry_version=target_registry_version,
                target_config=target_config,
                predicted_values=predicted_values,
            )
            
            logger.info(
                "Prediction target saved",
                signal_id=signal.signal_id,
                target_timestamp=target_timestamp.isoformat(),
                horizon_seconds=horizon_seconds,
                trace_id=trace_id,
            )
            return result

        except Exception as e:
            logger.error(
                "Failed to save prediction target",
                signal_id=signal.signal_id,
                error=str(e),
                exc_info=True,
                trace_id=trace_id,
            )
            raise

    def _format_predicted_values(
        self,
        prediction_result: Dict[str, Any],
        target_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Format predicted values based on target type and preset.

        Args:
            prediction_result: Model prediction result
            target_config: Target configuration

        Returns:
            Formatted predicted values dict
        """
        target_type = target_config.get("type", "classification")
        computation = target_config.get("computation", {})
        preset = computation.get("preset", "returns") if computation else "returns"
        confidence = prediction_result.get("confidence", 0.0)

        if target_type == "regression":
            # Regression: returns preset
            prediction = prediction_result.get("prediction", 0.0)
            return {
                "value": float(prediction),
                "confidence": float(confidence),
            }
        elif target_type == "classification":
            if preset == "next_candle_direction":
                # Classification: next_candle_direction preset
                buy_probability = prediction_result.get("buy_probability", 0.0)
                sell_probability = prediction_result.get("sell_probability", 0.0)
                direction = "green" if buy_probability > sell_probability else "red"
                return {
                    "direction": direction,
                    "confidence": float(confidence),
                }
            else:
                # Classification: default
                prediction = prediction_result.get("prediction")
                buy_probability = prediction_result.get("buy_probability", 0.0)
                sell_probability = prediction_result.get("sell_probability", 0.0)
                hold_probability = 1.0 - buy_probability - sell_probability
                
                # Map prediction to class (-1, 0, 1)
                if isinstance(prediction, (int, float)):
                    predicted_class = int(prediction)
                elif buy_probability > sell_probability:
                    predicted_class = 1
                elif sell_probability > buy_probability:
                    predicted_class = -1
                else:
                    predicted_class = 0

                return {
                    "class": predicted_class,
                    "probabilities": {
                        "-1": float(sell_probability),
                        "0": float(hold_probability),
                        "1": float(buy_probability),
                    },
                    "confidence": float(confidence),
                }
        elif target_type == "risk_adjusted":
            # Risk Adjusted: sharpe_ratio preset
            prediction = prediction_result.get("prediction", 0.0)
            return {
                "sharpe": float(prediction),
                "confidence": float(confidence),
            }
        else:
            # Fallback: generic format
            return {
                "value": float(prediction_result.get("prediction", 0.0)),
                "confidence": float(confidence),
            }

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

    async def _get_effective_confidence_threshold(
        self,
        active_model: Optional[Dict[str, Any]],
        asset: str,
        strategy_id: str,
        trace_id: Optional[str] = None,
    ) -> float:
        """
        Get effective confidence threshold from top-k analysis or fallback to static threshold.
        
        Tries to get top-k confidence threshold from model quality metrics.
        Falls back to static min_confidence_threshold if top-k threshold is not available.
        
        Args:
            active_model: Active model version record from database
            asset: Trading pair symbol
            strategy_id: Trading strategy identifier
            trace_id: Trace ID for request flow tracking
            
        Returns:
            Effective confidence threshold to use (0-1)
        """
        # If no active model, use static threshold
        if not active_model or not active_model.get("id"):
            logger.debug(
                "No active model available, using static threshold",
                asset=asset,
                strategy_id=strategy_id,
                static_threshold=self.min_confidence_threshold,
                trace_id=trace_id,
            )
            return self.min_confidence_threshold
        
        # Get top-k percentage: first try from model's training_config (optimal for this model),
        # then fallback to settings (global default)
        top_k_percentage = None
        
        # Check if model has optimal_top_k_percentage in training_config
        if active_model.get("training_config"):
            training_config = active_model["training_config"]
            if isinstance(training_config, str):
                try:
                    import json
                    training_config = json.loads(training_config)
                except (json.JSONDecodeError, TypeError):
                    training_config = None
            
            if training_config and isinstance(training_config, dict):
                optimal_k = training_config.get("optimal_top_k_percentage")
                if optimal_k is not None and isinstance(optimal_k, (int, float)):
                    top_k_percentage = int(optimal_k)
                    logger.debug(
                        "Using optimal top-k percentage from model training_config",
                        asset=asset,
                        strategy_id=strategy_id,
                        optimal_top_k_percentage=top_k_percentage,
                        trace_id=trace_id,
                    )
        
        # Fallback to settings if not found in training_config
        if top_k_percentage is None:
            top_k_percentage = getattr(settings, "model_signal_top_k_percentage", 10)
            logger.debug(
                "Using top-k percentage from settings (not found in model training_config)",
                asset=asset,
                strategy_id=strategy_id,
                top_k_percentage=top_k_percentage,
                trace_id=trace_id,
            )
        
        # Try to get top-k confidence threshold from quality metrics
        try:
            model_version_id = UUID(active_model["id"]) if isinstance(active_model["id"], str) else active_model["id"]
            metric_name = f"top_k_{top_k_percentage}_confidence_threshold"
            
            metrics = await self.quality_metrics_repo.get_by_model_version(
                model_version_id=model_version_id,
                metric_name=metric_name,
                dataset_split="test",
            )
            
            if metrics and len(metrics) > 0:
                # Get the latest metric (first in list, sorted by evaluated_at DESC)
                threshold_value = metrics[0].get("metric_value")
                if threshold_value is not None and isinstance(threshold_value, (int, float)):
                    threshold = float(threshold_value)
                    if 0.0 <= threshold <= 1.0:
                        logger.info(
                            "Using top-k confidence threshold from model quality metrics",
                            asset=asset,
                            strategy_id=strategy_id,
                            model_version_id=str(model_version_id),
                            top_k_percentage=top_k_percentage,
                            threshold=threshold,
                            trace_id=trace_id,
                        )
                        return threshold
                    else:
                        logger.warning(
                            "Top-k threshold value out of range, using static threshold",
                            asset=asset,
                            strategy_id=strategy_id,
                            threshold_value=threshold_value,
                            static_threshold=self.min_confidence_threshold,
                            trace_id=trace_id,
                        )
                else:
                    logger.debug(
                        "Top-k threshold value is None or invalid type, using static threshold",
                        asset=asset,
                        strategy_id=strategy_id,
                        threshold_value=threshold_value,
                        static_threshold=self.min_confidence_threshold,
                        trace_id=trace_id,
                    )
            else:
                logger.debug(
                    "Top-k confidence threshold not found in quality metrics, using static threshold",
                    asset=asset,
                    strategy_id=strategy_id,
                    model_version_id=str(model_version_id),
                    metric_name=metric_name,
                    static_threshold=self.min_confidence_threshold,
                    trace_id=trace_id,
                )
        except Exception as e:
            logger.warning(
                "Failed to get top-k confidence threshold, using static threshold",
                asset=asset,
                strategy_id=strategy_id,
                error=str(e),
                static_threshold=self.min_confidence_threshold,
                trace_id=trace_id,
                exc_info=True,
            )
        
        # Fallback to static threshold
        return self.min_confidence_threshold

    async def _create_rejected_signal(
        self,
        asset: str,
        strategy_id: str,
        model_version: Optional[str],
        confidence: float,
        effective_threshold: float,
        rejection_reason: str,
        prediction_result: Dict[str, Any],
        feature_vector: FeatureVector,
        active_model: Optional[Dict[str, Any]],
        raw_prediction_metadata: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> TradingSignal:
        """
        Create a rejected signal (low confidence or HOLD prediction).
        
        Args:
            asset: Trading pair symbol
            strategy_id: Trading strategy identifier
            model_version: Model version used
            confidence: Actual confidence score
            effective_threshold: Threshold that was not exceeded
            rejection_reason: Reason for rejection
            prediction_result: Full prediction result from model inference
            feature_vector: Feature vector used for prediction
            active_model: Active model version record
            raw_prediction_metadata: Raw prediction metadata to include
            trace_id: Trace ID for request flow tracking
            
        Returns:
            TradingSignal with is_rejected=True
        """
        # Determine signal type from prediction (even if rejected)
        signal_type = self._determine_signal_type(prediction_result)
        # If signal_type is None (HOLD), use 'buy' as default for rejected signal structure
        if signal_type is None:
            signal_type = "buy"  # Default, won't be used for trading
        
        # Extract price from feature vector
        current_price = float(feature_vector.features.get("mid_price", feature_vector.features.get("price", 0.0)))
        if current_price <= 0:
            logger.warning("Invalid price in feature vector for rejected signal", asset=asset, strategy_id=strategy_id, price=current_price)
            current_price = 0.0
        
        # Use minimum amount for rejected signals (won't be traded anyway)
        amount = self.min_amount
        
        # Create market data snapshot from feature vector
        market_data_snapshot = self._create_market_data_snapshot_from_features(feature_vector)
        
        # Get target registry info for metadata
        target_registry_version = None
        target_config = None
        prediction_horizon_seconds = None
        target_timestamp = None
        
        if active_model and active_model.get("training_config"):
            training_config = active_model["training_config"]
            if isinstance(training_config, str):
                training_config = json.loads(training_config)
            target_registry_version = training_config.get("target_registry_version")
        
        if not target_registry_version:
            active_target_registry_version = await target_registry_client.get_target_registry_version()
            if active_target_registry_version:
                target_registry_version = active_target_registry_version
        
        if target_registry_version:
            target_config = await target_registry_client.get_target_config(target_registry_version)
            if target_config:
                prediction_horizon_seconds = target_config.get("horizon", 0)
                if prediction_horizon_seconds > 0:
                    target_timestamp = datetime.utcnow() + timedelta(seconds=prediction_horizon_seconds)
        
        # Create rejected signal
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
                "reasoning": f"Rejected signal: {rejection_reason}",
                # Include full raw prediction data in metadata
                **raw_prediction_metadata,
                "model_version": model_version,
                "feature_registry_version": feature_vector.feature_registry_version,
                "target_registry_version": target_registry_version,
                "prediction_horizon_seconds": prediction_horizon_seconds,
                "target_timestamp": target_timestamp.isoformat() + "Z" if target_timestamp else None,
                "inference_timestamp": datetime.utcnow().isoformat() + "Z",
            },
            trace_id=trace_id,
            is_rejected=True,
            rejection_reason=rejection_reason,
            effective_threshold=effective_threshold,
        )
        
        # Store prediction data for potential prediction_targets saving
        signal._prediction_data = {
            'prediction_result': prediction_result,
            'feature_vector': feature_vector,
            'model_version': model_version,
            'trace_id': trace_id,
        }
        
        logger.info(
            "Created rejected signal",
            asset=asset,
            strategy_id=strategy_id,
            rejection_reason=rejection_reason,
            confidence=confidence,
            threshold=effective_threshold,
            trace_id=trace_id,
        )
        
        return signal


# Initialize intelligent signal generator with settings
def get_intelligent_generator():
    """Get intelligent signal generator instance with current settings."""
    return IntelligentSignalGenerator(
        min_confidence_threshold=settings.model_activation_threshold,  # Use activation threshold as confidence threshold
        min_amount=settings.warmup_min_amount,
        max_amount=settings.warmup_max_amount,
    )


# Global intelligent signal generator instance
intelligent_signal_generator = get_intelligent_generator()

