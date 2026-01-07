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
from ..models.features_response import FeaturesResponse
from ..models.market_data import MarketData
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
from ..services.instrument_info_client import instrument_info_client
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
            model_version_repo = ModelVersionRepository()
            if not model_version:
                # Get active model version from database (by strategy_id and symbol/asset)
                active_model = await model_version_repo.get_active_by_strategy_and_symbol(strategy_id, asset)
                model_version = active_model["version"] if active_model else None
            else:
                # If model_version is provided, load it to get training_config
                active_model = await model_version_repo.get_by_version(model_version)

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

            # Get model's training config to get feature registry version
            model_feature_registry_version = None
            model_target_registry_version = None
            training_config = None
            if active_model and active_model.get("training_config"):
                training_config = active_model["training_config"]
                if isinstance(training_config, str):
                    training_config = json.loads(training_config)
                model_feature_registry_version = training_config.get("feature_registry_version")
                model_target_registry_version = training_config.get("target_registry_version")

            # Get feature vector and market data from Feature Service with model's version
            logger.info("Getting feature vector and market data", asset=asset, strategy_id=strategy_id, feature_registry_version=model_feature_registry_version, trace_id=trace_id)
            features_response = await self._get_features_response(
                asset, 
                feature_registry_version=model_feature_registry_version,
                trace_id=trace_id
            )
            if not features_response:
                logger.warning("Features response unavailable, skipping signal generation", asset=asset, strategy_id=strategy_id, trace_id=trace_id)
                return None
            
            feature_vector = features_response.feature_vector
            market_data = features_response.market_data
            
            logger.info("Feature vector and market data retrieved", asset=asset, strategy_id=strategy_id, feature_count=len(feature_vector.features), price=market_data.price, feature_registry_version=feature_vector.feature_registry_version, trace_id=trace_id)

            # Version mismatch check is no longer needed - we always request the correct version

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

            # Get target config from model's training_config to determine task type
            # Reuse training_config if already loaded above, otherwise load it
            target_config = None
            if not training_config and active_model and active_model.get("training_config"):
                training_config = active_model["training_config"]
                if isinstance(training_config, str):
                    training_config = json.loads(training_config)
            
            if training_config:
                target_registry_version = training_config.get("target_registry_version")
                if target_registry_version:
                    target_config = await target_registry_client.get_target_config(target_registry_version)
            
            # Determine if this is a regression model early (needed for confidence threshold check)
            is_regression = prediction_result.get("probabilities") is None
            
            # Check confidence threshold (only for classification models)
            # For regression models, confidence is just normalized return magnitude, not real confidence
            confidence = prediction_result.get("confidence", 0.0)
            threshold_source = "top_k" if effective_threshold != self.min_confidence_threshold else "static"
            
            if not is_regression:
                # Only check confidence threshold for classification models
                logger.info(
                    "Checking confidence threshold",
                    asset=asset,
                    strategy_id=strategy_id,
                    confidence=confidence,
                    threshold=effective_threshold,
                    threshold_source=threshold_source,
                    trace_id=trace_id,
                )
            else:
                # For regression models, skip confidence threshold check
                logger.debug(
                    "Skipping confidence threshold check for regression model (confidence is normalized return magnitude)",
                    asset=asset,
                    strategy_id=strategy_id,
                    predicted_return=prediction_result.get("prediction"),
                    trace_id=trace_id,
                )
            
            # Determine signal type from prediction (before threshold check to have signal_type for rejected signals)
            # Pass target_config to reliably determine if model is classification or regression
            # Pass training_config to use quantile thresholds for regression models
            signal_type = self._determine_signal_type(
                prediction_result, 
                target_config=target_config,
                training_config=training_config,
            )
            
            # Prepare raw prediction data for metadata (for all signals - valid and rejected)
            # For regression models, we don't compute buy/sell probabilities - rely on predicted_return and thresholds
            # For classification models, use buy/sell probabilities from prediction_result
            buy_probability = prediction_result.get("buy_probability")
            sell_probability = prediction_result.get("sell_probability")
            
            # Only set buy/sell probabilities for classification models
            # For regression models, leave them as None
            if is_regression:
                # For regression: don't set buy/sell probabilities
                buy_probability = None
                sell_probability = None
            elif buy_probability is None and sell_probability is None:
                # Classification model but probabilities are missing - this shouldn't happen, but handle gracefully
                buy_probability = None
                sell_probability = None
            elif buy_probability is None:
                # Only buy_probability is None, set to complement of sell_probability
                buy_probability = 1.0 - (sell_probability if sell_probability is not None else 0.5)
            elif sell_probability is None:
                # Only sell_probability is None, set to complement of buy_probability
                sell_probability = 1.0 - buy_probability
            
            # For regression models, extract regression thresholds (quantile thresholds) from training_config
            regression_thresholds_info = None
            if is_regression:
                if training_config and isinstance(training_config, dict):
                    regression_thresholds = training_config.get("regression_thresholds")
                    if regression_thresholds and isinstance(regression_thresholds, dict):
                        method = regression_thresholds.get("method")
                        buy_threshold_value = regression_thresholds.get("buy_threshold_value")
                        sell_threshold_value = regression_thresholds.get("sell_threshold_value")
                        if method and buy_threshold_value is not None and sell_threshold_value is not None:
                            regression_thresholds_info = {
                                "method": method,
                                "buy_threshold": float(buy_threshold_value),
                                "sell_threshold": float(sell_threshold_value),
                            }
                            logger.info(
                                "Extracted regression thresholds from training_config",
                                asset=asset,
                                strategy_id=strategy_id,
                                method=method,
                                buy_threshold=buy_threshold_value,
                                sell_threshold=sell_threshold_value,
                                trace_id=trace_id,
                            )
                        else:
                            logger.warning(
                                "Regression thresholds in training_config missing required fields",
                                asset=asset,
                                strategy_id=strategy_id,
                                method=method,
                                buy_threshold_value=buy_threshold_value,
                                sell_threshold_value=sell_threshold_value,
                                trace_id=trace_id,
                            )
                    else:
                        logger.warning(
                            "No regression_thresholds found in training_config for regression model",
                            asset=asset,
                            strategy_id=strategy_id,
                            training_config_keys=list(training_config.keys()) if training_config else None,
                            trace_id=trace_id,
                        )
                else:
                    logger.warning(
                        "training_config not available for regression model - cannot extract regression thresholds",
                        asset=asset,
                        strategy_id=strategy_id,
                        is_regression=is_regression,
                        training_config_type=type(training_config).__name__ if training_config else None,
                        has_active_model=active_model is not None,
                        trace_id=trace_id,
                    )
            
            raw_prediction_metadata = {
                "prediction_result": {
                    "prediction": prediction_result.get("prediction"),
                    "buy_probability": buy_probability,
                    "sell_probability": sell_probability,
                    "confidence": confidence,
                    "probabilities": prediction_result.get("probabilities"),  # Raw probabilities array if available
                },
                "effective_threshold": effective_threshold,
                "threshold_source": threshold_source,
            }
            
            # Add regression thresholds info for regression models
            if regression_thresholds_info:
                raw_prediction_metadata["regression_thresholds"] = regression_thresholds_info
                logger.info(
                    "Added regression_thresholds to raw_prediction_metadata",
                    asset=asset,
                    strategy_id=strategy_id,
                    regression_thresholds=regression_thresholds_info,
                    trace_id=trace_id,
                )
            else:
                if is_regression:
                    logger.warning(
                        "regression_thresholds_info is None for regression model - will not be saved in metadata",
                        asset=asset,
                        strategy_id=strategy_id,
                        is_regression=is_regression,
                        has_training_config=training_config is not None,
                        trace_id=trace_id,
                    )
            
            # Check if confidence is below threshold (only for classification models)
            # For regression models, skip this check as confidence is not meaningful
            if not is_regression and confidence < effective_threshold:
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
                    market_data=market_data,
                    active_model=active_model,
                    raw_prediction_metadata=raw_prediction_metadata,
                    trace_id=trace_id,
                )
            
            # If model returned None (HOLD), create rejected signal
            # Determine rejection reason based on model type
            if signal_type is None:
                if is_regression:
                    # Regression model: HOLD due to predicted return within threshold range
                    logger.debug(
                        "Regression model predicted HOLD (return within threshold range) - creating rejected signal",
                        asset=asset,
                        strategy_id=strategy_id,
                        predicted_return=prediction_result.get("prediction"),
                        threshold=settings.model_regression_threshold,
                        trace_id=trace_id,
                    )
                    rejection_reason = "regression_hold_prediction"
                else:
                    # Classification model: HOLD due to hysteresis (min_probability_diff) or calibrated thresholds
                    buy_prob = prediction_result.get("buy_probability", 0.0)
                    sell_prob = prediction_result.get("sell_probability", 0.0)
                    probability_diff = abs(buy_prob - sell_prob)
                    has_calibrated_thresholds = prediction_result.get("_used_calibrated_thresholds", False)
                    
                    if has_calibrated_thresholds:
                        # Model has calibrated thresholds but none passed
                        logger.debug(
                            "Classification model predicted HOLD (calibrated thresholds not met) - creating rejected signal",
                            asset=asset,
                            strategy_id=strategy_id,
                            buy_probability=buy_prob,
                            sell_probability=sell_prob,
                            confidence=confidence,
                            threshold=effective_threshold,
                            trace_id=trace_id,
                        )
                        rejection_reason = "classification_hold_prediction_calibrated_thresholds"
                    else:
                        # HOLD due to insufficient probability difference (hysteresis)
                        logger.debug(
                            "Classification model predicted HOLD (insufficient probability difference) - creating rejected signal",
                            asset=asset,
                            strategy_id=strategy_id,
                            buy_probability=buy_prob,
                            sell_probability=sell_prob,
                            probability_diff=probability_diff,
                            min_probability_diff=settings.model_min_probability_diff,
                            confidence=confidence,
                            threshold=effective_threshold,
                            trace_id=trace_id,
                        )
                        rejection_reason = "classification_hold_prediction_hysteresis"
                
                return await self._create_rejected_signal(
                    asset=asset,
                    strategy_id=strategy_id,
                    model_version=model_version,
                    confidence=confidence,
                    effective_threshold=effective_threshold,
                    rejection_reason=rejection_reason,
                    prediction_result=prediction_result,
                    feature_vector=feature_vector,
                    market_data=market_data,
                    active_model=active_model,
                    raw_prediction_metadata=raw_prediction_metadata,
                    trace_id=trace_id,
                )
            
            # Log prediction details for debugging
            prediction = prediction_result.get("prediction")
            if isinstance(prediction, float):
                # Regression model - determine which threshold method was used
                threshold_info = {"method": "fixed", "threshold": settings.model_regression_threshold}
                if training_config and isinstance(training_config, dict):
                    regression_thresholds = training_config.get("regression_thresholds")
                    if regression_thresholds and isinstance(regression_thresholds, dict):
                        method = regression_thresholds.get("method")
                        if method == "quantile":
                            threshold_info = {
                                "method": "quantile",
                                "buy_threshold": regression_thresholds.get("buy_threshold_value"),
                                "sell_threshold": regression_thresholds.get("sell_threshold_value"),
                            }
                
                logger.info(
                    "Model prediction (regression)",
                    asset=asset,
                    strategy_id=strategy_id,
                    predicted_return=prediction,
                    predicted_return_pct=prediction * 100,
                    determined_signal_type=signal_type,
                    confidence=confidence,
                    threshold_info=threshold_info,
                    trace_id=trace_id,
                )
            else:
                # Classification model (or regression with computed probabilities)
                # Use computed values from raw_prediction_metadata if available
                log_buy_probability = buy_probability if buy_probability is not None else prediction_result.get("buy_probability", 0.0)
                log_sell_probability = sell_probability if sell_probability is not None else prediction_result.get("sell_probability", 0.0)
                logger.info(
                    "Model prediction probabilities",
                    asset=asset,
                    strategy_id=strategy_id,
                    buy_probability=log_buy_probability,
                    sell_probability=log_sell_probability,
                    determined_signal_type=signal_type,
                    confidence=confidence,
                    signal_direction_check=f"buy_prob({log_buy_probability:.4f}) {'>' if log_buy_probability > log_sell_probability else '<='} sell_prob({log_sell_probability:.4f})",
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

            # Extract price from market data
            current_price = market_data.price
            if current_price <= 0:
                logger.error(
                    "Invalid price in market data, skipping signal generation",
                    asset=asset,
                    strategy_id=strategy_id,
                    price=current_price,
                    trace_id=trace_id,
                )
                return None
            
            logger.debug(
                "Price extracted from market data",
                asset=asset,
                strategy_id=strategy_id,
                price=current_price,
                trace_id=trace_id,
            )

            # Get minimum order value from instrument_info table (with fallback to settings)
            # This ensures we use the actual exchange requirement, not a hardcoded value
            min_order_value_db = await instrument_info_client.get_min_order_value(asset)
            effective_min_amount = float(min_order_value_db) if min_order_value_db is not None else self.min_amount
            min_amount_source = "instrument_info" if min_order_value_db is not None else "settings"
            
            logger.debug(
                "Using minimum order amount",
                asset=asset,
                min_amount=effective_min_amount,
                source=min_amount_source,
                fallback_min_amount=self.min_amount,
                trace_id=trace_id,
            )

            # Get available resources (balance for BUY, position size for SELL) before calculating amount
            # This allows the model to calculate amount that respects available resources
            available_resource: Optional[float] = None
            if signal_type.lower() == "buy":
                available_resource = await balance_calculator.get_available_balance_for_buy(
                    trading_pair=asset,
                    current_price=current_price,
                )
                if available_resource is None:
                    logger.warning(
                        "Cannot get available balance for buy order, skipping signal generation",
                        asset=asset,
                        strategy_id=strategy_id,
                        trace_id=trace_id,
                    )
                    return None
                logger.info(
                    "Available balance for buy order",
                    asset=asset,
                    available_balance=available_resource,
                    min_amount=effective_min_amount,
                    min_amount_source=min_amount_source,
                    max_amount=self.max_amount,
                    trace_id=trace_id,
                )
            else:  # sell
                available_resource = await balance_calculator.get_available_position_for_sell(
                    trading_pair=asset,
                    current_price=current_price,
                )
                if available_resource is None:
                    logger.warning(
                        "Cannot get available position for sell order, skipping signal generation",
                        asset=asset,
                        strategy_id=strategy_id,
                        trace_id=trace_id,
                    )
                    return None
                logger.info(
                    "Available position for sell order",
                    asset=asset,
                    available_position_usdt=available_resource,
                    min_amount=effective_min_amount,
                    min_amount_source=min_amount_source,
                    max_amount=self.max_amount,
                    current_price=current_price,
                    trace_id=trace_id,
                )

            # Calculate order amount from model, considering available resources
            # Pass prediction_result for regression models to use predicted_return for position sizing
            # Pass available_resource to limit amount by available balance/position
            # Use effective_min_amount (from instrument_info or settings) instead of self.min_amount
            model_amount = self._calculate_amount(
                current_price, 
                confidence,
                prediction_result=prediction_result,
                available_resource=available_resource,
                min_amount=effective_min_amount,
            )
            
            # Check if amount calculation failed due to insufficient balance
            if model_amount is None:
                logger.warning(
                    "Cannot calculate signal amount: available resource is less than minimum amount",
                    asset=asset,
                    strategy_id=strategy_id,
                    signal_type=signal_type,
                    available_resource=available_resource,
                    min_amount=effective_min_amount,
                    min_amount_source=min_amount_source,
                    current_price=current_price,
                    trace_id=trace_id,
                )
                # Record skip metrics
                signal_skip_metrics.record_skip(
                    asset=asset,
                    strategy_id=strategy_id,
                    reason="insufficient_balance",
                )
                return None
            
            # Final check: verify that calculated amount is still affordable
            # This is a safety check in case of race conditions or price changes
            # Note: calculate_affordable_amount may use different balance source than get_available_balance_for_buy
            # (e.g., account-level vs coin-level), so this check is important
            adapted_amount = await balance_calculator.calculate_affordable_amount(
                trading_pair=asset,
                signal_type=signal_type,
                requested_amount=model_amount,
                current_price=current_price,
            )
            
            if adapted_amount is None:
                # Get more details about why balance is insufficient
                # For SELL signals, this is usually due to insufficient position size
                # For BUY signals, this is due to insufficient quote currency balance
                # This can happen if:
                # 1. Balance changed between get_available_balance_for_buy and calculate_affordable_amount
                # 2. Different balance sources are used (account-level vs coin-level)
                # 3. Safety margin is applied differently in the two methods
                logger.warning(
                    "Insufficient balance, skipping signal generation",
                    asset=asset,
                    strategy_id=strategy_id,
                    signal_type=signal_type,
                    original_model_amount=model_amount,
                    available_resource=available_resource,
                    current_price=current_price,
                    note="Check balance_calculator logs above for detailed position/balance information. This may indicate balance changed between checks or different balance sources were used.",
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
            # Create market data snapshot from market data for signal metadata
            market_data_snapshot = self._create_market_data_snapshot_from_market_data(market_data)

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

    def _determine_signal_type(
        self, 
        prediction_result: Dict[str, Any],
        target_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        buy_probability: Optional[float] = None,
        sell_probability: Optional[float] = None,
    ) -> Optional[str]:
        """
        Determine signal type from model prediction.

        Supports both classification and regression models:
        - Classification: Uses buy/sell probabilities with hysteresis (min_probability_diff)
        - Regression: Converts predicted return to signal using quantile thresholds (if available) or fixed threshold

        INVARIANT: For classification models without calibrated thresholds:
        - If |buy_probability - sell_probability| < min_probability_diff → HOLD (None)
        - This ensures hysteresis and prevents signals when model is uncertain
        - Confidence threshold is an additional filter, not a replacement for hysteresis

        Args:
            prediction_result: Prediction result from model inference
                - For classification: contains "buy_probability", "sell_probability"
                - For regression: contains "prediction" (float, predicted return value)
            target_config: Optional target config from model's training_config.
                If provided, uses target_config["type"] to reliably determine task type.
                If not provided, falls back to heuristic (prediction type).
            training_config: Optional training config from model version.
                Used to get quantile thresholds for regression models.

        Returns:
            'buy', 'sell', or None (for HOLD)
        """
        from ..config.settings import settings
        
        # Get prediction value for type checking
        prediction = prediction_result.get("prediction")
        
        # Determine task type: prefer target_config if available, otherwise use heuristic
        is_regression = False
        if target_config and isinstance(target_config, dict):
            target_type = target_config.get("type", "").lower()
            is_regression = target_type == "regression"
            logger.debug(
                "Determined task type from target_config",
                target_type=target_type,
                is_regression=is_regression,
            )
        else:
            # Fallback heuristic: check if prediction is a float (regression) or int (classification)
            is_regression = prediction is not None and isinstance(prediction, float)
            logger.debug(
                "Determined task type from prediction type (heuristic)",
                prediction_type=type(prediction).__name__ if prediction is not None else None,
                is_regression=is_regression,
            )
        
        if is_regression:
            # Regression model: convert predicted return to signal
            if prediction is None:
                logger.warning("Regression prediction is None, returning HOLD")
                return None
            predicted_return = float(prediction)
            
            # Check if quantile thresholds are available in training_config
            use_quantile_thresholds = False
            buy_threshold = None
            sell_threshold = None
            
            if training_config and isinstance(training_config, dict):
                regression_thresholds = training_config.get("regression_thresholds")
                if regression_thresholds and isinstance(regression_thresholds, dict):
                    method = regression_thresholds.get("method")
                    if method == "quantile":
                        buy_threshold = regression_thresholds.get("buy_threshold_value")
                        sell_threshold = regression_thresholds.get("sell_threshold_value")
                        if buy_threshold is not None and sell_threshold is not None:
                            use_quantile_thresholds = True
                            logger.debug(
                                "Using quantile thresholds for regression signal",
                                buy_threshold=buy_threshold,
                                sell_threshold=sell_threshold,
                                predicted_return=predicted_return,
                            )
            
            # Fallback to fixed threshold if quantile thresholds not available
            if not use_quantile_thresholds:
                threshold = settings.model_regression_threshold
                buy_threshold = threshold
                sell_threshold = -threshold
                logger.debug(
                    "Using fixed threshold for regression signal (quantile thresholds not available)",
                    threshold=threshold,
                    predicted_return=predicted_return,
                )
            
            # Determine signal based on thresholds
            # For regression models, we only use predicted_return and thresholds
            # We don't use buy/sell probabilities as they are not meaningful for regression
            if buy_threshold is not None and sell_threshold is not None:
                if predicted_return >= buy_threshold:
                    return "buy"
                elif predicted_return <= sell_threshold:
                    return "sell"
                else:
                    # HOLD: predicted return is within threshold range (hysteresis)
                    return None
            else:
                # Fallback: use fixed threshold
                threshold = settings.model_regression_threshold
                if predicted_return > threshold:
                    return "buy"
                elif predicted_return < -threshold:
                    return "sell"
                else:
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
        available_resource: Optional[float] = None,
        min_amount: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate order amount based on confidence and available resources.

        For regression models, can use predicted_return for more sophisticated position sizing.
        For BUY signals, available_resource is available balance in quote currency.
        For SELL signals, available_resource is available position size in quote currency.

        Args:
            current_price: Current market price
            confidence: Signal confidence score
            prediction_result: Optional prediction result (for regression models)
            available_resource: Optional available balance/position size in quote currency
            min_amount: Optional minimum order amount (uses self.min_amount if not provided)

        Returns:
            Order amount in quote currency, limited by available resources if provided.
            Returns None if available_resource is provided and is less than min_amount.
        """
        from ..config.settings import settings
        
        # Use provided min_amount or fallback to self.min_amount
        effective_min_amount = min_amount if min_amount is not None else self.min_amount
        
        # Base amount
        base_amount = (effective_min_amount + self.max_amount) / 2

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
                    available_resource=available_resource,
                )
            else:
                # Classification model: use confidence only
                confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
        else:
            # Fallback: use confidence only
            confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0

        amount = base_amount * confidence_multiplier
        calculated_amount_before_limit = amount

        # Limit by available resources first (if provided)
        # This ensures we don't calculate amount that exceeds available balance/position
        if available_resource is not None:
            # Check if available resource is sufficient before limiting
            if available_resource < effective_min_amount:
                logger.debug(
                    "Available resource is less than minimum amount, cannot generate signal",
                    calculated_amount_before_limit=calculated_amount_before_limit,
                    available_resource=available_resource,
                    min_amount=effective_min_amount,
                )
                return None
            
            amount = min(amount, available_resource)
            logger.debug(
                "Amount limited by available resources",
                calculated_amount_before_limit=calculated_amount_before_limit,
                available_resource=available_resource,
                amount_after_limit=amount,
            )
        
        # Clamp to valid range (min_amount, max_amount)
        # If available_resource was provided and is >= min_amount, we've already limited amount by it
        # But we still need to ensure amount meets minimum requirement (min_amount)
        # If the calculated amount is less than min_amount, clamp it to min_amount
        # The final balance check will reject the signal if min_amount exceeds available balance
        amount = max(effective_min_amount, min(self.max_amount, amount))

        return round(amount, 2)

    async def _get_features_response(
        self, 
        asset: str, 
        feature_registry_version: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> Optional[FeaturesResponse]:
        """
        Get features response (feature vector + market data) from Feature Service.

        Args:
            asset: Trading pair symbol
            feature_registry_version: Optional Feature Registry version
            trace_id: Optional trace ID

        Returns:
            FeaturesResponse or None if unavailable
        """
        # Fetch from REST API (market_data is always needed, so no caching for now)
        logger.debug("Fetching features response from Feature Service API", asset=asset, trace_id=trace_id)
        features_response = await feature_service_client.get_latest_features(
            asset, 
            feature_registry_version=feature_registry_version,
            trace_id=trace_id
        )
        
        return features_response

    def _create_market_data_snapshot_from_market_data(self, market_data: MarketData) -> MarketDataSnapshot:
        """
        Create MarketDataSnapshot from MarketData for signal metadata.

        Args:
            market_data: MarketData from Feature Service

        Returns:
            MarketDataSnapshot created from market data
        """
        return MarketDataSnapshot(
            price=market_data.price,
            spread=market_data.spread,
            volume_24h=market_data.volume_24h or 0.0,
            volatility=market_data.volatility or 0.0,
            orderbook_depth=market_data.orderbook_depth,
            technical_indicators=None,  # Not available in MarketData
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
                if target_config:
                    logger.debug(
                        "Using target_config from target_registry",
                        signal_id=signal.signal_id,
                        target_registry_version=target_registry_version,
                        target_type=target_config.get("type"),
                        preset=target_config.get("computation", {}).get("preset") if isinstance(target_config.get("computation"), dict) else None,
                        trace_id=trace_id,
                    )
            
            # Fallback to settings if target_config is unavailable (e.g., Feature Service timeout)
            # This ensures prediction_target is saved even during temporary service issues
            # Try to determine target type from model's training_config if available
            if not target_config:
                fallback_target_type = None
                if model_version:
                    from ..database.repositories.model_version_repo import ModelVersionRepository
                    model_version_repo = ModelVersionRepository()
                    model_record = await model_version_repo.get_by_version(model_version)
                    if model_record and model_record.get("training_config"):
                        training_config = model_record["training_config"]
                        if isinstance(training_config, str):
                            training_config = json.loads(training_config)
                        task_type = training_config.get("task_type")
                        # Map task_type to target_type: "regression" -> "regression", "classification" -> "classification"
                        if task_type in ["regression", "classification"]:
                            fallback_target_type = task_type
                            logger.debug(
                                "Using target type from model training_config for fallback",
                                model_version=model_version,
                                task_type=task_type,
                                signal_id=signal.signal_id,
                            )
                
                # If still not determined, default to regression (most common case)
                if not fallback_target_type:
                    fallback_target_type = "regression"
                    logger.debug(
                        "Using default regression target type for fallback",
                        signal_id=signal.signal_id,
                    )
                
                logger.warning(
                    "Target registry config not available, using fallback from model training_config or default",
                    signal_id=signal.signal_id,
                    target_registry_version=target_registry_version,
                    fallback_target_type=fallback_target_type,
                    trace_id=trace_id,
                )
                # Use fallback config - preset "returns" is always regression
                # But we use the determined target_type for consistency
                target_config = {
                    "type": fallback_target_type,
                    "horizon": settings.model_prediction_horizon_seconds,
                    "computation": {
                        "preset": "returns",
                        "threshold": settings.model_classification_threshold if fallback_target_type == "classification" else None,
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
        market_data: MarketData,
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
        # Get target config for signal type determination
        target_config = None
        training_config = None
        if active_model and active_model.get("training_config"):
            training_config = active_model["training_config"]
            if isinstance(training_config, str):
                training_config = json.loads(training_config)
            target_registry_version = training_config.get("target_registry_version")
            if target_registry_version:
                target_config = await target_registry_client.get_target_config(target_registry_version)
        
        # Determine signal type from prediction (even if rejected)
        signal_type = self._determine_signal_type(
            prediction_result, 
            target_config=target_config,
            training_config=training_config,
        )
        # If signal_type is None (HOLD), try to infer from prediction value
        # This ensures rejected signals reflect model's actual prediction direction
        if signal_type is None:
            prediction = prediction_result.get("prediction")
            if prediction == -1:
                signal_type = "sell"  # Model predicted DOWN/SELL
            elif prediction == 1:
                signal_type = "buy"  # Model predicted UP/BUY
            else:
                # Fallback: use buy_probability vs sell_probability comparison
                buy_prob = prediction_result.get("buy_probability", 0.0)
                sell_prob = prediction_result.get("sell_probability", 0.0)
                if sell_prob > buy_prob:
                    signal_type = "sell"
                else:
                    signal_type = "buy"  # Default fallback
        
        # Extract price from feature vector
        # For rejected signals, we don't have market_data, so skip price extraction
        # This is only used for logging, so 0.0 is acceptable
        current_price = 0.0
        
        # Use minimum amount for rejected signals (won't be traded anyway)
        amount = self.min_amount
        
        # Create market data snapshot from market data
        market_data_snapshot = self._create_market_data_snapshot_from_market_data(market_data)
        
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

