"""
Model inference service.

Prepares features from order/position state and market data,
runs model prediction, and generates confidence scores.
MUST capture market data snapshot at inference time for inclusion in generated signals.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from ..models.position_state import OrderPositionState
from ..models.signal import MarketDataSnapshot
from ..consumers.market_data_consumer import market_data_cache
from ..config.logging import get_logger
from ..config.exceptions import ModelInferenceError

logger = get_logger(__name__)


class ModelInference:
    """Performs model inference for trading signal generation."""

    def __init__(self):
        """Initialize model inference service."""
        pass

    def prepare_features(
        self,
        asset: str,
        order_position_state: Optional[OrderPositionState] = None,
        market_data_snapshot: Optional[MarketDataSnapshot] = None,
    ) -> pd.DataFrame:
        """
        Prepare features for model inference from current state.

        Args:
            asset: Trading pair symbol (e.g., 'BTCUSDT')
            order_position_state: Current order and position state
            market_data_snapshot: Market data snapshot at inference time

        Returns:
            DataFrame with features (single row)

        Raises:
            ModelInferenceError: If required data is missing
        """
        # Get market data snapshot if not provided
        if not market_data_snapshot:
            market_data = market_data_cache.get_market_data(asset)
            if not market_data:
                raise ModelInferenceError(f"Market data unavailable for asset {asset}")

            market_data_snapshot = MarketDataSnapshot(
                price=market_data["price"],
                spread=market_data["spread"],
                volume_24h=market_data["volume_24h"],
                volatility=market_data["volatility"],
                orderbook_depth=market_data.get("orderbook_depth"),
                technical_indicators=market_data.get("technical_indicators"),
            )

        # Engineer features
        features = self._engineer_inference_features(asset, order_position_state, market_data_snapshot)

        # Convert to DataFrame (single row)
        features_df = pd.DataFrame([features])
        logger.debug("Prepared features for inference", asset=asset, feature_count=len(features_df.columns))
        return features_df

    def _engineer_inference_features(
        self,
        asset: str,
        order_position_state: Optional[OrderPositionState],
        market_snapshot: MarketDataSnapshot,
    ) -> Dict[str, Any]:
        """
        Engineer features for inference from current state.

        Args:
            asset: Trading pair symbol
            order_position_state: Current order and position state
            market_snapshot: Market data snapshot at inference time

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Price features (from market snapshot at inference time)
        features["price"] = market_snapshot.price
        features["price_log"] = np.log(market_snapshot.price) if market_snapshot.price > 0 else 0.0

        # Spread features
        features["spread"] = market_snapshot.spread
        features["spread_percent"] = (
            (market_snapshot.spread / market_snapshot.price * 100) if market_snapshot.price > 0 else 0.0
        )

        # Volume features
        features["volume_24h"] = market_snapshot.volume_24h
        features["volume_24h_log"] = np.log(market_snapshot.volume_24h + 1)  # +1 to avoid log(0)

        # Volatility features
        features["volatility"] = market_snapshot.volatility
        features["volatility_squared"] = market_snapshot.volatility**2

        # Order book depth features (if available)
        if market_snapshot.orderbook_depth:
            features["bid_depth"] = market_snapshot.orderbook_depth.get("bid_depth", 0.0)
            features["ask_depth"] = market_snapshot.orderbook_depth.get("ask_depth", 0.0)
            features["depth_imbalance"] = (
                (features["bid_depth"] - features["ask_depth"]) / (features["bid_depth"] + features["ask_depth"] + 1e-10)
            )
        else:
            features["bid_depth"] = 0.0
            features["ask_depth"] = 0.0
            features["depth_imbalance"] = 0.0

        # Technical indicators (if available)
        if market_snapshot.technical_indicators:
            indicators = market_snapshot.technical_indicators
            features["rsi"] = indicators.get("rsi", 50.0)  # Default to neutral
            features["macd"] = indicators.get("macd", 0.0)
            features["macd_signal"] = indicators.get("macd_signal", 0.0)
            features["macd_histogram"] = indicators.get("macd_histogram", 0.0)
            features["moving_average_20"] = indicators.get("moving_average_20", market_snapshot.price)
            features["moving_average_50"] = indicators.get("moving_average_50", market_snapshot.price)
            features["bollinger_upper"] = indicators.get("bollinger_upper", market_snapshot.price)
            features["bollinger_lower"] = indicators.get("bollinger_lower", market_snapshot.price)
            features["bollinger_width"] = (
                (features["bollinger_upper"] - features["bollinger_lower"]) / market_snapshot.price
                if market_snapshot.price > 0
                else 0.0
            )
        else:
            # Default values when technical indicators not available
            features["rsi"] = 50.0
            features["macd"] = 0.0
            features["macd_signal"] = 0.0
            features["macd_histogram"] = 0.0
            features["moving_average_20"] = market_snapshot.price
            features["moving_average_50"] = market_snapshot.price
            features["bollinger_upper"] = market_snapshot.price
            features["bollinger_lower"] = market_snapshot.price
            features["bollinger_width"] = 0.0

        # Position features (from order/position state)
        if order_position_state:
            position = order_position_state.get_position(asset)
            if position:
                features["position_size"] = float(position.size)
                features["position_size_abs"] = abs(float(position.size))
                features["unrealized_pnl"] = float(position.unrealized_pnl)
                features["realized_pnl"] = float(position.realized_pnl)
                features["has_position"] = 1 if position.size != 0 else 0
                if position.average_entry_price:
                    features["entry_price"] = float(position.average_entry_price)
                    features["price_vs_entry"] = (
                        (market_snapshot.price - float(position.average_entry_price)) / float(position.average_entry_price) * 100
                        if position.average_entry_price > 0
                        else 0.0
                    )
                else:
                    features["entry_price"] = market_snapshot.price
                    features["price_vs_entry"] = 0.0
            else:
                # No position
                features["position_size"] = 0.0
                features["position_size_abs"] = 0.0
                features["unrealized_pnl"] = 0.0
                features["realized_pnl"] = 0.0
                features["has_position"] = 0
                features["entry_price"] = market_snapshot.price
                features["price_vs_entry"] = 0.0

            # Open orders features (must match feature_engineer logic for consistency)
            asset_orders = [
                order
                for order in order_position_state.orders
                if order.asset == asset and order.status in ("pending", "partially_filled")
            ]
            features["open_orders_count"] = len(asset_orders)
            features["pending_buy_orders"] = len([o for o in asset_orders if o.side.upper() == "BUY"])
            features["pending_sell_orders"] = len([o for o in asset_orders if o.side.upper() == "SELL"])

            # Total exposure
            total_exposure = order_position_state.get_total_exposure(asset)
            features["total_exposure"] = float(total_exposure)
            features["total_exposure_abs"] = abs(float(total_exposure))
        else:
            # No order/position state available
            features["position_size"] = 0.0
            features["position_size_abs"] = 0.0
            features["unrealized_pnl"] = 0.0
            features["realized_pnl"] = 0.0
            features["has_position"] = 0
            features["entry_price"] = market_snapshot.price
            features["price_vs_entry"] = 0.0
            features["open_orders_count"] = 0
            features["pending_buy_orders"] = 0
            features["pending_sell_orders"] = 0
            features["total_exposure"] = 0.0
            features["total_exposure_abs"] = 0.0

        # Asset features (categorical - hash encoding)
        features["asset_hash"] = hash(asset) % 1000

        # Strategy features (categorical - will be set by caller if available)
        # Default to 0 if not provided
        features["strategy_hash"] = 0

        # Execution features (not available at inference time, set to defaults)
        features["execution_price"] = market_snapshot.price  # Use current price as proxy
        features["execution_quantity"] = 0.0
        features["execution_fees"] = 0.0
        features["execution_value"] = 0.0
        features["slippage"] = 0.0
        features["slippage_percent"] = 0.0
        features["price_change"] = 0.0
        features["price_change_percent"] = 0.0
        features["execution_delay_seconds"] = 0.0
        features["execution_delay_minutes"] = 0.0

        # Side features (not known at inference time, set to defaults)
        features["side_buy"] = 0
        features["side_sell"] = 0

        # Market context features (from inference time market snapshot)
        features["execution_spread"] = market_snapshot.spread
        features["execution_volume_24h"] = market_snapshot.volume_24h
        features["execution_volatility"] = market_snapshot.volatility

        # Market change features (no change at inference time)
        features["spread_change"] = 0.0
        features["volume_change"] = 0.0
        features["volatility_change"] = 0.0

        return features

    def predict(
        self,
        model: Any,
        features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run model prediction on features.

        Args:
            model: Trained model object (scikit-learn or XGBoost)
            features: DataFrame with features (single row)

        Returns:
            Dictionary with prediction results:
            - prediction: Predicted class or value
            - confidence: Confidence score (0-1)
            - probabilities: Class probabilities (for classification)
        """
        try:
            # Handle missing values
            features = features.fillna(features.mean())

            # Check if model has predict_proba (classification)
            if hasattr(model, "predict_proba"):
                # Classification model
                probabilities = model.predict_proba(features)[0]
                prediction = model.predict(features)[0]

                # Calculate confidence as max probability
                confidence = float(np.max(probabilities))

                # For binary classification, map probabilities to buy/sell
                if len(probabilities) == 2:
                    # Assume: [class_0_prob, class_1_prob]
                    # class_0 = sell, class_1 = buy (or vice versa depending on training)
                    # For now, use class_1 probability as buy confidence
                    buy_probability = float(probabilities[1])
                    sell_probability = float(probabilities[0])
                else:
                    # Multi-class: use max probability
                    buy_probability = float(probabilities[1]) if len(probabilities) > 1 else 0.0
                    sell_probability = float(probabilities[0]) if len(probabilities) > 0 else 0.0

                result = {
                    "prediction": int(prediction),
                    "confidence": confidence,
                    "probabilities": probabilities.tolist() if hasattr(probabilities, "tolist") else list(probabilities),
                    "buy_probability": buy_probability,
                    "sell_probability": sell_probability,
                }
            else:
                # Regression model
                prediction = model.predict(features)[0]
                # For regression, confidence is based on prediction magnitude
                # Normalize to 0-1 range (assuming predictions are in reasonable range)
                confidence = min(1.0, max(0.0, abs(float(prediction)) / 100.0))  # Adjust divisor as needed

                result = {
                    "prediction": float(prediction),
                    "confidence": confidence,
                    "probabilities": None,
                }

            logger.debug("Model prediction completed", prediction=result.get("prediction"), confidence=result.get("confidence"))
            return result

        except Exception as e:
            logger.error("Model prediction failed", error=str(e), exc_info=True)
            raise ModelInferenceError(f"Model prediction failed: {e}") from e

    def get_market_data_snapshot(self, asset: str) -> Optional[MarketDataSnapshot]:
        """
        Get current market data snapshot for an asset.

        Args:
            asset: Trading pair symbol

        Returns:
            MarketDataSnapshot or None if data unavailable
        """
        market_data = market_data_cache.get_market_data(asset)
        if not market_data:
            logger.warning("Market data unavailable", asset=asset)
            return None

        return MarketDataSnapshot(
            price=market_data["price"],
            spread=market_data["spread"],
            volume_24h=market_data["volume_24h"],
            volatility=market_data["volatility"],
            orderbook_depth=market_data.get("orderbook_depth"),
            technical_indicators=market_data.get("technical_indicators"),
        )


# Global model inference instance
model_inference = ModelInference()

