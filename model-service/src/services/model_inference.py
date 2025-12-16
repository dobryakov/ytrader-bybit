"""
Model inference service.

Prepares features from Feature Service feature vectors and order/position state,
runs model prediction, and generates confidence scores.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from ..models.position_state import OrderPositionState
from ..models.feature_vector import FeatureVector
from ..config.logging import get_logger
from ..config.exceptions import ModelInferenceError
from ..config.settings import settings

logger = get_logger(__name__)


class ModelInference:
    """Performs model inference for trading signal generation."""

    def __init__(self):
        """Initialize model inference service."""
        pass

    def prepare_features(
        self,
        feature_vector: FeatureVector,
        order_position_state: Optional[OrderPositionState] = None,
        asset: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prepare features for model inference from Feature Service feature vector.

        Receives ready FeatureVector from Feature Service, extracts features dict,
        adds position features if available, and performs feature alignment with model's expected features.

        Args:
            feature_vector: FeatureVector from Feature Service
            order_position_state: Optional current order and position state (for position features)
            asset: Optional asset symbol (defaults to feature_vector.symbol)

        Returns:
            DataFrame with features (single row)

        Raises:
            ModelInferenceError: If required data is missing
        """
        asset = asset or feature_vector.symbol

        # Start with features from Feature Service
        features = feature_vector.features.copy()

        # Add position features if order_position_state is available
        # These features are computed locally as they depend on current state
        if order_position_state:
            position_features = self._add_position_features(asset, order_position_state, features)
            features.update(position_features)

        # Convert to DataFrame (single row)
        features_df = pd.DataFrame([features])
        
        logger.debug(
            "Prepared features for inference",
            asset=asset,
            feature_count=len(features_df.columns),
            feature_registry_version=feature_vector.feature_registry_version,
            trace_id=feature_vector.trace_id,
        )
        return features_df

    def _add_position_features(
        self,
        asset: str,
        order_position_state: OrderPositionState,
        existing_features: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Add position and order features to existing features from Feature Service.

        These features are computed locally as they depend on current order/position state
        which is not included in Feature Service feature vectors.

        Args:
            asset: Trading pair symbol
            order_position_state: Current order and position state
            existing_features: Existing features from Feature Service

        Returns:
            Dictionary of additional position/order feature names to values
        """
        position_features = {}

        # Extract current price from existing features (should be in FeatureVector)
        current_price = existing_features.get("mid_price", existing_features.get("price", 0.0))

        # Open orders features (must match training dataset features for consistency)
        asset_orders = [
            order
            for order in order_position_state.orders
            if order.asset == asset and order.status in ("pending", "partially_filled")
        ]
        position_features["open_orders_count"] = float(len(asset_orders))
        position_features["pending_buy_orders"] = float(len([o for o in asset_orders if o.side.upper() == "BUY"]))
        position_features["pending_sell_orders"] = float(len([o for o in asset_orders if o.side.upper() == "SELL"]))

        # Position features (from order/position state)
        position = order_position_state.get_position(asset)
        if position:
            position_features["position_size"] = float(position.size)
            position_features["position_size_abs"] = abs(float(position.size))
            position_features["unrealized_pnl"] = float(position.unrealized_pnl)
            position_features["realized_pnl"] = float(position.realized_pnl)
            position_features["has_position"] = 1.0 if position.size != 0 else 0.0
            if position.average_entry_price and position.average_entry_price > 0:
                position_features["entry_price"] = float(position.average_entry_price)
                position_features["price_vs_entry"] = (
                    (current_price - float(position.average_entry_price)) / float(position.average_entry_price) * 100
                )
            else:
                position_features["entry_price"] = current_price
                position_features["price_vs_entry"] = 0.0
        else:
            # No position for this asset
            position_features["position_size"] = 0.0
            position_features["position_size_abs"] = 0.0
            position_features["unrealized_pnl"] = 0.0
            position_features["realized_pnl"] = 0.0
            position_features["has_position"] = 0.0
            position_features["entry_price"] = current_price
            position_features["price_vs_entry"] = 0.0

        # Total exposure
        total_exposure = order_position_state.get_total_exposure(asset)
        position_features["total_exposure"] = float(total_exposure)
        position_features["total_exposure_abs"] = abs(float(total_exposure))

        # Asset features (categorical - hash encoding)
        position_features["asset_hash"] = float(hash(asset) % 1000)

        # Strategy features (categorical - will be set by caller if available)
        # Default to 0 if not provided
        position_features["strategy_hash"] = 0.0

        return position_features

    def _compute_legacy_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute legacy feature names from Feature Service features for backward compatibility.
        
        This function maps Feature Service feature names to old feature names that models
        were trained with before Feature Service integration.
        
        Args:
            features: DataFrame with features from Feature Service
            
        Returns:
            DataFrame with additional legacy features computed
        """
        # Feature name mappings: old_name -> computation from new features
        # spread_percent: can be computed from spread_rel or spread_abs / price * 100
        if "spread_percent" not in features.columns:
            if "spread_rel" in features.columns:
                # spread_rel is relative spread - typically in decimal format (0.001 = 0.1%)
                # Convert to percentage by multiplying by 100
                # If values are already > 1, assume they're already in percentage format
                spread_rel = features["spread_rel"]
                # Check if values are likely in percentage format (typical spread_percent is 0.01-1.0%)
                # If spread_rel values are > 1, they're likely already in percentage
                max_val = spread_rel.abs().max() if len(spread_rel) > 0 else 0.0
                if max_val > 1.0:
                    # Already in percentage format
                    features["spread_percent"] = spread_rel
                    logger.debug("spread_rel appears to be in percentage format, using directly")
                else:
                    # Decimal format (0.001 = 0.1%), convert to percentage
                    features["spread_percent"] = spread_rel * 100.0
                    logger.debug("spread_rel converted from decimal to percentage")
            elif "spread_abs" in features.columns:
                # Compute spread_percent from absolute spread and price
                price_col = None
                for price_name in ["mid_price", "price", "close_price", "last_price"]:
                    if price_name in features.columns:
                        price_col = price_name
                        break
                
                if price_col:
                    price = features[price_col]
                    spread_abs = features["spread_abs"]
                    # Avoid division by zero and handle inf/nan
                    features["spread_percent"] = (
                        (spread_abs / price * 100.0)
                        .fillna(0.0)
                        .replace([float('inf'), float('-inf')], 0.0)
                    )
                    logger.debug(f"spread_percent computed from spread_abs and {price_col}")
                else:
                    features["spread_percent"] = 0.0
                    logger.debug("Could not compute spread_percent: spread_abs available but no price column found")
            elif "bid_ask_spread" in features.columns:
                # Alternative: use bid_ask_spread if available
                price_col = None
                for price_name in ["mid_price", "price", "close_price", "last_price"]:
                    if price_name in features.columns:
                        price_col = price_name
                        break
                
                if price_col:
                    price = features[price_col]
                    spread = features["bid_ask_spread"]
                    features["spread_percent"] = (
                        (spread / price * 100.0)
                        .fillna(0.0)
                        .replace([float('inf'), float('-inf')], 0.0)
                    )
                    logger.debug(f"spread_percent computed from bid_ask_spread and {price_col}")
                else:
                    features["spread_percent"] = 0.0
                    logger.debug("Could not compute spread_percent: bid_ask_spread available but no price column found")
            else:
                # Fallback: set to 0 if cannot compute
                features["spread_percent"] = 0.0
                logger.debug("Could not compute spread_percent, using default 0.0")
        
        # Map other common legacy features
        # volume_24h: may be named differently in Feature Service
        if "volume_24h" not in features.columns:
            if "volume_24h_usdt" in features.columns:
                features["volume_24h"] = features["volume_24h_usdt"]
            elif "volume_24h_base" in features.columns:
                features["volume_24h"] = features["volume_24h_base"]
            else:
                # Try to get from any volume feature
                volume_cols = [col for col in features.columns if "volume" in col.lower()]
                if volume_cols:
                    features["volume_24h"] = features[volume_cols[0]]
                else:
                    features["volume_24h"] = 0.0
        
        # volatility: may be named differently
        if "volatility" not in features.columns:
            if "realized_volatility" in features.columns:
                features["volatility"] = features["realized_volatility"]
            elif "volatility_1h" in features.columns:
                features["volatility"] = features["volatility_1h"]
            else:
                features["volatility"] = 0.0
        
        # price: map from mid_price or close_price
        if "price" not in features.columns:
            if "mid_price" in features.columns:
                features["price"] = features["mid_price"]
            elif "close_price" in features.columns:
                features["price"] = features["close_price"]
            elif "last_price" in features.columns:
                features["price"] = features["last_price"]
        
        return features

    def _align_features(self, features: pd.DataFrame, model: Any) -> pd.DataFrame:
        """
        Align features with model's expected features.

        Performs feature alignment: validates that all required features are present,
        adds missing features with default values or computes them from available features,
        removes extra features, and preserves order.

        Args:
            features: DataFrame with features from Feature Service + position features
            model: Trained model object

        Returns:
            DataFrame with aligned features matching model's expected features

        Raises:
            ModelInferenceError: If critical features are missing
        """
        # Optionally compute legacy features for backward compatibility with old models
        # This should be disabled for new models trained on Feature Service features
        if settings.feature_service_legacy_feature_compatibility:
            features = self._compute_legacy_features(features)
        
        # Get expected feature names from model (what it was trained with)
        # XGBoost models store feature names in different places depending on version
        expected_feature_names = None
        
        # Try feature_names_in_ first (newer XGBoost versions)
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            expected_feature_names = list(model.feature_names_in_)
            logger.debug("Using feature_names_in_ from model", count=len(expected_feature_names))
        
        # Try get_booster().feature_names (older XGBoost versions or when feature_names_in_ is None)
        if expected_feature_names is None and hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                if hasattr(booster, "feature_names") and booster.feature_names:
                    expected_feature_names = list(booster.feature_names)
                    logger.debug("Using feature_names from booster", count=len(expected_feature_names))
            except Exception as e:
                logger.warning("Failed to get feature names from booster", error=str(e))
        
        # Fallback: use provided features (should not happen, but handle gracefully)
        if expected_feature_names is None or len(expected_feature_names) == 0:
            logger.warning(
                "Could not determine model's expected features, using provided features",
                provided_count=len(features.columns),
            )
            expected_feature_names = list(features.columns)
        
        logger.debug(
            "Features before alignment",
            feature_count=len(features.columns),
            model_expected_count=len(expected_feature_names),
        )
        
        # Define critical features that must be present (can be customized)
        critical_features = []  # Empty list means no features are critical - all can use defaults
        
        # Validate and align features
        missing_features = set(expected_feature_names) - set(features.columns)
        if missing_features:
            missing_critical = missing_features.intersection(critical_features)
            if missing_critical:
                error_msg = f"Critical features missing from FeatureVector: {missing_critical}"
                logger.error("Feature alignment failed - critical features missing", missing_features=missing_critical)
                raise ModelInferenceError(error_msg)
            
            # Check if model requires legacy features (old model trained before Feature Service integration)
            legacy_feature_patterns = ["spread_percent", "volume_24h", "volatility", "price"]
            has_legacy_features = any(
                any(pattern in feature_name.lower() for pattern in legacy_feature_patterns)
                for feature_name in missing_features
            )
            
            if has_legacy_features and not settings.feature_service_legacy_feature_compatibility:
                logger.warning(
                    "Model requires legacy features but legacy compatibility is disabled",
                    missing_features=list(missing_features),
                    recommendation="Either enable FEATURE_SERVICE_LEGACY_FEATURE_COMPATIBILITY=true or retrain model on Feature Service features",
                )
            
            # Add missing features with default values
            for feature_name in missing_features:
                # Use appropriate default based on feature name
                if "hash" in feature_name.lower():
                    default_value = 0
                elif "count" in feature_name.lower() or "has_" in feature_name.lower():
                    default_value = 0.0
                else:
                    default_value = 0.0
                
                features[feature_name] = default_value
                logger.warning(
                    "Missing feature in inference, using default",
                    feature_name=feature_name,
                    default_value=default_value,
                )
        
        # Remove extra features that model doesn't expect
        extra_features = set(features.columns) - set(expected_feature_names)
        if extra_features:
            logger.debug(
                "Removing extra features not expected by model",
                extra_features=list(extra_features),
                model_expected_count=len(expected_feature_names),
                provided_count=len(features.columns),
            )
            features = features.drop(columns=list(extra_features))
        
        # Reorder columns to match model's expected order
        features = features[expected_feature_names]
        
        logger.debug(
            "Features after alignment",
            feature_count=len(features.columns),
            aligned=True,
        )
        
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
            # Perform feature alignment with model's expected features
            features = self._align_features(features, model)
            
            # Handle missing values
            features = features.fillna(features.mean())

            # Check if model has predict_proba (classification)
            if hasattr(model, "predict_proba"):
                # Classification model
                probabilities = model.predict_proba(features)[0]

                # Determine semantic class probabilities using optional label mapping.
                # If the model was trained with remapped labels (e.g. {-1,+1}->{0,1}),
                # ModelTrainer can store a mapping {0: -1, 1: 1} on the model instance
                # under _label_mapping_for_inference. We use that to interpret
                # probabilities in terms of "up"/"down" classes.
                label_mapping = getattr(model, "_label_mapping_for_inference", None)
                task_variant = getattr(model, "_task_variant", None)
                probability_thresholds = getattr(model, "_probability_thresholds", None)

                # Build reverse mapping: semantic_label -> probability (if mapping available)
                sem_probs: Dict[Any, float] = {}
                if label_mapping and isinstance(label_mapping, dict):
                    for class_idx, sem_label in label_mapping.items():
                        try:
                            idx = int(class_idx)
                        except (TypeError, ValueError):
                            idx = class_idx
                        if isinstance(idx, int) and 0 <= idx < len(probabilities):
                            sem_probs[sem_label] = float(probabilities[idx])

                # Default: work directly in model class index space
                predicted_class_idx: Optional[int] = None
                semantic_prediction: Any = None

                # 1) Prefer per-model calibrated thresholds (learned on validation).
                #    Для бинарного направления {-1, +1} используем отдельные пороги
                #    для buy (+1) и sell (-1) и явно допускаем зону hold (0), если
                #    ни одна сторона не набрала достаточную уверенность.
                if probability_thresholds and isinstance(probability_thresholds, dict) and sem_probs:
                    # Normalize threshold keys to semantic labels (e.g. "-1", "1" -> -1, 1)
                    thresholds_sem: Dict[Any, float] = {}
                    for k, v in probability_thresholds.items():
                        try:
                            key = int(k)
                        except (TypeError, ValueError):
                            key = k
                        thresholds_sem[key] = float(v)

                    buy_threshold = thresholds_sem.get(1)
                    sell_threshold = thresholds_sem.get(-1)
                    p_buy = sem_probs.get(1, 0.0)
                    p_sell = sem_probs.get(-1, 0.0)

                    candidates: Dict[Any, float] = {}

                    # Кандидат на buy, если P(buy) >= T_buy
                    if buy_threshold is not None and 1 in sem_probs and p_buy >= buy_threshold:
                        candidates[1] = p_buy

                    # Кандидат на sell, если P(sell) >= T_sell
                    if sell_threshold is not None and -1 in sem_probs and p_sell >= sell_threshold:
                        candidates[-1] = p_sell

                    if candidates:
                        # Если обе стороны прошли порог, выбираем с максимальной вероятностью
                        semantic_prediction = max(candidates.items(), key=lambda kv: kv[1])[0]
                        logger.debug(
                            "Prediction with calibrated probability thresholds",
                            buy_threshold=buy_threshold,
                            sell_threshold=sell_threshold,
                            p_buy=p_buy,
                            p_sell=p_sell,
                            semantic_prediction=semantic_prediction,
                            passed_labels=list(candidates.keys()),
                        )
                    else:
                        # Ни один порог не выполнен — интерпретируем как отсутствие сигнала (hold)
                        semantic_prediction = 0
                        logger.debug(
                            "No calibrated thresholds passed; returning hold (0)",
                            buy_threshold=buy_threshold,
                            sell_threshold=sell_threshold,
                            p_buy=p_buy,
                            p_sell=p_sell,
                        )

                # 2) If no per-model thresholds, optionally use global thresholds from settings.
                if semantic_prediction is None:
                    # Use threshold-based prediction if enabled and thresholds are configured
                    if settings.model_prediction_use_threshold_calibration:
                        predicted_class_idx = self._predict_with_thresholds(probabilities)
                    else:
                        # Default: use argmax over model class indices
                        predicted_class_idx = int(np.argmax(probabilities))

                    # Map predicted class index back to semantic label if mapping is available
                    if label_mapping and isinstance(label_mapping, dict):
                        # label_mapping: {model_class_idx -> semantic_label (-1/0/+1, etc.)}
                        semantic_prediction = label_mapping.get(predicted_class_idx, predicted_class_idx)
                    else:
                        semantic_prediction = predicted_class_idx

                # Calculate confidence as max probability
                confidence = float(np.max(probabilities))

                # Derive buy/sell probabilities based on semantic labels if possible.
                buy_probability = 0.0
                sell_probability = 0.0

                if sem_probs:
                    # For directional binary targets we typically use -1 (down) and +1 (up).
                    if 1 in sem_probs or -1 in sem_probs:
                        buy_probability = float(sem_probs.get(1, 0.0))
                        sell_probability = float(sem_probs.get(-1, 0.0))
                    else:
                        # Fallback: try semantic labels 0/1 as up/down
                        buy_probability = float(sem_probs.get(1, 0.0))
                        sell_probability = float(sem_probs.get(0, 0.0))
                else:
                    # No semantic mapping available: fall back to legacy convention.
                    # For multi-class classification: [class_0_prob, class_1_prob]
                    # class_0 = buy, class_1 = sell (original behavior).
                    if len(probabilities) >= 2:
                        buy_probability = float(probabilities[0])
                        sell_probability = float(probabilities[1])
                    else:
                        # Fallback for single class or unexpected format
                        buy_probability = float(probabilities[0]) if len(probabilities) > 0 else 0.0
                        sell_probability = 0.0

                result = {
                    "prediction": int(semantic_prediction),
                    "confidence": confidence,
                    "probabilities": probabilities.tolist() if hasattr(probabilities, "tolist") else list(probabilities),
                    "buy_probability": buy_probability,
                    "sell_probability": sell_probability,
                }
            else:
                # Regression model
                prediction = model.predict(features)[0]
                predicted_return = float(prediction)

                # For regression, confidence is based on prediction magnitude.
                # Use configurable max expected return for normalization.
                # `settings` is imported at module level to avoid UnboundLocalError.
                max_expected_return = settings.model_regression_max_expected_return
                confidence = min(1.0, max(0.0, abs(predicted_return) / max_expected_return))

                result = {
                    "prediction": predicted_return,
                    "confidence": confidence,
                    "probabilities": None,
                    "predicted_return": predicted_return,  # Alias for clarity
                    "predicted_return_pct": predicted_return * 100,  # Percentage for logging
                }

            logger.debug("Model prediction completed", prediction=result.get("prediction"), confidence=result.get("confidence"))
            return result

        except Exception as e:
            logger.error("Model prediction failed", error=str(e), exc_info=True)
            raise ModelInferenceError(f"Model prediction failed: {e}") from e
    
    def _predict_with_thresholds(self, probabilities: np.ndarray) -> int:
        """
        Predict class using threshold-based approach instead of argmax.
        
        This improves recall for minority classes by using lower thresholds.
        If a class probability exceeds its threshold, that class is predicted.
        If multiple classes exceed thresholds, the one with highest probability is chosen.
        If no class exceeds threshold, falls back to argmax.
        
        Args:
            probabilities: Array of class probabilities [prob_class_0, prob_class_1, ...]
            
        Returns:
            Predicted class index
        """
        thresholds = {
            0: settings.model_prediction_threshold_class_0,
            1: settings.model_prediction_threshold_class_1,
            -1: settings.model_prediction_threshold_class_neg1,
        }
        
        # Determine class mapping based on number of classes
        # For 3 classes: [0, 1, 2] might map to [-1, 0, 1] or [0, 1, 2]
        num_classes = len(probabilities)
        
        # If we have 3 classes, assume mapping: [0, 1, 2] -> [-1, 0, 1] or [0, 1, 2]
        # We'll check thresholds for both mappings
        candidate_classes = []
        
        if num_classes == 3:
            # Try mapping [0, 1, 2] -> [-1, 0, 1]
            for model_class_idx, original_class in enumerate([-1, 0, 1]):
                threshold = thresholds.get(original_class)
                if threshold is not None and probabilities[model_class_idx] >= threshold:
                    candidate_classes.append((model_class_idx, probabilities[model_class_idx]))
            
            # If no candidates with [-1, 0, 1] mapping, try [0, 1, 2] mapping
            if not candidate_classes:
                for model_class_idx in range(num_classes):
                    threshold = thresholds.get(model_class_idx)
                    if threshold is not None and probabilities[model_class_idx] >= threshold:
                        candidate_classes.append((model_class_idx, probabilities[model_class_idx]))
        else:
            # For other numbers of classes, use direct mapping
            for model_class_idx in range(num_classes):
                threshold = thresholds.get(model_class_idx)
                if threshold is not None and probabilities[model_class_idx] >= threshold:
                    candidate_classes.append((model_class_idx, probabilities[model_class_idx]))
        
        # If we have candidates, choose the one with highest probability
        if candidate_classes:
            # Sort by probability descending
            candidate_classes.sort(key=lambda x: x[1], reverse=True)
            predicted_class = candidate_classes[0][0]
            logger.debug(
                "Threshold-based prediction",
                predicted_class=predicted_class,
                probability=float(candidate_classes[0][1]),
                candidates=[(c, float(p)) for c, p in candidate_classes],
            )
            return predicted_class
        
        # Fallback to argmax if no class exceeds threshold
        predicted_class = int(np.argmax(probabilities))
        logger.debug(
            "Threshold-based prediction: no class exceeded threshold, using argmax",
            predicted_class=predicted_class,
            probabilities=probabilities.tolist(),
            thresholds=thresholds,
        )
        return predicted_class


# Global model inference instance
model_inference = ModelInference()

