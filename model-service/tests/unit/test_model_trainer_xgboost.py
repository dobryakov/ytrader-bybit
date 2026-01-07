"""
Unit tests for ModelTrainer XGBoost integration.

Goal: ensure that ModelTrainer.train_model does not pass unsupported
fit() keyword arguments (eval_metric, early_stopping_rounds) to the
installed xgboost version in the container.
"""

import pytest
import pandas as pd

from unittest.mock import MagicMock

from src.services.model_trainer import ModelTrainer
from src.models.training_dataset import TrainingDataset


@pytest.mark.asyncio
async def test_xgboost_fit_called_without_unsupported_kwargs(monkeypatch):
    """
    Verify that ModelTrainer uses only supported kwargs for XGBoost.fit().

    We don't rely on xgboost internals here; instead we patch the model_class
    used by ModelTrainer so we can inspect the actual call to fit().
    """
    # Create minimal fake dataset-like object
    # Simple training data
    X = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0.0, 1.0, 0.5, -0.5])
    
    # Create validation split (required for train_model)
    X_val = pd.DataFrame({"feature1": [5.0, 6.0], "feature2": [0.5, 0.6]})
    y_val = pd.Series([0.0, 1.0])

    dataset = TrainingDataset(
        strategy_id="test-strategy",
        features=X,
        labels=y,
    )

    trainer = ModelTrainer()

    # Patch supported_model_types so that we control the model implementation
    calls = {}

    class DummyXGBModel:
        def __init__(self, **kwargs):
            # Store constructor kwargs for assertions if needed
            calls["init_kwargs"] = kwargs

        def fit(self, *args, **kwargs):
            # Record the kwargs actually passed to fit()
            calls["fit_kwargs"] = kwargs
            # Simulate successful training
            return self

    trainer.supported_model_types = {
        "xgboost": {"classifier": DummyXGBModel, "regressor": DummyXGBModel}
    }

    # Enable early stopping via hyperparameters
    hyperparameters = {"early_stopping_rounds": 10}

    # Call train_model; it should not raise TypeError about unexpected kwargs
    model = trainer.train_model(
        dataset=dataset,
        validation_features=X_val,
        validation_labels=y_val,
        model_type="xgboost",
        task_type="regression",
        hyperparameters=hyperparameters,
    )

    assert isinstance(model, DummyXGBModel)

    # Ensure fit() did NOT receive unsupported kwargs that broke us earlier
    fit_kwargs = calls.get("fit_kwargs", {})
    assert "eval_metric" not in fit_kwargs
    assert "early_stopping_rounds" not in fit_kwargs


def test_train_model_uses_validation_split_for_early_stopping():
    """Test that train_model uses provided validation split for early stopping."""
    # Create training and validation data
    X = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0.0, 1.0, 0.0, 1.0])
    
    X_val = pd.DataFrame({"feature1": [5.0, 6.0], "feature2": [0.5, 0.6]})
    y_val = pd.Series([0.0, 1.0])

    dataset = TrainingDataset(
        strategy_id="test-strategy",
        features=X,
        labels=y,
    )

    trainer = ModelTrainer()
    calls = {}

    class DummyXGBModel:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs

        def fit(self, *args, **kwargs):
            calls["fit_kwargs"] = kwargs
            calls["fit_args"] = args
            # Check that eval_set contains validation data
            if "eval_set" in kwargs:
                eval_set = kwargs["eval_set"]
                assert len(eval_set) == 1
                val_features, val_labels = eval_set[0]
                # Verify validation data matches what we passed
                assert len(val_features) == len(X_val)
                assert len(val_labels) == len(y_val)
            return self

    trainer.supported_model_types = {
        "xgboost": {"classifier": DummyXGBModel, "regressor": DummyXGBModel}
    }

    hyperparameters = {"early_stopping_rounds": 10}

    model = trainer.train_model(
        dataset=dataset,
        validation_features=X_val,
        validation_labels=y_val,
        model_type="xgboost",
        task_type="classification",
        hyperparameters=hyperparameters,
    )

    assert isinstance(model, DummyXGBModel)
    # Verify eval_set was passed to fit()
    fit_kwargs = calls.get("fit_kwargs", {})
    assert "eval_set" in fit_kwargs
    # Verify validation data is in eval_set
    eval_set = fit_kwargs["eval_set"]
    assert len(eval_set) == 1
    val_features, val_labels = eval_set[0]
    assert len(val_features) == len(X_val)
    assert len(val_labels) == len(y_val)


def test_train_model_uses_validation_split_for_hyperparameter_optimization():
    """Test that _optimize_hyperparameters uses provided validation split."""
    # Create training and validation data
    X = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0.0, 1.0, 0.0, 1.0])
    
    X_val = pd.DataFrame({"feature1": [5.0, 6.0], "feature2": [0.5, 0.6]})
    y_val = pd.Series([0.0, 1.0])

    trainer = ModelTrainer()
    
    # Mock the model class to track calls
    optimization_calls = {}
    
    class DummyXGBModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            # Track that fit was called with training data
            optimization_calls["fit_called"] = True
            optimization_calls["fit_args"] = args
            return self
        
        def predict(self, X):
            # Track that predict was called with validation data
            optimization_calls["predict_called"] = True
            optimization_calls["predict_args"] = X
            # Return dummy predictions matching validation labels length
            return pd.Series([0.0] * len(X))

    trainer.supported_model_types = {
        "xgboost": {"classifier": DummyXGBModel, "regressor": DummyXGBModel}
    }
    
    # Patch settings to enable hyperparameter tuning
    from unittest.mock import patch
    with patch("src.services.model_trainer.settings") as mock_settings:
        mock_settings.model_training_hyperparameter_tuning = True
        mock_settings.model_training_tuning_method = "grid_search"
        mock_settings.model_training_tuning_max_iterations = 1  # Limit to 1 iteration for speed
        
        base_hyperparameters = {"n_estimators": 100}
        
        # Call _optimize_hyperparameters directly
        result = trainer._optimize_hyperparameters(
            X=X,
            y=y,
            validation_features=X_val,
            validation_labels=y_val,
            model_type="xgboost",
            task_type="classification",
            base_hyperparameters=base_hyperparameters,
        )
        
        # Verify that predict was called with validation features
        assert optimization_calls.get("predict_called", False)
        # Verify that predict received validation features
        predict_args = optimization_calls.get("predict_args")
        assert predict_args is not None
        assert len(predict_args) == len(X_val)


