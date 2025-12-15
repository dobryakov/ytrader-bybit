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
        model_type="xgboost",
        task_type="regression",
        hyperparameters=hyperparameters,
    )

    assert isinstance(model, DummyXGBModel)

    # Ensure fit() did NOT receive unsupported kwargs that broke us earlier
    fit_kwargs = calls.get("fit_kwargs", {})
    assert "eval_metric" not in fit_kwargs
    assert "early_stopping_rounds" not in fit_kwargs


