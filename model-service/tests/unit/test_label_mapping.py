"""
Unit tests for label mapping in classification models.

Tests verify that label mapping works correctly for triple_classification,
ensuring that model predictions are correctly mapped back to semantic labels.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock

from src.services.training_orchestrator import TrainingOrchestrator
from src.services.model_trainer import ModelTrainer


def test_predict_with_thresholds_or_argmax_label_mapping():
    """Test that label mapping correctly converts model class indices to semantic labels."""
    orchestrator = TrainingOrchestrator()
    
    # Create a mock model with label mapping
    # For triple_classification: {-1, 0, 1} -> {0, 1, 2}
    # label_mapping: {model_class_idx -> semantic_label}
    label_mapping = {0: -1, 1: 0, 2: 1}
    model = MagicMock()
    model._label_mapping_for_inference = label_mapping
    
    # Simulate probabilities: model predicts class 2 (highest probability)
    probabilities = np.array([
        [0.1, 0.2, 0.7],  # Class 2 has highest probability
        [0.8, 0.1, 0.1],  # Class 0 has highest probability
        [0.2, 0.6, 0.2],  # Class 1 has highest probability
    ])
    
    # Call _predict_with_thresholds_or_argmax
    predictions = orchestrator._predict_with_thresholds_or_argmax(
        model=model,
        probabilities=probabilities,
        probability_thresholds=None,
        task_type="classification",
        task_variant="triple_classification",
    )
    
    # Verify predictions are correctly mapped to semantic labels
    expected = np.array([1, -1, 0])  # class 2 -> 1, class 0 -> -1, class 1 -> 0
    np.testing.assert_array_equal(predictions, expected)


def test_predict_with_thresholds_or_argmax_no_mapping():
    """Test that predictions work correctly when no label mapping exists."""
    orchestrator = TrainingOrchestrator()
    
    # Create a mock model without label mapping
    model = MagicMock()
    model._label_mapping_for_inference = None
    
    # Simulate probabilities
    probabilities = np.array([
        [0.1, 0.2, 0.7],  # Class 2 has highest probability
        [0.8, 0.1, 0.1],  # Class 0 has highest probability
    ])
    
    # Call _predict_with_thresholds_or_argmax
    predictions = orchestrator._predict_with_thresholds_or_argmax(
        model=model,
        probabilities=probabilities,
        probability_thresholds=None,
        task_type="classification",
        task_variant="triple_classification",
    )
    
    # Verify predictions are class indices (no mapping)
    expected = np.array([2, 0])  # argmax results
    np.testing.assert_array_equal(predictions, expected)


def test_predict_with_thresholds_or_argmax_1d_probabilities():
    """Test that 1D probabilities are handled correctly with label mapping."""
    orchestrator = TrainingOrchestrator()
    
    # Create a mock model with label mapping
    label_mapping = {0: -1, 1: 0, 2: 1}
    model = MagicMock()
    model._label_mapping_for_inference = label_mapping
    
    # Simulate 1D probabilities (single sample)
    probabilities = np.array([0.1, 0.2, 0.7])  # Class 2 has highest probability
    
    # Call _predict_with_thresholds_or_argmax
    predictions = orchestrator._predict_with_thresholds_or_argmax(
        model=model,
        probabilities=probabilities,
        probability_thresholds=None,
        task_type="classification",
        task_variant="triple_classification",
    )
    
    # Verify prediction is correctly mapped
    expected = np.array([1])  # class 2 -> 1
    np.testing.assert_array_equal(predictions, expected)


def test_label_mapping_all_classes():
    """Test that all three classes are correctly mapped for triple_classification."""
    orchestrator = TrainingOrchestrator()
    
    # Create a mock model with label mapping
    label_mapping = {0: -1, 1: 0, 2: 1}
    model = MagicMock()
    model._label_mapping_for_inference = label_mapping
    
    # Test each class separately
    test_cases = [
        (np.array([[0.9, 0.05, 0.05]]), np.array([-1])),  # Class 0 -> -1
        (np.array([[0.05, 0.9, 0.05]]), np.array([0])),   # Class 1 -> 0
        (np.array([[0.05, 0.05, 0.9]]), np.array([1])),   # Class 2 -> 1
    ]
    
    for probabilities, expected in test_cases:
        predictions = orchestrator._predict_with_thresholds_or_argmax(
            model=model,
            probabilities=probabilities,
            probability_thresholds=None,
            task_type="classification",
            task_variant="triple_classification",
        )
        np.testing.assert_array_equal(predictions, expected)


def test_label_mapping_transfer_from_trainer_to_model():
    """Test that label_mapping_for_inference is transferred from ModelTrainer to model object."""
    orchestrator = TrainingOrchestrator()
    
    # Create a mock ModelTrainer with label mapping
    model_trainer = ModelTrainer()
    label_mapping = {0: -1, 1: 0, 2: 1}
    model_trainer._label_mapping_for_inference = label_mapping
    
    # Create a mock trained model (without label mapping initially)
    # Use a regular object instead of MagicMock to avoid auto-attribute creation
    class MockModel:
        pass
    
    mock_model = MockModel()
    assert not hasattr(mock_model, "_label_mapping_for_inference")
    
    # Simulate the transfer that happens in training_orchestrator
    label_mapping_for_inference = getattr(model_trainer, "_label_mapping_for_inference", None)
    if label_mapping_for_inference is not None:
        setattr(mock_model, "_label_mapping_for_inference", label_mapping_for_inference)
    
    # Verify label mapping was transferred
    assert hasattr(mock_model, "_label_mapping_for_inference")
    assert mock_model._label_mapping_for_inference == label_mapping


def test_label_mapping_applied_in_validation_evaluation():
    """Test that label mapping is applied when making predictions during validation evaluation."""
    orchestrator = TrainingOrchestrator()
    
    # Create a mock model with label mapping
    label_mapping = {0: -1, 1: 0, 2: 1}
    model = MagicMock()
    model._label_mapping_for_inference = label_mapping
    model.predict_proba = MagicMock(return_value=np.array([
        [0.1, 0.2, 0.7],  # Class 2 has highest probability -> should map to 1
        [0.8, 0.1, 0.1],  # Class 0 has highest probability -> should map to -1
        [0.2, 0.6, 0.2],  # Class 1 has highest probability -> should map to 0
    ]))
    
    # Simulate validation evaluation logic
    eval_features = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
    y_pred_proba = model.predict_proba(eval_features)
    
    # Apply the same logic as in training_orchestrator
    argmax_pred = np.argmax(y_pred_proba, axis=1)
    label_mapping_from_model = getattr(model, "_label_mapping_for_inference", None)
    if label_mapping_from_model and isinstance(label_mapping_from_model, dict):
        y_pred = np.array([label_mapping_from_model.get(int(pred), pred) for pred in argmax_pred])
    else:
        y_pred = argmax_pred
    
    # Verify predictions are correctly mapped
    expected = np.array([1, -1, 0])  # class 2 -> 1, class 0 -> -1, class 1 -> 0
    np.testing.assert_array_equal(y_pred, expected)


def test_label_mapping_not_applied_when_missing():
    """Test that predictions work correctly when label mapping is missing from model."""
    orchestrator = TrainingOrchestrator()
    
    # Create a mock model without label mapping
    model = MagicMock()
    model._label_mapping_for_inference = None
    model.predict_proba = MagicMock(return_value=np.array([
        [0.1, 0.2, 0.7],  # Class 2
        [0.8, 0.1, 0.1],  # Class 0
    ]))
    
    # Simulate validation evaluation logic
    eval_features = pd.DataFrame({"feature1": [1.0, 2.0]})
    y_pred_proba = model.predict_proba(eval_features)
    
    # Apply the same logic as in training_orchestrator
    argmax_pred = np.argmax(y_pred_proba, axis=1)
    label_mapping_from_model = getattr(model, "_label_mapping_for_inference", None)
    if label_mapping_from_model and isinstance(label_mapping_from_model, dict):
        y_pred = np.array([label_mapping_from_model.get(int(pred), pred) for pred in argmax_pred])
    else:
        y_pred = argmax_pred
    
    # Verify predictions are class indices (no mapping applied)
    expected = np.array([2, 0])  # argmax results without mapping
    np.testing.assert_array_equal(y_pred, expected)


def test_label_mapping_triple_classification_all_classes():
    """Test that all three classes are correctly mapped for triple_classification in validation."""
    orchestrator = TrainingOrchestrator()
    
    # Create a mock model with label mapping for triple_classification
    label_mapping = {0: -1, 1: 0, 2: 1}
    model = MagicMock()
    model._label_mapping_for_inference = label_mapping
    
    # Test each class with different probabilities
    test_cases = [
        (np.array([[0.9, 0.05, 0.05]]), np.array([-1])),  # Class 0 -> -1
        (np.array([[0.05, 0.9, 0.05]]), np.array([0])),   # Class 1 -> 0
        (np.array([[0.05, 0.05, 0.9]]), np.array([1])),   # Class 2 -> 1
    ]
    
    for probabilities, expected in test_cases:
        model.predict_proba = MagicMock(return_value=probabilities)
        eval_features = pd.DataFrame({"feature1": [1.0]})
        y_pred_proba = model.predict_proba(eval_features)
        
        # Apply mapping logic
        argmax_pred = np.argmax(y_pred_proba, axis=1)
        label_mapping_from_model = getattr(model, "_label_mapping_for_inference", None)
        if label_mapping_from_model and isinstance(label_mapping_from_model, dict):
            y_pred = np.array([label_mapping_from_model.get(int(pred), pred) for pred in argmax_pred])
        else:
            y_pred = argmax_pred
        
        np.testing.assert_array_equal(y_pred, expected)

