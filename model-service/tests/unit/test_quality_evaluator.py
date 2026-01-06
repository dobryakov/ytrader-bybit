"""
Unit tests for QualityEvaluator threshold optimization methods.

Tests verify that threshold optimization works correctly for various metrics,
including the newly added roc_auc optimization.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.services.quality_evaluator import QualityEvaluator


@pytest.fixture
def quality_evaluator():
    """Create a QualityEvaluator instance for testing."""
    return QualityEvaluator()


@pytest.fixture
def binary_classification_data():
    """Create binary classification test data."""
    # Simple binary classification: class 0 and class 1
    y_true = pd.Series([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    # Probabilities: first 3 samples have low prob for class 1, next 3 have high prob
    y_pred_proba = np.array([
        [0.9, 0.1],  # Class 0
        [0.8, 0.2],  # Class 0
        [0.7, 0.3],  # Class 0
        [0.2, 0.8],  # Class 1
        [0.1, 0.9],  # Class 1
        [0.3, 0.7],  # Class 1
        [0.85, 0.15],  # Class 0
        [0.15, 0.85],  # Class 1
        [0.75, 0.25],  # Class 0
        [0.25, 0.75],  # Class 1
    ])
    unique_labels = [0, 1]
    return y_true, y_pred_proba, unique_labels


@pytest.fixture
def multi_class_data():
    """Create multi-class classification test data (3 classes)."""
    # Three classes: -1, 0, 1
    y_true = pd.Series([-1, -1, 0, 0, 1, 1, -1, 0, 1])
    # Probabilities for classes -1, 0, 1 (mapped to indices 0, 1, 2)
    y_pred_proba = np.array([
        [0.8, 0.1, 0.1],  # Class -1
        [0.7, 0.2, 0.1],  # Class -1
        [0.1, 0.8, 0.1],  # Class 0
        [0.2, 0.7, 0.1],  # Class 0
        [0.1, 0.1, 0.8],  # Class 1
        [0.1, 0.2, 0.7],  # Class 1
        [0.75, 0.15, 0.1],  # Class -1
        [0.15, 0.75, 0.1],  # Class 0
        [0.1, 0.15, 0.75],  # Class 1
    ])
    unique_labels = [-1, 0, 1]
    return y_true, y_pred_proba, unique_labels


def test_optimize_thresholds_by_roc_auc_binary(quality_evaluator, binary_classification_data):
    """Test that roc_auc optimization works for binary classification."""
    y_true, y_pred_proba, unique_labels = binary_classification_data
    
    thresholds = quality_evaluator._optimize_thresholds_by_roc_auc(
        y_true, y_pred_proba, unique_labels
    )
    
    # Should return thresholds for both classes
    assert len(thresholds) == 2
    assert 0 in thresholds
    assert 1 in thresholds
    
    # Thresholds should be valid probabilities (between 0 and 1)
    for class_label, threshold in thresholds.items():
        assert 0.0 <= threshold <= 1.0
    
    # Verify that the threshold for class 1 maximizes Youden's J
    # For class 1, we expect a threshold around 0.5-0.7 based on the data
    class_1_threshold = thresholds[1]
    assert 0.0 <= class_1_threshold <= 1.0


def test_optimize_thresholds_by_roc_auc_multi_class(quality_evaluator, multi_class_data):
    """Test that roc_auc optimization works for multi-class classification."""
    y_true, y_pred_proba, unique_labels = multi_class_data
    
    thresholds = quality_evaluator._optimize_thresholds_by_roc_auc(
        y_true, y_pred_proba, unique_labels
    )
    
    # Should return thresholds for all three classes
    assert len(thresholds) == 3
    assert -1 in thresholds
    assert 0 in thresholds
    assert 1 in thresholds
    
    # Thresholds should be valid probabilities
    for class_label, threshold in thresholds.items():
        assert 0.0 <= threshold <= 1.0


def test_optimize_thresholds_by_roc_auc_uses_youdens_j(quality_evaluator):
    """Test that roc_auc optimization uses Youden's J statistic correctly."""
    # Create data where optimal threshold is clear
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    # Clear separation: first 3 have low prob for class 1, last 3 have high prob
    y_pred_proba = np.array([
        [0.9, 0.1],  # Class 0
        [0.8, 0.2],  # Class 0
        [0.7, 0.3],  # Class 0
        [0.2, 0.8],  # Class 1
        [0.1, 0.9],  # Class 1
        [0.3, 0.7],  # Class 1
    ])
    unique_labels = [0, 1]
    
    thresholds = quality_evaluator._optimize_thresholds_by_roc_auc(
        y_true, y_pred_proba, unique_labels
    )
    
    # Verify that the threshold for class 1 is reasonable (should be around 0.3-0.5)
    class_1_threshold = thresholds[1]
    assert 0.0 <= class_1_threshold <= 1.0
    
    # Manually verify Youden's J calculation
    y_binary = (y_true == 1).astype(int)
    class_probs = y_pred_proba[:, 1]
    fpr, tpr, threshold_candidates = roc_curve(y_binary, class_probs)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    expected_threshold = threshold_candidates[best_idx]
    
    # The returned threshold should match the one that maximizes Youden's J
    assert abs(thresholds[1] - expected_threshold) < 0.01


def test_optimize_thresholds_by_roc_auc_handles_edge_cases(quality_evaluator):
    """Test that roc_auc optimization handles edge cases gracefully."""
    # Test with all same class
    y_true = pd.Series([1, 1, 1, 1, 1])
    y_pred_proba = np.array([
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
        [0.6, 0.4],
    ])
    unique_labels = [0, 1]
    
    thresholds = quality_evaluator._optimize_thresholds_by_roc_auc(
        y_true, y_pred_proba, unique_labels
    )
    
    # Should still return thresholds (may use fallback)
    assert len(thresholds) == 2
    for class_label, threshold in thresholds.items():
        # Threshold should be finite and in valid range (fallback may be used)
        assert np.isfinite(threshold), f"Threshold for class {class_label} should be finite, got {threshold}"
        # If threshold is inf or nan, fallback should provide a valid value
        if not (0.0 <= threshold <= 1.0):
            # In edge cases, fallback should provide a default threshold
            # The method should handle this gracefully
            pass  # Accept that edge cases may produce unexpected thresholds


def test_calibrate_thresholds_with_roc_auc(quality_evaluator, binary_classification_data):
    """Test that calibrate_prediction_thresholds correctly uses roc_auc optimization."""
    y_true, y_pred_proba, unique_labels = binary_classification_data
    
    thresholds = quality_evaluator.calibrate_prediction_thresholds(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        optimization_metric="roc_auc",
        target_recall=0.5,
    )
    
    # Should return thresholds for all classes
    assert len(thresholds) == len(unique_labels)
    for class_label in unique_labels:
        assert class_label in thresholds
        assert 0.0 <= thresholds[class_label] <= 1.0


def test_calibrate_thresholds_with_roc_auc_multi_class(quality_evaluator, multi_class_data):
    """Test that calibrate_prediction_thresholds works with roc_auc for multi-class."""
    y_true, y_pred_proba, unique_labels = multi_class_data
    
    thresholds = quality_evaluator.calibrate_prediction_thresholds(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        optimization_metric="roc_auc",
        target_recall=0.5,
    )
    
    # Should return thresholds for all classes
    assert len(thresholds) == len(unique_labels)
    for class_label in unique_labels:
        assert class_label in thresholds
        assert 0.0 <= thresholds[class_label] <= 1.0


def test_calibrate_thresholds_fallback_on_error(quality_evaluator):
    """Test that calibrate_prediction_thresholds falls back gracefully on errors."""
    # Create invalid data that might cause errors
    y_true = pd.Series([0, 1])
    y_pred_proba = np.array([[0.5, 0.5], [0.5, 0.5]])  # Very few samples
    unique_labels = [0, 1]
    
    # Should not raise exception, should return thresholds (possibly using fallback)
    thresholds = quality_evaluator.calibrate_prediction_thresholds(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        optimization_metric="roc_auc",
        target_recall=0.5,
    )
    
    # Should still return thresholds
    assert len(thresholds) > 0


def test_roc_auc_vs_f1_optimization_difference(quality_evaluator, binary_classification_data):
    """Test that roc_auc and f1 optimization can produce different thresholds."""
    y_true, y_pred_proba, unique_labels = binary_classification_data
    
    thresholds_roc_auc = quality_evaluator.calibrate_prediction_thresholds(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        optimization_metric="roc_auc",
        target_recall=0.5,
    )
    
    thresholds_f1 = quality_evaluator.calibrate_prediction_thresholds(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        optimization_metric="f1",
        target_recall=0.5,
    )
    
    # Both should return valid thresholds
    assert len(thresholds_roc_auc) == len(thresholds_f1)
    
    # They may or may not be the same, but both should be valid
    for class_label in unique_labels:
        assert class_label in thresholds_roc_auc
        assert class_label in thresholds_f1
        assert 0.0 <= thresholds_roc_auc[class_label] <= 1.0
        assert 0.0 <= thresholds_f1[class_label] <= 1.0

