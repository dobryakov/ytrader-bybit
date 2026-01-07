"""
Unit tests for optimal top-k selection strategy in TrainingOrchestrator.
"""

from src.services.training_orchestrator import TrainingOrchestrator


def test_select_optimal_top_k_coverage_first_prefers_higher_k_with_similar_accuracy():
    """
    coverage_first should prefer larger k when accuracy is reasonably close
    to the best one (>= 50% of max accuracy).
    """
    orchestrator = TrainingOrchestrator()

    top_k_results = {
        "top_k_10_accuracy": 0.70,
        "top_k_10_lift": 1.5,
        "top_k_20_accuracy": 0.68,
        "top_k_20_lift": 1.4,
        "top_k_30_accuracy": 0.60,
        "top_k_30_lift": 1.3,
        "top_k_50_accuracy": 0.40,
        "top_k_50_lift": 1.1,
    }

    optimal_k = orchestrator._select_optimal_top_k_percentage(  # type: ignore[attr-defined]
        top_k_results=top_k_results,
        strategy="coverage_first",
        trace_id=None,
    )

    # Max accuracy = 0.70 -> 50% threshold = 0.35
    # All k have accuracy >= 0.35, so we should pick the largest k = 50
    assert optimal_k == 50


def test_select_optimal_top_k_coverage_first_drops_too_low_accuracy():
    """
    coverage_first should drop candidates whose accuracy is far below the best
    ( < 50% of max accuracy ) before maximizing k.
    """
    orchestrator = TrainingOrchestrator()

    top_k_results = {
        "top_k_10_accuracy": 0.80,
        "top_k_10_lift": 1.6,
        "top_k_20_accuracy": 0.78,
        "top_k_20_lift": 1.5,
        # 30% has very low accuracy compared to best (0.2 vs 0.8)
        "top_k_30_accuracy": 0.20,
        "top_k_30_lift": 1.4,
        "top_k_50_accuracy": 0.45,
        "top_k_50_lift": 1.2,
    }

    optimal_k = orchestrator._select_optimal_top_k_percentage(  # type: ignore[attr-defined]
        top_k_results=top_k_results,
        strategy="coverage_first",
        trace_id=None,
    )

    # Max accuracy = 0.80 -> 50% threshold = 0.40
    # k=30 has accuracy 0.20 < 0.40 -> excluded
    # Candidates for coverage_first: k=10 (0.8), k=20 (0.78), k=50 (0.45)
    # Among them, largest k is 50
    assert optimal_k == 50


