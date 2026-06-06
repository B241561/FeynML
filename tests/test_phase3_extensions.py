"""
Tests for Phase 3 Extension Implementations
==========================================
Tests for newly implemented Phase 3 features:
- SHAP plotting functions
- Error clustering (UMAP, failure clustering, confusion patterns)
- Label shift detection
- Manual slice discovery
- Covariate shift detection
- Enhanced unsupervised drift detection
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.abspath(os.path.join(_HERE, "..", "scratch", "phase3"))
if _PHASE3 not in sys.path:
    sys.path.insert(0, _PHASE3)


def test_shap_dependence_data():
    """Test SHAP dependence plot data preparation."""
    from shap_explainer import shap_dependence_data
    
    shap_matrix = [[0.1, -0.2, 0.3], [0.2, -0.1, 0.4], [0.15, -0.25, 0.35]]
    feature_values = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [0.8, 1.8, 2.8]]
    feature_names = ["f0", "f1", "f2"]
    
    result = shap_dependence_data(shap_matrix, feature_values, feature_names, 0)
    
    assert result["feature"] == "f0"
    assert result["feature_idx"] == 0
    assert len(result["x_values"]) == 3
    assert len(result["y_values"]) == 3
    assert "correlation" in result
    assert "interpretation" in result
    print("✓ SHAP dependence data test passed")


def test_shap_summary_plot_data():
    """Test SHAP summary plot data preparation."""
    from shap_explainer import shap_summary_plot_data
    
    shap_matrix = [[0.1, -0.2, 0.3], [0.2, -0.1, 0.4], [0.15, -0.25, 0.35]]
    feature_values = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [0.8, 1.8, 2.8]]
    feature_names = ["f0", "f1", "f2"]
    
    result = shap_summary_plot_data(shap_matrix, feature_values, feature_names)
    
    assert "features" in result
    assert len(result["features"]) == 3
    assert result["n_samples"] == 3
    assert result["n_features"] == 3
    assert "interpretation" in result
    assert "normalized_values" in result["features"][0]
    print("✓ SHAP summary plot data test passed")


def test_shap_waterfall_plot_data():
    """Test SHAP waterfall plot data preparation."""
    from shap_explainer import shap_waterfall_plot_data
    
    shap_values = [0.5, -0.3, 0.2, -0.1]
    feature_names = ["age", "income", "credit_score", "debt_ratio"]
    feature_values = [35, 50000, 720, 0.3]
    base_value = 0.4
    prediction = 0.7
    
    result = shap_waterfall_plot_data(shap_values, feature_names, feature_values, base_value, prediction)
    
    assert result["base_value"] == 0.4
    assert result["prediction"] == 0.7
    assert len(result["bars"]) == 4
    assert "x_position" in result["bars"][0]
    assert "color_intensity" in result["bars"][0]
    assert "interpretation" in result
    print("✓ SHAP waterfall plot data test passed")


def test_error_embeddings():
    """Test error embeddings computation."""
    from error_clustering import compute_error_embeddings
    import random
    
    random.seed(42)
    X = [[1.0, 2.0], [1.5, 2.5], [0.8, 1.8], [1.2, 2.2]]
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]  # One error
    
    result = compute_error_embeddings(X, y_true, y_pred, n_components=2, seed=42)
    
    assert "embeddings" in result
    assert "error_labels" in result
    assert len(result["embeddings"]) == 4
    assert len(result["error_labels"]) == 4
    assert result["error_rate"] == 0.25
    print("✓ Error embeddings test passed")


def test_failure_clustering():
    """Test failure clustering."""
    from error_clustering import cluster_errors
    import random
    
    random.seed(42)
    X = [[1.0, 2.0], [1.5, 2.5], [0.8, 1.8], [1.2, 2.2], [5.0, 6.0]]
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 0, 0, 1, 1]  # Two errors
    
    result = cluster_errors(X, y_true, y_pred, n_clusters=2, seed=42)
    
    assert "n_errors" in result
    assert result["n_errors"] == 2
    assert "clusters" in result
    print("✓ Failure clustering test passed")


def test_confusion_pattern_analysis():
    """Test confusion pattern analysis."""
    from error_clustering import confusion_pattern_analysis
    
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 2, 2, 0]  # Some errors
    
    result = confusion_pattern_analysis(y_true, y_pred)
    
    assert "confusion_matrix" in result
    assert "overall_accuracy" in result
    assert "error_patterns" in result
    assert "per_class_recall" in result
    assert "per_class_precision" in result
    assert result["n_classes"] == 3
    print("✓ Confusion pattern analysis test passed")


def test_label_shift_detection():
    """Test label shift detection."""
    from drift_detection import label_shift_detection
    
    y_train = [0] * 80 + [1] * 20  # 80% class 0
    y_current = [0] * 50 + [1] * 50  # 50% class 0 (shifted)
    
    result = label_shift_detection(y_train, y_current, alpha=0.05)
    
    assert "chi2_statistic" in result
    assert "label_details" in result
    assert "shift_detected" in result
    assert len(result["label_details"]) == 2
    print("✓ Label shift detection test passed")


def test_covariate_shift_detection():
    """Test covariate shift detection."""
    from drift_detection import covariate_shift_detection
    import random
    
    random.seed(42)
    X_train = [[random.gauss(0, 1) for _ in range(3)] for _ in range(100)]
    X_current = [[random.gauss(2, 1) for _ in range(3)] for _ in range(100)]  # Shifted
    
    result = covariate_shift_detection(X_train, X_current, feature_names=["f0", "f1", "f2"])
    
    assert "feature_shifts" in result
    assert result["n_features"] == 3
    assert "overall_shift" in result
    assert len(result["feature_shifts"]) == 3
    print("✓ Covariate shift detection test passed")


def test_manual_slice_explorer():
    """Test manual slice explorer."""
    from slice_finder import ManualSliceExplorer
    import random
    
    random.seed(42)
    data_rows = [
        {"age": "young", "gender": "male"},
        {"age": "young", "gender": "female"},
        {"age": "senior", "gender": "male"},
        {"age": "senior", "gender": "female"},
    ] * 25
    losses = [random.random() for _ in range(len(data_rows))]
    
    explorer = ManualSliceExplorer(data_rows, losses, ["age", "gender"])
    
    # Test evaluate_slice
    result = explorer.evaluate_slice({"age": "senior"})
    assert "size" in result
    assert "slice_loss" in result
    assert "effect_size" in result
    
    # Test suggest_slices
    suggestions = explorer.suggest_slices(max_slices=5, effect_threshold=0.0)
    assert isinstance(suggestions, list)
    
    print("✓ Manual slice explorer test passed")


def test_enhanced_unsupervised_drift_detection():
    """Test enhanced unsupervised drift detection."""
    from drift_detection import enhanced_unsupervised_drift_detection
    import random
    
    random.seed(42)
    X_ref = [[random.gauss(0, 1) for _ in range(3)] for _ in range(50)]
    X_cur = [[random.gauss(2, 1) for _ in range(3)] for _ in range(50)]  # Shifted
    
    result = enhanced_unsupervised_drift_detection(X_ref, X_cur, feature_names=["f0", "f1", "f2"])
    
    assert "methods" in result
    assert "overall_drift" in result
    assert "confidence" in result
    assert "domain_classifier" in result["methods"]
    assert "mmd" in result["methods"]
    print("✓ Enhanced unsupervised drift detection test passed")


def run_all_tests():
    """Run all Phase 3 extension tests."""
    print("=" * 65)
    print("Phase 3 Extension Tests")
    print("=" * 65)
    
    try:
        test_shap_dependence_data()
        test_shap_summary_plot_data()
        test_shap_waterfall_plot_data()
        test_error_embeddings()
        test_failure_clustering()
        test_confusion_pattern_analysis()
        test_label_shift_detection()
        test_covariate_shift_detection()
        test_manual_slice_explorer()
        test_enhanced_unsupervised_drift_detection()
        
        print("\n" + "=" * 65)
        print("All Phase 3 extension tests PASSED ✓")
        print("=" * 65)
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
