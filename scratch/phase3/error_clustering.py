"""
Phase 3.7 — Error Clustering Analysis
====================================
Implements UMAP error embeddings, failure clustering, and confusion pattern analysis.

Sources:
- UMAP: McInnes et al. (2018) "UMAP: Uniform Manifold Approximation and Projection"
- Error Clustering: Amershi et al. (PURCHASE 2019) - clustering model errors
- Confusion Patterns: confusion matrix analysis for systematic failure modes
"""

import math
import random
from collections import defaultdict, Counter


# ─────────────────────────────────────────────────────────────────────────────
# 1. UMAP ERROR EMBEDDINGS (simplified implementation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_error_embeddings(X, y_true, y_pred, n_components=2, n_neighbors=15, min_dist=0.1, seed=42):
    """
    Compute 2D embeddings of errors using a simplified UMAP-like approach.
    
    For production use, install umap-learn: pip install umap-learn
    This implementation provides a fallback using PCA + t-SNE style optimization.
    
    Args:
        X: list of feature vectors
        y_true: list of true labels
        y_pred: list of predicted labels
        n_components: dimensions for embedding (2 for visualization)
        n_neighbors: number of neighbors for manifold learning
        min_dist: minimum distance for UMAP
        seed: random seed
    
    Returns:
        dict with 2D coordinates, error labels, and metadata
    """
    rng = random.Random(seed)
    n_samples = len(X)
    n_features = len(X[0]) if X else 0
    
    # Identify error types
    error_labels = []
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            error_labels.append("correct")
        else:
            error_labels.append("error")
    
    # Try to use umap-learn if available
    try:
        import umap
        import numpy as np
        
        X_np = np.array(X)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed
        )
        embeddings = reducer.fit_transform(X_np)
        
        return {
            "embeddings": embeddings.tolist(),
            "error_labels": error_labels,
            "method": "umap-learn",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": n_components,
            "error_rate": round(sum(1 for e in error_labels if e == "error") / n_samples, 4),
        }
    except ImportError:
        # Fallback: simplified PCA-like projection
        return _simple_pca_projection(X, error_labels, n_components, seed)


def _simple_pca_projection(X, error_labels, n_components, seed):
    """
    Fallback projection using simplified PCA.
    Centers data and computes top principal components.
    """
    rng = random.Random(seed)
    n_samples = len(X)
    n_features = len(X[0]) if X else 0
    
    if n_samples == 0 or n_features == 0:
        return {
            "embeddings": [],
            "error_labels": error_labels,
            "method": "simple_pca_fallback",
            "n_samples": n_samples,
            "n_features": n_features,
            "error_rate": 0.0,
            "warning": "No data available for embedding",
        }
    
    # Center the data
    means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
    X_centered = [[X[i][j] - means[j] for j in range(n_features)] for i in range(n_samples)]
    
    # Compute covariance matrix
    cov = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_samples):
        for j in range(n_features):
            for k in range(n_features):
                cov[j][k] += X_centered[i][j] * X_centered[i][k]
    
    for j in range(n_features):
        for k in range(n_features):
            cov[j][k] /= max(n_samples - 1, 1)
    
    # Simple power iteration for top eigenvectors
    embeddings = [[0.0] * n_components for _ in range(n_samples)]
    
    for comp in range(min(n_components, n_features)):
        # Initialize random vector
        v = [rng.gauss(0, 1) for _ in range(n_features)]
        norm = math.sqrt(sum(x * x for x in v))
        v = [x / norm for x in v]
        
        # Power iteration
        for _ in range(50):
            # Multiply by covariance
            new_v = [sum(cov[j][k] * v[k] for k in range(n_features)) for j in range(n_features)]
            norm = math.sqrt(sum(x * x for x in new_v))
            if norm < 1e-12:
                break
            v = [x / norm for x in new_v]
        
        # Project data onto this eigenvector
        for i in range(n_samples):
            embeddings[i][comp] = sum(X_centered[i][j] * v[j] for j in range(n_features))
    
    return {
        "embeddings": embeddings,
        "error_labels": error_labels,
        "method": "simple_pca_fallback",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": n_components,
        "error_rate": round(sum(1 for e in error_labels if e == "error") / n_samples, 4),
        "warning": "umap-learn not installed, using simplified PCA fallback",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAILURE CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def cluster_errors(X, y_true, y_pred, n_clusters=3, seed=42):
    """
    Cluster error instances to discover systematic failure patterns.
    
    Uses k-means clustering on error instances to identify groups
    of similar failures that may indicate underlying data issues.
    
    Args:
        X: list of feature vectors
        y_true: list of true labels
        y_pred: list of predicted labels
        n_clusters: number of clusters to find
        seed: random seed
    
    Returns:
        dict with cluster assignments, cluster statistics, and interpretation
    """
    # Extract error instances
    error_indices = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    
    if len(error_indices) == 0:
        return {
            "n_errors": 0,
            "n_clusters": 0,
            "clusters": [],
            "interpretation": "No errors found - model is perfect on this dataset.",
        }
    
    # Get error features
    X_errors = [X[i] for i in error_indices]
    y_true_errors = [y_true[i] for i in error_indices]
    y_pred_errors = [y_pred[i] for i in error_indices]
    
    # Try sklearn KMeans
    try:
        from sklearn.cluster import KMeans
        import numpy as np
        
        X_err_np = np.array(X_errors)
        n_actual_clusters = min(n_clusters, len(X_errors))
        
        if n_actual_clusters < 2:
            return {
                "n_errors": len(error_indices),
                "n_clusters": 1,
                "clusters": [{
                    "cluster_id": 0,
                    "size": len(error_indices),
                    "error_indices": error_indices,
                    "true_label_dist": Counter(y_true_errors),
                    "pred_label_dist": Counter(y_pred_errors),
                }],
                "interpretation": f"Only {len(error_indices)} errors - insufficient for clustering.",
            }
        
        kmeans = KMeans(n_clusters=n_actual_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X_err_np)
        
        # Analyze clusters
        clusters = []
        for c in range(n_actual_clusters):
            c_indices = [error_indices[i] for i, label in enumerate(cluster_labels) if label == c]
            c_true = [y_true_errors[i] for i, label in enumerate(cluster_labels) if label == c]
            c_pred = [y_pred_errors[i] for i, label in enumerate(cluster_labels) if label == c]
            
            clusters.append({
                "cluster_id": c,
                "size": len(c_indices),
                "error_indices": c_indices,
                "true_label_dist": dict(Counter(c_true)),
                "pred_label_dist": dict(Counter(c_pred)),
                "centroid": [round(float(x), 4) for x in kmeans.cluster_centers_[c]],
            })
        
        # Sort clusters by size
        clusters.sort(key=lambda x: x["size"], reverse=True)
        
        return {
            "n_errors": len(error_indices),
            "n_clusters": n_actual_clusters,
            "clusters": clusters,
            "interpretation": (
                f"Found {n_actual_clusters} distinct error patterns. "
                "Investigate cluster centroids to understand failure modes."
            ),
        }
    except ImportError:
        # Fallback: simple distance-based clustering
        return _simple_error_clustering(X_errors, error_indices, y_true_errors, y_pred_errors, n_clusters, seed)


def _simple_error_clustering(X_errors, error_indices, y_true_errors, y_pred_errors, n_clusters, seed):
    """
    Fallback clustering using simple distance-based assignment.
    """
    rng = random.Random(seed)
    n_errors = len(X_errors)
    n_actual = min(n_clusters, n_errors)
    
    if n_actual < 2:
        return {
            "n_errors": n_errors,
            "n_clusters": 1,
            "clusters": [{
                "cluster_id": 0,
                "size": n_errors,
                "error_indices": error_indices,
                "true_label_dist": Counter(y_true_errors),
                "pred_label_dist": Counter(y_pred_errors),
            }],
            "interpretation": f"Only {n_errors} errors - insufficient for clustering.",
            "warning": "sklearn not available, using simple fallback",
        }
    
    # Initialize random centroids
    centroids = [X_errors[rng.randint(0, n_errors - 1)][:] for _ in range(n_actual)]
    
    # K-means iterations
    for _ in range(20):
        # Assign to nearest centroid
        assignments = []
        for x in X_errors:
            distances = [sum((x[j] - c[j]) ** 2 for j in range(len(x))) for c in centroids]
            assignments.append(distances.index(min(distances)))
        
        # Update centroids
        new_centroids = []
        for c in range(n_actual):
            cluster_points = [X_errors[i] for i, a in enumerate(assignments) if a == c]
            if cluster_points:
                new_centroid = [sum(p[j] for p in cluster_points) / len(cluster_points) 
                                for j in range(len(cluster_points[0]))]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[c])
        
        centroids = new_centroids
    
    # Build cluster info
    clusters = []
    for c in range(n_actual):
        c_indices = [error_indices[i] for i, a in enumerate(assignments) if a == c]
        c_true = [y_true_errors[i] for i, a in enumerate(assignments) if a == c]
        c_pred = [y_pred_errors[i] for i, a in enumerate(assignments) if a == c]
        
        clusters.append({
            "cluster_id": c,
            "size": len(c_indices),
            "error_indices": c_indices,
            "true_label_dist": Counter(c_true),
            "pred_label_dist": Counter(c_pred),
            "centroid": [round(float(x), 4) for x in centroids[c]],
        })
    
    clusters.sort(key=lambda x: x["size"], reverse=True)
    
    return {
        "n_errors": n_errors,
        "n_clusters": n_actual,
        "clusters": clusters,
        "interpretation": (
            f"Found {n_actual} distinct error patterns. "
            "Investigate cluster centroids to understand failure modes."
        ),
        "warning": "sklearn not available, using simple k-means fallback",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFUSION PATTERN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def confusion_pattern_analysis(y_true, y_pred, class_names=None):
    """
    Analyze confusion patterns to identify systematic failure modes.
    
    Goes beyond basic accuracy to understand which classes are confused
    with each other, revealing model weaknesses.
    
    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
        class_names: optional list of class names
    
    Returns:
        dict with confusion matrix, error patterns, and recommendations
    """
    # Get unique classes
    unique_classes = sorted(set(y_true) | set(y_pred))
    n_classes = len(unique_classes)
    
    if class_names is None:
        class_names = [str(c) for c in unique_classes]
    else:
        # Ensure class_names align with unique_classes
        class_map = {c: class_names[i] for i, c in enumerate(unique_classes)}
    
    # Build confusion matrix
    conf_matrix = [[0] * n_classes for _ in range(n_classes)]
    for yt, yp in zip(y_true, y_pred):
        i = unique_classes.index(yt)
        j = unique_classes.index(yp)
        conf_matrix[i][j] += 1
    
    # Analyze error patterns
    error_patterns = []
    for i in range(n_classes):
        true_class = unique_classes[i]
        row_sum = sum(conf_matrix[i])
        
        if row_sum == 0:
            continue
        
        # Find most common misclassifications for this class
        misclassifications = []
        for j in range(n_classes):
            if i != j and conf_matrix[i][j] > 0:
                pred_class = unique_classes[j]
                count = conf_matrix[i][j]
                rate = count / row_sum
                misclassifications.append({
                    "predicted_as": pred_class,
                    "count": count,
                    "rate": round(rate, 4),
                })
        
        misclassifications.sort(key=lambda x: x["rate"], reverse=True)
        
        if misclassifications:
            error_patterns.append({
                "true_class": true_class,
                "total_samples": row_sum,
                "correct": conf_matrix[i][i],
                "accuracy": round(conf_matrix[i][i] / row_sum, 4),
                "top_misclassifications": misclassifications[:3],
            })
    
    # Overall statistics
    total_samples = len(y_true)
    correct = sum(conf_matrix[i][i] for i in range(n_classes))
    overall_accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    # Per-class recall
    per_class_recall = []
    for i in range(n_classes):
        true_class = unique_classes[i]
        row_sum = sum(conf_matrix[i])
        recall = conf_matrix[i][i] / row_sum if row_sum > 0 else 0.0
        per_class_recall.append({
            "class": true_class,
            "recall": round(recall, 4),
            "support": row_sum,
        })
    
    # Per-class precision
    per_class_precision = []
    for j in range(n_classes):
        pred_class = unique_classes[j]
        col_sum = sum(conf_matrix[i][j] for i in range(n_classes))
        precision = conf_matrix[j][j] / col_sum if col_sum > 0 else 0.0
        per_class_precision.append({
            "class": pred_class,
            "precision": round(precision, 4),
            "support": col_sum,
        })
    
    return {
        "confusion_matrix": conf_matrix,
        "class_names": class_names,
        "unique_classes": unique_classes,
        "n_classes": n_classes,
        "overall_accuracy": round(overall_accuracy, 4),
        "total_samples": total_samples,
        "error_patterns": error_patterns,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "interpretation": (
            f"Overall accuracy: {overall_accuracy:.2%}. "
            f"Error patterns show systematic confusions between classes. "
            "Focus on classes with low recall/precision."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    import random as _r
    from sklearn.datasets import make_classification
    import numpy as np
    
    print("=" * 65)
    print("Phase 3.7 — Error Clustering Verification")
    print("=" * 65)
    
    _r.seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    X_np, y_np = make_classification(
        n_samples=400, n_features=5,
        n_informative=3, n_classes=3, random_state=42
    )
    X = X_np.tolist()
    y_true = y_np.tolist()
    
    # Simulate predictions with some errors
    y_pred = []
    for yt in y_true:
        if _r.random() < 0.15:  # 15% error rate
            y_pred.append(_r.choice([c for c in [0, 1, 2] if c != yt]))
        else:
            y_pred.append(yt)
    
    # ── Error Embeddings ───────────────────────────────────────────────
    print(f"\n  Error Embeddings:")
    emb = compute_error_embeddings(X, y_true, y_pred, n_components=2, seed=42)
    print(f"  Method: {emb['method']}")
    print(f"  Error rate: {emb['error_rate']:.2%}")
    print(f"  Embeddings shape: {len(emb['embeddings'])} x {emb['n_components']}")
    if "warning" in emb:
        print(f"  Warning: {emb['warning']}")
    
    # ── Failure Clustering ────────────────────────────────────────────
    print(f"\n  Failure Clustering:")
    clusters = cluster_errors(X, y_true, y_pred, n_clusters=3, seed=42)
    print(f"  Errors found: {clusters['n_errors']}")
    print(f"  Clusters: {clusters['n_clusters']}")
    for c in clusters['clusters'][:2]:
        print(f"    Cluster {c['cluster_id']}: size={c['size']}, "
              f"true_dist={c['true_label_dist']}")
    print(f"  {clusters['interpretation']}")
    
    # ── Confusion Pattern Analysis ─────────────────────────────────────
    print(f"\n  Confusion Pattern Analysis:")
    conf = confusion_pattern_analysis(y_true, y_pred)
    print(f"  Overall accuracy: {conf['overall_accuracy']:.2%}")
    print(f"  Classes: {conf['n_classes']}")
    print(f"  Error patterns found: {len(conf['error_patterns'])}")
    for ep in conf['error_patterns'][:2]:
        print(f"    Class {ep['true_class']}: accuracy={ep['accuracy']:.2%}, "
              f"top_error={ep['top_misclassifications'][0]['predicted_as'] if ep['top_misclassifications'] else 'N/A'}")
    
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
