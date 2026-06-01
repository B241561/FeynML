"""
Phase 1.5 — K-Means Clustering & PCA from Scratch
====================================================
Topics:
  • K-Means algorithm & K-Means++ initialization
  • Elbow method (inertia vs K)
  • Silhouette score
  • PCA via eigendecomposition
  • Explained variance ratio
  • Verification against scikit-learn
"""

import math
import random


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def euclidean(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

def mean_vec(vectors):
    d = len(vectors[0])
    n = len(vectors)
    return [sum(v[j] for v in vectors) / n for j in range(d)]

def mat_mul(A, B):
    rA, cA = len(A), len(A[0])
    rB, cB = len(B), len(B[0])
    C = [[0.0]*cB for _ in range(rA)]
    for i in range(rA):
        for j in range(cB):
            C[i][j] = sum(A[i][k]*B[k][j] for k in range(cA))
    return C

def mat_transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def mat_vec(A, v):
    return [sum(A[i][j]*v[j] for j in range(len(v))) for i in range(len(A))]

def dot(a, b):
    return sum(x*y for x,y in zip(a,b))

def vec_norm(v):
    return math.sqrt(sum(x**2 for x in v))

def vec_normalize(v):
    n = vec_norm(v)
    return [x/n for x in v] if n > 1e-12 else v

def vec_scale(v, s):
    return [x*s for x in v]

def vec_sub(a, b):
    return [x-y for x,y in zip(a,b)]

def vec_add(a, b):
    return [x+y for x,y in zip(a,b)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

class KMeans:
    """
    Lloyd's Algorithm:
      1. Initialize K centroids
      2. Assign each point to nearest centroid (E-step)
      3. Recompute centroids as cluster means (M-step)
      4. Repeat until convergence (centroids stop moving)
    
    Convergence guaranteed: inertia decreases monotonically.
    Not guaranteed to find global optimum → use multiple restarts.
    
    Time complexity: O(n * K * d * iterations)
    
    K-Means++ initialization (Arthur & Vassilvitskii, 2007):
      • First centroid: chosen uniformly at random
      • Each subsequent centroid: chosen with probability ∝ D(x)²
        where D(x) = distance from x to nearest existing centroid
      • Gives O(log K) approximation guarantee
      • Reduces number of iterations needed significantly
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 init='k-means++', n_init=10, random_state=42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.centroids_ = []
        self.labels_ = []
        self.inertia_ = float('inf')
        self.n_iter_ = 0

    def _init_centroids(self, X, rng):
        n = len(X)
        if self.init == 'k-means++':
            # K-Means++ seeding
            centroids = [X[rng.randint(0, n-1)]]
            for _ in range(self.k - 1):
                # Distance from each point to nearest centroid
                dists = [min(euclidean(x, c)**2 for c in centroids) for x in X]
                total = sum(dists)
                probs = [d / total for d in dists]
                # Weighted random selection
                r = rng.random()
                cumsum = 0.0
                chosen = 0
                for i, p in enumerate(probs):
                    cumsum += p
                    if r <= cumsum:
                        chosen = i
                        break
                centroids.append(X[chosen][:])
            return centroids
        else:
            # Random initialization
            return [X[i][:] for i in rng.sample(range(n), self.k)]

    def _assign_labels(self, X, centroids):
        return [min(range(self.k), key=lambda k: euclidean(x, centroids[k])) for x in X]

    def _compute_inertia(self, X, labels, centroids):
        return sum(euclidean(X[i], centroids[labels[i]])**2 for i in range(len(X)))

    def _run_once(self, X, rng):
        centroids = self._init_centroids(X, rng)
        labels = self._assign_labels(X, centroids)

        for it in range(self.max_iter):
            # Recompute centroids
            new_centroids = []
            for k in range(self.k):
                cluster = [X[i] for i in range(len(X)) if labels[i] == k]
                if cluster:
                    new_centroids.append(mean_vec(cluster))
                else:
                    new_centroids.append(centroids[k])  # keep old centroid

            # Check convergence
            shift = max(euclidean(centroids[k], new_centroids[k]) for k in range(self.k))
            centroids = new_centroids
            labels = self._assign_labels(X, centroids)

            if shift < self.tol:
                return centroids, labels, it + 1

        return centroids, labels, self.max_iter

    def fit(self, X):
        rng = random.Random(self.random_state)
        best_inertia = float('inf')

        for _ in range(self.n_init):
            centroids, labels, n_iter = self._run_once(X, rng)
            inertia = self._compute_inertia(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia = inertia
                self.centroids_ = centroids
                self.labels_ = labels
                self.inertia_ = inertia
                self.n_iter_ = n_iter

        return self

    def predict(self, X):
        return self._assign_labels(X, self.centroids_)

    def fit_predict(self, X):
        return self.fit(X).labels_

    @staticmethod
    def elbow_curve(X, k_range=range(1, 11), random_state=42):
        """Compute inertia for each K — look for the 'elbow'."""
        return {k: KMeans(n_clusters=k, random_state=random_state).fit(X).inertia_
                for k in k_range}


# ─────────────────────────────────────────────────────────────────────────────
# 2. SILHOUETTE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def silhouette_score(X, labels):
    """
    Silhouette coefficient for each sample:
      s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    a(i) = mean distance to other samples in same cluster  (cohesion)
    b(i) = min mean distance to samples in other clusters  (separation)
    
    Range: [-1, 1]
      +1 = sample is well inside its cluster
       0 = sample is on the boundary
      -1 = sample is in the wrong cluster
    
    Average silhouette = quality of clustering.
    """
    n = len(X)
    clusters = sorted(set(labels))
    if len(clusters) < 2:
        return 0.0

    sil = []
    for i in range(n):
        same = [j for j in range(n) if j != i and labels[j] == labels[i]]
        a_i = sum(euclidean(X[i], X[j]) for j in same) / max(len(same), 1)

        b_i = float('inf')
        for c in clusters:
            if c == labels[i]:
                continue
            other = [j for j in range(n) if labels[j] == c]
            if other:
                avg = sum(euclidean(X[i], X[j]) for j in other) / len(other)
                b_i = min(b_i, avg)

        denom = max(a_i, b_i)
        sil.append((b_i - a_i) / denom if denom > 0 else 0.0)

    return sum(sil) / n


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRINCIPAL COMPONENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def power_iteration(A, n_iter=500, seed=1):
    """Find dominant eigenvector via power iteration."""
    rng = random.Random(seed)
    d = len(A)
    v = [rng.gauss(0, 1) for _ in range(d)]
    v = vec_normalize(v)
    lam = 0.0
    for _ in range(n_iter):
        Av = mat_vec(A, v)
        lam_new = dot(v, Av)
        v_new = vec_normalize(Av)
        if abs(lam_new - lam) < 1e-9:
            lam = lam_new; v = v_new; break
        lam = lam_new; v = v_new
    return lam, v

def deflate(A, lam, v):
    """Hotelling's deflation: remove top eigenvector contribution."""
    d = len(A)
    vvT = [[v[i]*v[j] for j in range(d)] for i in range(d)]
    return [[A[i][j] - lam*vvT[i][j] for j in range(d)] for i in range(d)]

class PCA:
    """
    Principal Component Analysis via eigendecomposition of covariance matrix.
    
    Algorithm:
      1. Center data: X_c = X - mean(X)
      2. Compute covariance matrix: C = (1/(n-1)) * X_cᵀ X_c
      3. Find eigenvectors of C (principal components = axes of maximum variance)
      4. Project: Z = X_c @ W   (W = matrix of eigenvectors)
    
    Geometric intuition:
      • PCA finds the directions of maximum variance in the data
      • First PC = direction of most variance
      • Each subsequent PC = orthogonal to all previous, max remaining variance
      • Projecting onto top K PCs gives the best K-dim reconstruction (min MSE)
    
    Uses power iteration + deflation to extract eigenvectors one by one.
    For production: use SVD (more numerically stable).
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = []          # eigenvectors (PCs), shape: K × d
        self.explained_variance_ = []  # eigenvalues
        self.explained_variance_ratio_ = []
        self.mean_ = []

    def fit(self, X):
        n, d = len(X), len(X[0])

        # Center data
        self.mean_ = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        X_c = [[X[i][j] - self.mean_[j] for j in range(d)] for i in range(n)]

        # Covariance matrix: C = X_cᵀ X_c / (n-1)
        Xt = mat_transpose(X_c)
        XtX = mat_mul(Xt, X_c)
        C = [[XtX[i][j] / (n-1) for j in range(d)] for i in range(d)]

        # Extract top K eigenvectors via power iteration + deflation
        C_work = [row[:] for row in C]
        eigenvalues, eigenvectors = [], []
        for _ in range(self.n_components):
            lam, v = power_iteration(C_work, n_iter=1000, seed=_+1)
            eigenvalues.append(max(lam, 0.0))
            eigenvectors.append(v)
            C_work = deflate(C_work, lam, v)

        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues
        total_var = sum(max(0, C[i][i]) for i in range(d))
        self.explained_variance_ratio_ = [
            ev / total_var if total_var > 0 else 0.0
            for ev in eigenvalues
        ]
        return self

    def transform(self, X):
        """Project X onto principal components."""
        n, d = len(X), len(X[0])
        X_c = [[X[i][j] - self.mean_[j] for j in range(d)] for i in range(n)]
        return [[dot(X_c[i], self.components_[k]) for k in range(self.n_components)]
                for i in range(n)]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """Reconstruct approximate original data from projected coordinates."""
        result = []
        for z in Z:
            recon = [self.mean_[j] for j in range(len(self.mean_))]
            for k in range(self.n_components):
                recon = vec_add(recon, vec_scale(self.components_[k], z[k]))
            result.append(recon)
        return result

    def reconstruction_error(self, X):
        """Mean squared reconstruction error — measures info lost."""
        Z = self.transform(X)
        X_recon = self.inverse_transform(Z)
        n = len(X)
        return sum(
            sum((X[i][j] - X_recon[i][j])**2 for j in range(len(X[0])))
            for i in range(n)
        ) / n


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    from sklearn.cluster import KMeans as SkKM
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score as sk_sil
    import numpy as np

    print("=" * 60)
    print("Phase 1.5 — K-Means & PCA Verification")
    print("=" * 60)

    # --- K-Means ---
    X_np, y_true = make_blobs(n_samples=200, n_features=2, centers=3,
                               cluster_std=0.8, random_state=42)
    X = X_np.tolist()

    km = KMeans(n_clusters=3, random_state=42).fit(X)
    sk_km = SkKM(n_clusters=3, random_state=42, n_init=10).fit(X_np)

    # Compare inertias (should be similar)
    ok_inertia = abs(km.inertia_ - sk_km.inertia_) / (sk_km.inertia_ + 1e-9) < 0.05
    print(f"  KMeans inertia ours={km.inertia_:.2f}  sk={sk_km.inertia_:.2f}  [{'✓ PASS' if ok_inertia else '✗ FAIL'}]")

    # Silhouette
    our_sil = silhouette_score(X, km.labels_)
    sk_sil_score = sk_sil(X_np, np.array(km.labels_))
    ok_sil = abs(our_sil - sk_sil_score) < 0.05
    print(f"  Silhouette      ours={our_sil:.4f}  sk={sk_sil_score:.4f}  [{'✓ PASS' if ok_sil else '✗ FAIL'}]")

    # --- PCA ---
    from sklearn.datasets import make_classification
    X_cls, _ = make_classification(n_samples=200, n_features=6,
                                    n_informative=4, random_state=42)
    X_pca = X_cls.tolist()

    pca = PCA(n_components=3).fit(X_pca)
    sk_pca = SkPCA(n_components=3).fit(X_cls)

    evr_ours = pca.explained_variance_ratio_
    evr_sk   = sk_pca.explained_variance_ratio_.tolist()

    ok_evr = all(abs(evr_ours[i] - evr_sk[i]) < 0.05 for i in range(3))
    print(f"  PCA EVR ours={[round(e,3) for e in evr_ours]}  sk={[round(e,3) for e in evr_sk]}  [{'✓ PASS' if ok_evr else '✗ FAIL'}]")

    Z = pca.transform(X_pca)
    Z_sk = sk_pca.transform(X_cls)
    # Signs may differ — check magnitudes
    ok_proj = all(abs(abs(Z[i][0]) - abs(Z_sk[i][0])) < 0.5 for i in range(10))
    print(f"  PCA projections magnitude match (first PC)  [{'✓ PASS' if ok_proj else '✗ FAIL'}]")

    rec_err = pca.reconstruction_error(X_pca)
    print(f"  PCA reconstruction error (3 components): {rec_err:.4f}")
    print(f"  Cumulative explained variance: {sum(evr_ours):.4f}")

    print()
    print("  Key intuitions:")
    print("  • K-Means++ greatly reduces chance of bad initialization")
    print("  • Elbow method: pick K where inertia decrease slows down")
    print("  • Silhouette > 0.5 = reasonable clustering")
    print("  • PCA: first K components capture most variance with min dims")
    print("  • Check cumulative EVR ≥ 0.95 for good dimensionality reduction")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
