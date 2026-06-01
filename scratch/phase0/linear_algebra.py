"""
Phase 0.1 — Linear Algebra from Scratch
=========================================
No NumPy in core implementations.
Every function verified against NumPy at the bottom.

Topics:
  Vector operations (add, scale, dot, norm, cosine similarity)
  Matrix operations (add, multiply, transpose, determinant, inverse)
  Gaussian elimination (row reduction)
  Eigenvector power iteration demo
  Geometric intuition comments throughout
"""

import math
import copy


# ─────────────────────────────────────────────────────────────────────────────
# 1. VECTOR OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def vec_add(a, b):
    """Element-wise addition. Geometrically: tip-to-tail placement of arrows."""
    assert len(a) == len(b), "Vectors must have same dimension"
    return [a[i] + b[i] for i in range(len(a))]

def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def vec_scale(v, scalar):
    """Scaling: stretches or shrinks the vector. Negative scalar flips direction."""
    return [x * scalar for x in v]

def dot_product(a, b):
    """
    dot(a,b) = Σ a_i * b_i = |a||b|cos(θ)
    Geometric meaning: projection of a onto b (times |b|).
    dot = 0  → vectors are perpendicular (orthogonal)
    dot > 0  → angle < 90° (same general direction)
    dot < 0  → angle > 90° (opposite general direction)
    """
    assert len(a) == len(b)
    return sum(a[i] * b[i] for i in range(len(a)))

def vec_norm(v, p=2):
    """
    L2 norm (Euclidean length): sqrt(Σ v_i²)
    L1 norm (Manhattan):        Σ |v_i|
    """
    if p == 2:
        return math.sqrt(sum(x**2 for x in v))
    elif p == 1:
        return sum(abs(x) for x in v)
    else:
        return sum(abs(x)**p for x in v) ** (1/p)

def vec_normalize(v):
    """Unit vector in direction of v. ||v_hat|| = 1."""
    n = vec_norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return vec_scale(v, 1.0 / n)

def cosine_similarity(a, b):
    """
    cos(θ) = dot(a,b) / (|a| * |b|)
    ∈ [-1, 1]. Used as similarity measure in NLP, recommendation systems.
    1 = identical direction, 0 = orthogonal, -1 = opposite.
    """
    return dot_product(a, b) / (vec_norm(a) * vec_norm(b))

def outer_product(a, b):
    """Outer product: n×m matrix where M[i][j] = a[i] * b[j]."""
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MATRIX OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def mat_shape(A):
    return len(A), len(A[0])

def mat_zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def mat_identity(n):
    I = mat_zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def mat_add(A, B):
    r, c = mat_shape(A)
    return [[A[i][j] + B[i][j] for j in range(c)] for i in range(r)]

def mat_scale(A, scalar):
    return [[A[i][j] * scalar for j in range(len(A[0]))] for i in range(len(A))]

def mat_transpose(A):
    """
    Transpose: flip rows and columns.
    (A^T)[i][j] = A[j][i]
    Geometric: reflects the matrix across its main diagonal.
    """
    r, c = mat_shape(A)
    return [[A[i][j] for i in range(r)] for j in range(c)]

def mat_multiply(A, B):
    """
    Matrix multiplication: C[i][j] = dot(row_i of A, col_j of B)
    Geometric: composition of linear transformations.
    AB means: first apply B, then apply A.
    Time complexity: O(n³) for n×n matrices.
    """
    rA, cA = mat_shape(A)
    rB, cB = mat_shape(B)
    assert cA == rB, f"Shape mismatch: ({rA}×{cA}) @ ({rB}×{cB})"
    C = mat_zeros(rA, cB)
    for i in range(rA):
        for j in range(cB):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(cA))
    return C

def mat_vec_multiply(A, v):
    """Matrix-vector product: Av. Each row of A dotted with v."""
    r, c = mat_shape(A)
    assert c == len(v)
    return [sum(A[i][j] * v[j] for j in range(c)) for i in range(r)]

def mat_frobenius_norm(A):
    """sqrt(Σ A[i][j]²) — generalisation of vector L2 norm to matrices."""
    return math.sqrt(sum(A[i][j]**2 for i in range(len(A)) for j in range(len(A[0]))))


# ─────────────────────────────────────────────────────────────────────────────
# 3. GAUSSIAN ELIMINATION & INVERSE
# ─────────────────────────────────────────────────────────────────────────────

def _augment(A, b=None):
    """Create augmented matrix [A | b] or [A | I] for inversion."""
    n = len(A)
    if b is None:
        b = mat_identity(n)
        aug = [A[i][:] + b[i][:] for i in range(n)]
    else:
        aug = [A[i][:] + [b[i]] for i in range(n)]
    return aug

def gaussian_elimination(A, b):
    """
    Solve Ax = b via Gaussian elimination with partial pivoting.
    Returns solution vector x.
    
    Algorithm:
      1. Forward elimination: create upper-triangular form
      2. Back substitution: solve from bottom row upward
    """
    n = len(A)
    aug = _augment(A, b)

    for col in range(n):
        # Partial pivoting: swap row with largest absolute value in this column
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular (no unique solution)")

        # Eliminate below pivot
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x

def mat_inverse(A):
    """
    Matrix inverse via Gauss-Jordan elimination on [A | I].
    A * A^{-1} = I
    Only square matrices with non-zero determinant are invertible.
    """
    n = len(A)
    aug = [A[i][:] + mat_identity(n)[i][:] for i in range(n)]

    for col in range(n):
        # Partial pivoting
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular — inverse does not exist")

        # Normalize pivot row
        aug[col] = [x / pivot for x in aug[col]]

        # Eliminate ALL rows (not just below) — Gauss-Jordan
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for k in range(2 * n):
                aug[row][k] -= factor * aug[col][k]

    return [aug[i][n:] for i in range(n)]

def determinant(A):
    """
    Determinant via LU-style elimination.
    det = product of pivots (with sign flips for row swaps).
    |det| = volume of parallelepiped spanned by rows.
    det = 0 → matrix is singular (rows are linearly dependent).
    """
    n = len(A)
    M = copy.deepcopy(A)
    det = 1.0
    sign = 1

    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if max_row != col:
            M[col], M[max_row] = M[max_row], M[col]
            sign *= -1

        if abs(M[col][col]) < 1e-12:
            return 0.0

        det *= M[col][col]
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for k in range(col, n):
                M[row][k] -= factor * M[col][k]

    return sign * det


# ─────────────────────────────────────────────────────────────────────────────
# 4. EIGENVALUES & EIGENVECTORS (Power Iteration)
# ─────────────────────────────────────────────────────────────────────────────

def power_iteration(A, n_iter=1000, tol=1e-9, seed=42):
    """
    Power iteration: finds the DOMINANT eigenvector (largest |eigenvalue|).
    
    Algorithm:
      1. Start with random vector b
      2. Repeatedly multiply: b = A @ b / ||A @ b||
      3. Converges to the eigenvector for the largest eigenvalue
    
    Geometric intuition:
      Eigenvectors are the "axes" of a linear transformation — directions that
      only get scaled (not rotated) when A is applied.
      Av = λv  where λ is the eigenvalue (scaling factor).
    """
    import random
    rng = random.Random(seed)
    n = len(A)
    b = [rng.gauss(0, 1) for _ in range(n)]
    b = vec_normalize(b)

    eigenvalue = 0.0
    for _ in range(n_iter):
        Ab = mat_vec_multiply(A, b)
        eigenvalue_new = dot_product(b, Ab)
        b_new = vec_normalize(Ab)

        if abs(eigenvalue_new - eigenvalue) < tol:
            eigenvalue = eigenvalue_new
            b = b_new
            break
        eigenvalue = eigenvalue_new
        b = b_new

    return eigenvalue, b


# ─────────────────────────────────────────────────────────────────────────────
# 5. COVARIANCE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def covariance_matrix(X):
    """
    Compute n_features × n_features covariance matrix from data matrix X
    (shape: n_samples × n_features).
    
    Cov[i][j] = E[(Xi - μi)(Xj - μj)]
    Diagonal: variances of each feature.
    Off-diagonal: covariances (how features co-vary).
    """
    n = len(X)
    d = len(X[0])
    means = [sum(X[row][j] for row in range(n)) / n for j in range(d)]

    cov = mat_zeros(d, d)
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum(
                (X[row][i] - means[i]) * (X[row][j] - means[j])
                for row in range(n)
            ) / (n - 1)
    return cov, means


# ─────────────────────────────────────────────────────────────────────────────
# 6. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    import numpy as np

    print("=" * 60)
    print("Phase 0.1 — Linear Algebra Verification")
    print("=" * 60)

    # --- Dot product ---
    a, b = [1, 2, 3], [4, 5, 6]
    our_dot = dot_product(a, b)
    np_dot  = float(np.dot(a, b))
    ok = abs(our_dot - np_dot) < 1e-9
    print(f"  dot([1,2,3],[4,5,6]) = {our_dot}  (numpy={np_dot})  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Matrix multiply ---
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    our_C = mat_multiply(A, B)
    np_C  = np.matmul(A, B).tolist()
    ok = all(abs(our_C[i][j] - np_C[i][j]) < 1e-9 for i in range(2) for j in range(2))
    print(f"  matmul([[1,2],[3,4]], [[5,6],[7,8]]) = {our_C}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Transpose ---
    M = [[1, 2, 3], [4, 5, 6]]
    our_T = mat_transpose(M)
    np_T  = np.array(M).T.tolist()
    ok = our_T == np_T
    print(f"  transpose([[1,2,3],[4,5,6]]) = {our_T}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Inverse ---
    A3 = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    our_inv = mat_inverse(A3)
    np_inv  = np.linalg.inv(A3).tolist()
    ok = all(abs(our_inv[i][j] - np_inv[i][j]) < 1e-6 for i in range(3) for j in range(3))
    print(f"  inverse(A) matches numpy  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Verify A @ A^{-1} ≈ I ---
    AI = mat_multiply(A3, our_inv)
    I  = mat_identity(3)
    ok_id = all(abs(AI[i][j] - I[i][j]) < 1e-6 for i in range(3) for j in range(3))
    print(f"  A @ A_inv ≈ I  [{'✓ PASS' if ok_id else '✗ FAIL'}]")

    # --- Determinant ---
    our_det = determinant(A3)
    np_det  = float(np.linalg.det(A3))
    ok = abs(our_det - np_det) < 1e-6
    print(f"  det(A) = {our_det:.6f}  (numpy={np_det:.6f})  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Solve Ax = b ---
    b_vec = [1.0, 2.0, 3.0]
    our_x = gaussian_elimination(A3, b_vec)
    np_x  = np.linalg.solve(A3, b_vec).tolist()
    ok = all(abs(our_x[i] - np_x[i]) < 1e-6 for i in range(3))
    print(f"  solve Ax=b: ours={[round(x,4) for x in our_x]}  numpy={[round(x,4) for x in np_x]}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Eigenvalue (power iteration) ---
    A_sym = [[4, 1], [1, 3]]   # symmetric → real eigenvalues
    our_lam, our_v = power_iteration(A_sym)
    np_lam, np_v   = np.linalg.eig(A_sym)
    max_lam = float(np_lam[np.argmax(np.abs(np_lam))])
    ok = abs(our_lam - max_lam) < 1e-4
    print(f"  dominant eigenvalue: ours={our_lam:.4f}  numpy_max={max_lam:.4f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Cosine similarity ---
    x = [1, 0, 0]
    y = [1, 1, 0]
    our_cos = cosine_similarity(x, y)
    np_cos  = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    ok = abs(our_cos - np_cos) < 1e-9
    print(f"  cosine_sim([1,0,0],[1,1,0]) = {our_cos:.4f}  (expected {np_cos:.4f})  [{'✓ PASS' if ok else '✗ FAIL'}]")

    print()
    print("  Key geometric intuitions:")
    print("  • dot(a,b)=0 → perpendicular vectors → uncorrelated features")
    print("  • det=0 → singular matrix → features are linearly dependent")
    print("  • eigenvector = direction unchanged by transformation")
    print("  • covariance matrix diagonal = feature variances")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
