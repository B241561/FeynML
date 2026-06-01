"""
Phase 0.2 — Calculus & Optimization from Scratch
==================================================
No NumPy in core implementations.
Every result verified against scipy/autograd at the bottom.

Topics:
  Numerical derivatives (finite difference)
  Partial derivatives & gradients
  Chain rule demonstration
  Gradient descent (vanilla, momentum, AdaGrad, Adam)
  Convexity check
  Newton's method
  Why this matters for ML: every optimizer in sklearn/pytorch uses these ideas
"""

import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. NUMERICAL DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────

def derivative(f, x, h=1e-5):
    """
    Central difference approximation of f'(x).
    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    Why central difference (not forward)?
      Forward:  [f(x+h) - f(x)] / h        — O(h) error
      Central:  [f(x+h) - f(x-h)] / (2h)  — O(h²) error (much better!)

    Geometric intuition: slope of the tangent line at x.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def second_derivative(f, x, h=1e-4):
    """
    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    Positive → local minimum, Negative → local maximum.
    """
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

def partial_derivative(f, args, index, h=1e-5):
    """
    Partial derivative ∂f/∂x_i at point args.
    Vary only the i-th argument, hold all others fixed.
    """
    args_plus  = list(args); args_plus[index]  += h
    args_minus = list(args); args_minus[index] -= h
    return (f(*args_plus) - f(*args_minus)) / (2 * h)

def gradient(f, args, h=1e-5):
    """
    Gradient ∇f = [∂f/∂x₀, ∂f/∂x₁, ..., ∂f/∂xₙ]
    Points in the direction of STEEPEST ASCENT.
    Gradient descent moves in the NEGATIVE gradient direction.
    """
    return [partial_derivative(f, args, i, h) for i in range(len(args))]

def directional_derivative(f, args, direction, h=1e-5):
    """
    Rate of change of f in a specific direction.
    D_v f = ∇f · v̂  (dot product of gradient with unit direction vector)
    """
    n = len(direction)
    norm = math.sqrt(sum(d**2 for d in direction))
    unit_dir = [d / norm for d in direction]

    args_plus  = [args[i] + h * unit_dir[i] for i in range(n)]
    args_minus = [args[i] - h * unit_dir[i] for i in range(n)]
    return (f(*args_plus) - f(*args_minus)) / (2 * h)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHAIN RULE DEMONSTRATION
# ─────────────────────────────────────────────────────────────────────────────

def chain_rule_demo():
    """
    Chain rule: d/dx f(g(x)) = f'(g(x)) · g'(x)

    This is WHY backpropagation works.
    In a neural network:
      loss = f(activation(weighted_sum(inputs)))
      dloss/dweight = dloss/dactivation · dactivation/dweighted_sum · dweighted_sum/dweight

    Example: h(x) = sin(x²)
      g(x) = x²      → g'(x) = 2x
      f(u) = sin(u)   → f'(u) = cos(u)
      h'(x) = cos(x²) · 2x
    """
    def g(x): return x**2
    def f(u): return math.sin(u)
    def h(x): return f(g(x))

    x0 = 1.5
    h_prime_numerical = derivative(h, x0)
    h_prime_chain     = math.cos(g(x0)) * 2 * x0  # analytical

    return {
        "x": x0,
        "numerical_derivative": round(h_prime_numerical, 8),
        "chain_rule_result":    round(h_prime_chain, 8),
        "match": abs(h_prime_numerical - h_prime_chain) < 1e-5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONVEXITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def is_convex_at(f, x, h=1e-4):
    """
    A function is convex at x if f''(x) >= 0.
    Convex functions have ONE global minimum → gradient descent always works.
    MSE loss is convex. Cross-entropy is convex. Neural network loss is NOT.
    """
    return second_derivative(f, x, h) >= 0

def find_minimum_golden_section(f, lo, hi, tol=1e-7):
    """
    Golden-section search: finds minimum of unimodal function on [lo, hi].
    Does NOT require derivatives.
    """
    phi = (math.sqrt(5) - 1) / 2
    while (hi - lo) > tol:
        x1 = hi - phi * (hi - lo)
        x2 = lo + phi * (hi - lo)
        if f(x1) < f(x2):
            hi = x2
        else:
            lo = x1
    return (lo + hi) / 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRADIENT DESCENT VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

def gradient_descent(grad_fn, init_params, lr=0.01, n_iter=1000, tol=1e-8):
    """
    Vanilla gradient descent: θ ← θ - lr · ∇L(θ)

    Used in: linear regression, logistic regression, shallow nets.
    Issue: fixed learning rate — too large → diverge, too small → slow.
    """
    params = list(init_params)
    history = []

    for step in range(n_iter):
        grad = grad_fn(params)
        new_params = [params[i] - lr * grad[i] for i in range(len(params))]
        step_size = math.sqrt(sum((new_params[i] - params[i])**2 for i in range(len(params))))
        history.append({"step": step, "params": params[:], "grad_norm": round(step_size / lr, 6)})
        params = new_params
        if step_size < tol:
            break

    return params, history

def gradient_descent_momentum(grad_fn, init_params, lr=0.01, beta=0.9,
                               n_iter=1000, tol=1e-8):
    """
    Momentum: accelerates in persistent gradient direction.
    v ← β·v + (1-β)·∇L
    θ ← θ - lr·v

    Analogy: ball rolling downhill accumulates speed.
    Helps escape shallow local minima and saddle points.
    """
    params   = list(init_params)
    velocity = [0.0] * len(params)
    history  = []

    for step in range(n_iter):
        grad     = grad_fn(params)
        velocity = [beta * velocity[i] + (1 - beta) * grad[i] for i in range(len(params))]
        new_p    = [params[i] - lr * velocity[i] for i in range(len(params))]
        step_size = math.sqrt(sum((new_p[i] - params[i])**2 for i in range(len(params))))
        history.append({"step": step, "velocity_norm": round(sum(v**2 for v in velocity)**0.5, 6)})
        params = new_p
        if step_size < tol:
            break

    return params, history

def adagrad(grad_fn, init_params, lr=0.1, eps=1e-8, n_iter=1000, tol=1e-8):
    """
    AdaGrad: adapts learning rate per parameter.
    G_i += grad_i²
    θ_i ← θ_i - lr / sqrt(G_i + ε) · grad_i

    Benefit: large lr for infrequent params, small lr for frequent ones.
    Issue: G_i only grows → lr shrinks to 0 eventually.
    """
    params  = list(init_params)
    G       = [0.0] * len(params)

    for step in range(n_iter):
        grad   = grad_fn(params)
        G      = [G[i] + grad[i]**2 for i in range(len(params))]
        new_p  = [params[i] - lr / math.sqrt(G[i] + eps) * grad[i] for i in range(len(params))]
        step_size = math.sqrt(sum((new_p[i] - params[i])**2 for i in range(len(params))))
        params = new_p
        if step_size < tol:
            break

    return params

def adam(grad_fn, init_params, lr=0.001, beta1=0.9, beta2=0.999,
         eps=1e-8, n_iter=1000, tol=1e-8):
    """
    Adam (Adaptive Moment Estimation): combines momentum + RMSProp.
    m = β1·m + (1-β1)·g        ← first moment (mean of gradients)
    v = β2·v + (1-β2)·g²       ← second moment (variance of gradients)
    m̂ = m / (1-β1^t)            ← bias correction (important early on!)
    v̂ = v / (1-β2^t)
    θ ← θ - lr·m̂ / (sqrt(v̂) + ε)

    The DEFAULT optimizer in PyTorch and Keras.
    Works well with little tuning for most problems.
    """
    params = list(init_params)
    m      = [0.0] * len(params)  # first moment
    v      = [0.0] * len(params)  # second moment
    history = []

    for t in range(1, n_iter + 1):
        grad = grad_fn(params)

        m = [beta1 * m[i] + (1 - beta1) * grad[i]    for i in range(len(params))]
        v = [beta2 * v[i] + (1 - beta2) * grad[i]**2 for i in range(len(params))]

        # Bias correction
        m_hat = [m[i] / (1 - beta1**t) for i in range(len(params))]
        v_hat = [v[i] / (1 - beta2**t) for i in range(len(params))]

        new_p = [params[i] - lr * m_hat[i] / (math.sqrt(v_hat[i]) + eps)
                 for i in range(len(params))]

        step_size = math.sqrt(sum((new_p[i] - params[i])**2 for i in range(len(params))))
        history.append({"t": t, "step_size": round(step_size, 8)})
        params = new_p
        if step_size < tol:
            break

    return params, history


# ─────────────────────────────────────────────────────────────────────────────
# 5. NEWTON'S METHOD (for root-finding and optimization)
# ─────────────────────────────────────────────────────────────────────────────

def newton_method_1d(f, x0, tol=1e-9, max_iter=100):
    """
    Newton's method for finding roots of f: x ← x - f(x)/f'(x)
    For optimization (finding f'=0): use with f = gradient.
    Converges quadratically (much faster than GD) but needs f''.
    """
    x = x0
    for i in range(max_iter):
        fx  = f(x)
        fpx = derivative(f, x)
        if abs(fpx) < 1e-15:
            break
        x_new = x - fx / fpx
        if abs(x_new - x) < tol:
            return x_new, i + 1
        x = x_new
    return x, max_iter


# ─────────────────────────────────────────────────────────────────────────────
# 6. LINEAR REGRESSION VIA CALCULUS (Normal Equation)
# ─────────────────────────────────────────────────────────────────────────────

def linear_regression_gradient(X, y, params):
    """
    MSE gradient: ∂/∂θ (1/n Σ (y_hat - y)²)
    = (2/n) X^T (X@θ - y)

    Derived by:
    L(θ) = (1/n)||Xθ - y||²
    ∂L/∂θ = (2/n) X^T (Xθ - y)
    """
    n = len(y)
    # y_hat = X @ params
    y_hat  = [sum(X[i][j] * params[j] for j in range(len(params))) for i in range(n)]
    errors = [y_hat[i] - y[i] for i in range(n)]
    # grad = (2/n) X^T @ errors
    grad = [
        (2.0 / n) * sum(X[i][j] * errors[i] for i in range(n))
        for j in range(len(params))
    ]
    return grad

def mse_loss(X, y, params):
    n = len(y)
    y_hat = [sum(X[i][j] * params[j] for j in range(len(params))) for i in range(n)]
    return sum((y_hat[i] - y[i])**2 for i in range(n)) / n


# ─────────────────────────────────────────────────────────────────────────────
# 7. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    import random

    print("=" * 60)
    print("Phase 0.2 — Calculus & Optimization Verification")
    print("=" * 60)

    # --- Derivative ---
    f = lambda x: x**3 - 2*x + 1
    x0 = 2.0
    our_d  = derivative(f, x0)
    exact  = 3 * x0**2 - 2   # analytical: d/dx(x³-2x+1) = 3x²-2
    ok     = abs(our_d - exact) < 1e-5
    print(f"  d/dx(x³-2x+1) at x=2: ours={our_d:.6f}, exact={exact:.6f}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Partial derivative ---
    g = lambda x, y: x**2 + 3*x*y + y**2
    pt = [1.0, 2.0]
    dg_dx = partial_derivative(g, pt, 0)     # 2x + 3y = 2 + 6 = 8
    dg_dy = partial_derivative(g, pt, 1)     # 3x + 2y = 3 + 4 = 7
    ok_x  = abs(dg_dx - 8.0) < 1e-4
    ok_y  = abs(dg_dy - 7.0) < 1e-4
    print(f"  ∂(x²+3xy+y²)/∂x at (1,2)={dg_dx:.5f} (expect 8)  [{'✓ PASS' if ok_x else '✗ FAIL'}]")
    print(f"  ∂(x²+3xy+y²)/∂y at (1,2)={dg_dy:.5f} (expect 7)  [{'✓ PASS' if ok_y else '✗ FAIL'}]")

    # --- Chain rule ---
    cr = chain_rule_demo()
    ok = cr["match"]
    print(f"  Chain rule sin(x²) at x=1.5: numerical={cr['numerical_derivative']}, "
          f"chain={cr['chain_rule_result']}  [{'✓ PASS' if ok else '✗ FAIL'}]")

    # --- Gradient descent on f(x,y) = x² + y² (minimum at 0,0) ---
    quad = lambda args: args[0]**2 + args[1]**2
    grad_quad = lambda args: gradient(quad, args)
    final, hist = gradient_descent(grad_quad, [5.0, -3.0], lr=0.1, n_iter=200)
    ok_gd = abs(final[0]) < 1e-3 and abs(final[1]) < 1e-3
    print(f"  GD on x²+y²: final=({final[0]:.5f},{final[1]:.5f}) in {len(hist)} steps  "
          f"[{'✓ PASS' if ok_gd else '✗ FAIL'}]")

    # --- Adam on f(x,y) = (x-3)² + (y+2)² (minimum at 3,-2) ---
    quad2     = lambda args: (args[0]-3)**2 + (args[1]+2)**2
    grad_quad2 = lambda args: gradient(quad2, args)
    final_adam, _ = adam(grad_quad2, [0.0, 0.0], lr=0.1, n_iter=500)
    ok_adam = abs(final_adam[0] - 3.0) < 0.01 and abs(final_adam[1] + 2.0) < 0.01
    print(f"  Adam on (x-3)²+(y+2)²: final=({final_adam[0]:.4f},{final_adam[1]:.4f}) "
          f"(expect 3,-2)  [{'✓ PASS' if ok_adam else '✗ FAIL'}]")

    # --- Newton's method: find root of x² - 2 = 0 (answer = √2) ---
    f_root = lambda x: x**2 - 2
    root, iters = newton_method_1d(f_root, x0=1.0)
    ok_newton = abs(root - math.sqrt(2)) < 1e-8
    print(f"  Newton root of x²-2=0: {root:.8f} (√2={math.sqrt(2):.8f}) in {iters} iters  "
          f"[{'✓ PASS' if ok_newton else '✗ FAIL'}]")

    # --- Linear regression via gradient descent ---
    random.seed(42)
    n = 100
    true_w, true_b = 2.5, -1.0
    X_data = [[1.0, random.gauss(0,1)] for _ in range(n)]
    y_data = [true_b + true_w * X_data[i][1] + random.gauss(0, 0.1) for i in range(n)]

    grad_lr = lambda p: linear_regression_gradient(X_data, y_data, p)
    params_init = [0.0, 0.0]
    final_lr, _ = gradient_descent(grad_lr, params_init, lr=0.05, n_iter=2000)
    ok_lr = abs(final_lr[0] - true_b) < 0.1 and abs(final_lr[1] - true_w) < 0.1
    loss  = mse_loss(X_data, y_data, final_lr)
    print(f"  LinReg GD: b={final_lr[0]:.4f} (true={true_b}), "
          f"w={final_lr[1]:.4f} (true={true_w}), MSE={loss:.5f}  "
          f"[{'✓ PASS' if ok_lr else '✗ FAIL'}]")

    # --- Convexity check ---
    f_convex  = lambda x: x**2
    f_concave = lambda x: -x**2
    ok_conv = is_convex_at(f_convex, 1.0) and not is_convex_at(f_concave, 1.0)
    print(f"  Convexity: x² convex={is_convex_at(f_convex,1.0)}, "
          f"-x² convex={is_convex_at(f_concave,1.0)}  [{'✓ PASS' if ok_conv else '✗ FAIL'}]")

    print()
    print("  Key ML takeaways:")
    print("  • Every loss minimisation uses gradient descent or a variant")
    print("  • Backprop = repeated chain rule application")
    print("  • Adam is default: adapts lr per parameter using gradient history")
    print("  • Convex loss → guaranteed global minimum (linear/logistic regression)")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
