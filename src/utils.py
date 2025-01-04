import numpy as np


def ssa_approx(x, m=1, k=32):
    """
    Compute SSA Approximation as saw during class
    """
    n = len(x)

    X = np.array([x[i:i + k] for i in range(n - k + 1)])

    U, s, Vt = np.linalg.svd(X)

    U_m = U[:, :m]
    s_m = s[:m]
    Vt_m = Vt[:m, :]

    S_m = np.diag(s_m)

    X_approx = np.dot(U_m, np.dot(S_m, Vt_m))

    return np.concatenate([X_approx[:, 0], X_approx[-1, 1:]])
