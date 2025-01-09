import numpy as np
import random


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


def generate_processus(T=300, k=5, sigma=5, scale=2, mean=-0.5):
    taus = np.sort(random.sample(range(1, T), k))
    taus = np.insert(taus, 0, 0)
    taus = np.append(taus, T)

    coefs = (np.random.uniform(size=k+1)+mean)*scale
    all_segments = []
    prev_value = 0
    for i, tau in enumerate(taus[:-1]):

        next_value = prev_value + coefs[i] * (taus[i+1]-taus[i])
        all_segments.append(np.linspace(prev_value, next_value, taus[i+1]-taus[i]))
        prev_value = next_value
    
    return np.concatenate(all_segments) + np.random.randn(T)*sigma

