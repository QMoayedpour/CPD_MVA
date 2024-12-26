import numpy as np


def get_coefs(y, sigma, tauk, t):
    """Compute the coefficients as described in the paper,
    """
    s = t - tauk

    A = (s + 1) * (2 * s + 1) / (6 * s * sigma**2)
    B = (s + 1) / sigma**2 - (s + 1) * (2 * s + 1) / (3 * s * sigma**2)

    sum_y_linear = np.sum(y[tauk:t] * np.arange(1, s + 1))
    sum_y = np.sum(y[tauk:t])
    sum_y_squared = np.sum(y[tauk:t]**2)

    C = -2 / (s * sigma**2) * (sum_y_linear)
    D = sum_y_squared / sigma**2
    E = -C - 2 * sum_y / sigma**2
    F = (s - 1) * (2 * s - 1) / (6 * s * sigma**2)

    return A, B, float(C), float(D), float(E), F
