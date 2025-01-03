import numpy as np
import ast
from tqdm import tqdm
from scipy.stats import median_abs_deviation
from .cost import LogCost
import matplotlib.pyplot as plt


class CPOP(object):
    def __init__(self, y, sigma=None, beta=1, h=LogCost(1)):
        self.y = y
        if sigma is None:
            self.sigma = median_abs_deviation(np.diff(y))
        else:
            self.sigma = sigma

        self.beta = beta
        self.h = h
        self.n = len(y)
        self.taus_t = [[0]]
        self.coefs = {}
        self.coefs_t = {}

    def _reset_coefs(self):
        self.taus_t = [[0]]
        self.coefs = {}
        self.coefs_t = {}

    def get_coefs(self, tauk, t):
        """Compute the coefficients as described in the paper."""
        s = t - tauk
        A = (s + 1) * (2 * s + 1) / (6 * s * self.sigma**2)
        B = (s + 1) / self.sigma**2 - (s + 1) * (2 * s + 1) / (3 * s * self.sigma**2)

        sum_y_linear = np.sum(self.y[tauk:t] * np.arange(1, s + 1))
        sum_y = np.sum(self.y[tauk:t])
        sum_y_squared = np.sum(self.y[tauk:t]**2)

        C = -2 / (s * self.sigma**2) * (sum_y_linear)
        D = sum_y_squared / self.sigma**2
        E = -C - 2 * sum_y / self.sigma**2
        F = (s - 1) * (2 * s - 1) / (6 * s * self.sigma**2)

        return A, B, float(C), float(D), float(E), F

    def get_coefs_at_null(self, t, tauk=0):
        A, B, C, D, E, F = self.get_coefs(tauk, t)

        if F == 0:  # pas de division par 0
            return np.array([self.y[0] ** 2]*3),  -2 * self.y[0], 1
        else:
            alpha = -E / 2 / F
            gamma = -B / 2 / F
            a = D + alpha * E + F * alpha**2 + self.h(t)
            b = alpha * B + C + E * gamma + 2 * F * alpha * gamma
            c = A + B * gamma + F * gamma**2

            return np.array([a, b, c]), alpha, gamma

    def get_min_inf(self, coefs):
        min_col2 = min(array[2] for array in coefs.values())
        candidates = {key: array for key, array in coefs.items() if array[2] == min_col2}

        if len(candidates) > 1:
            max_col1 = max(array[1] for array in candidates.values())
            candidates = {key: array for key, array in candidates.items() if array[1] == max_col1}

        if len(candidates) > 1:
            min_col0 = min(array[0] for array in candidates.values())
            candidates = {key: array for key, array in candidates.items() if array[0] == min_col0}

        return next(iter(candidates.keys()))

    def get_min_C(self, tauk, t, past_coefs):
        s = t - tauk
        A, B, C, D, E, F = self.get_coefs(tauk, t)

        alpha = -(E + past_coefs[1]) / (2*(F + past_coefs[2]))
        gamma = -B / (2*(F + past_coefs[2]))

        a = (
            past_coefs[0]
            + past_coefs[1] * alpha
            + past_coefs[2] * alpha**2
            + D
            + E * alpha
            + F * alpha**2
            + self.h(s)
            + self.beta
        )
        b = (
            past_coefs[1] * gamma
            + 2 * past_coefs[2] * alpha * gamma
            + B * alpha
            + C
            + E * gamma
            + 2 * F * alpha * gamma
        )
        c = past_coefs[2] * gamma**2 + A + B * gamma + F * gamma**2

        return np.array([a, b, c]), alpha, gamma

    def get_mean_diff(self, phi_curr, coefs_1, coefs_2, idx, remove=[], eps=1e-5):
        x = []
        coefs = coefs_1 - coefs_2

        if coefs[2] != 0:
            delta = coefs[1] ** 2 - 4 * coefs[2] * coefs[0]
            if delta < 0:
                remove.append(idx)
                x.append(np.inf)
                return (x[0], remove)
            else:
                root1 = (-coefs[1] + np.sqrt(delta)) / 2 / coefs[2]
                root2 = (-coefs[1] - np.sqrt(delta)) / 2 / coefs[2]
        else:
            if coefs[1] != 0:
                root1 = -coefs[0] / coefs[1]
                root2 = -coefs[0] / coefs[1]
            else:
                if coefs[0] >= 0:
                    remove.append(idx)
                    x.append(np.inf)
                    return (x[0], remove)

        if root1 > phi_curr + eps and root2 > phi_curr + eps:
            x.append(min(root1, root2))
        elif root1 > phi_curr + eps:
            x.append(root1)
        elif root2 > phi_curr + eps:
            x.append(root2)
        else:
            remove.append(idx)
            x.append(np.inf)

        return (x[0], remove)

    def get_int_t(self, coefs, taus_t):
        phi_curr = -np.inf
        key_curr = self.get_min_inf(coefs)
        taus_t = [str(tau) for tau in taus_t]
        remove = []
        intervals = {tau: [] for tau in taus_t}

        while len(set(taus_t) - set(remove)) > 1:
            x = {}
            for tau in taus_t:
                if tau in remove:
                    continue
                if tau == key_curr:
                    x[tau] = np.inf
                    continue
                x[tau], remove = self.get_mean_diff(phi_curr, coefs[tau], coefs[key_curr], tau, remove)

            if not x:
                break

            tau_min = min(x, key=x.get)
            if tau_min == np.inf:
                taus_t = []
            phi_new = x[tau_min]

            intervals[key_curr].append((phi_curr, phi_new))

            key_curr = tau_min
            phi_curr = phi_new

        if not intervals[key_curr]:
            intervals[key_curr].append((phi_curr, np.inf))

        for tau in intervals:
            if not intervals[tau]:
                intervals[tau] = None

        return intervals

    def update_T_hat(self, coefs, taus_t, t):
        intervals = self.get_int_t(coefs, taus_t)

        T_t_star = [ast.literal_eval(tau) for tau in intervals if intervals[tau] is not None]

        T_t_prune = self.ineq_prun(self.coefs)

        taus_t_ = T_t_star.copy()

        for tau in T_t_star:
            if tau not in T_t_prune: # Ici T_t_prune renvoie les partitions qui ne vérifient pas
                                     # L'autre cond de pruning
                                     # Peut etre #TODO modifier comment on fait l'intersection des 2 sets
                taus_t_.append(tau + [t])

        return taus_t_

    def get_val(self, coefs_dict):
        output_dict = {}

        for key, coefs in coefs_dict.items():
            if coefs[2] == 0:
                output_dict[key] = coefs[0]
            else:
                output_dict[key] = float(coefs[0] - coefs[1] ** 2 / 4 / coefs[2])

        return output_dict

    def ineq_prun(self, coefs):

        bound_dict = self.get_val(coefs)

        bounds_values = np.array(list(bound_dict.values()))
        min_bound = bounds_values.min()

        taus_out = []
        for key, bound in bound_dict.items():
            if bound > min_bound + self.K:
                taus_out.append(ast.literal_eval(key))
        return taus_out

    def run(self):

        self.K = 2 * self.beta + self.h(1) + self.h(self.n)

        for t in range(1, self.n+1):
            self.coefs_t[t] = {}
            for i, tau in enumerate(self.taus_t):
                if len(tau) == 1 and tau[0] == 0:
                    self.coefs[f"{tau}"], _, _ = self.get_coefs_at_null(t)
                    self.coefs_t[t][f"{tau}"] = self.coefs[f"{tau}"]
                    continue

                A, B, C, D, E, F = self.get_coefs(tau[-1], t)
                self.coefs[f"{tau}"], _, __ = self.get_min_C(tau[-1], t,
                                                             self.coefs_t[tau[-1]][f"{tau[:-1]}"])
                self.coefs_t[t][f"{tau}"] = self.coefs[f"{tau}"]

            self.taus_t = self.update_T_hat(self.coefs, self.taus_t, t)
            self.coefs = {tau: self.coefs[f"{tau}"] for tau in self.coefs if ast.literal_eval(tau) in self.taus_t}

        res = self.get_val(self.coefs)
        ckpts = ast.literal_eval(min(res, key=res.get))
        self.ckpts = [x-1 if x!=0 else 0 for x in ckpts]
        self.ckpts += [len(self.y)-1]
        return self.ckpts

    def get_phis(self, ckpts):
        coef, alpha, gamma = self.get_coefs_at_null(ckpts[1]+1)

        list_alphas = [alpha]
        list_gammas = [gamma]
        list_phi = []

        for i, ckpt in enumerate(ckpts[2:]):

            coef, alpha, gamma = self.get_min_C(ckpts[i+1]+1, ckpt + 1, coef) 
            
            list_alphas.append(alpha)
            list_gammas.append(gamma)
        
        phi = -coef[1] / (2 * coef[2])
        list_phi.append(phi)
        for alpha, gamma in zip(list_alphas[::-1], list_gammas[::-1]):
            list_phi.append(alpha + gamma * list_phi[-1])

        return list_phi[::-1]

    def approx_f(self, ckpts, phis):
        phi1 = np.array([phis[0]])
        approx = [phi1]
        for i in range(1, len(ckpts)):
            slope = (phis[i] - phis[i - 1]) / (ckpts[i] - ckpts[i - 1])
            x = np.arange(ckpts[i - 1] + 1, ckpts[i] + 1)
            y = slope * (x - ckpts[i - 1]) + phis[i - 1]
            approx.append(y)

        return np.concatenate(approx)

    def compute_approx_and_plot(self, ckpts=None, logs=False, verbose=True):
        if ckpts is None:
            ckpts = self.ckpts
        if ckpts[0] != 0:
            ckpts.insert(0, 0)
        if ckpts[-1] != len(self.y)-1:
            ckpts.insert(-1, len(self.y)-1)

        self.phis = self.get_phis(ckpts)

        self.approx = self.approx_f(ckpts, self.phis)
        if verbose:
            plt.figure(figsize=(12, 6))
            plt.plot(self.y, c="blue")
            plt.plot(self.approx, c="r", label="Approximation")
            plt.scatter(ckpts[1:-1], self.approx[ckpts[1:-1]], c="r")
            plt.show()
        if logs:
            return self.approx
    
    def _loglikelihood(self):
        n = len(self.y)
        log_likelihood = 0

        for t in range(self.n):

            log_likelihood += -0.5 * (np.log(2 * np.pi * self.sigma**2) + ((self.y[t] - self.approx[t]) ** 2) / self.sigma**2)

        return log_likelihood
    
    def BIC(self):
        return -2*self._loglikelihood() + len(self.phis) * np.log(self.n)
    
    def compute_max_bic(self, beta_range=np.linspace(0.5, 20, 39), verbose=True):

        bic_values = []
        for i in tqdm(beta_range):

            self.beta = i

            self._reset_coefs()
            _ = self.run()
            self.compute_approx_and_plot(verbose=False)

            bic_values.append(self.BIC())
        
        idx_min = bic_values.index(min(bic_values))
        self.beta = beta_range[idx_min]
        self._reset_coefs()
        self.run()
        print("Beta for min BIC:", self.beta)
        print("BIC:", bic_values[idx_min])
        if verbose:
            self.compute_approx_and_plot(verbose=True)
