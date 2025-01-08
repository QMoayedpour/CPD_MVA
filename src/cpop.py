import numpy as np
import math
import ast
from tqdm import tqdm
from scipy.stats import median_abs_deviation
from src.cost import LogCost
import matplotlib.pyplot as plt


class CPOP(object):
    def __init__(self, y, sigma=None, beta=1, h=LogCost(1), idxs=None):
        self.y = y
        if sigma is None:
            self.sigma = median_abs_deviation(np.diff(y))
            # Si sigma n'est pas encore calculé, on prends la MAD
            # de l'incrément de la série
        else:
            self.sigma = sigma

        self.beta = beta
        self.h = h
        self.n = len(y)
        self.taus_t = [[0]]
        self.coefs = {}
        self.coefs_t = {}
        self.recompute_sigma = False
        self.idxs = idxs
    
    def update_sigma(self):
        self.sigma = np.std(self.y - self.approx)

    def reset_sigma(self):
        self.sigma = median_abs_deviation(np.diff(self.y))

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

        rec_1 = -E /(2*F)
        rec_2 = -B /(2*F)
        a = D + self.h(t-tauk) + self.beta - (E**2)/(4*F)
        b = C - 2*(E*B)/(4*F)
        c = A - (B**2)/(4*F)

        return np.array([a, b, c]), rec_1, rec_2

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

        rec_1 = -(E + past_coefs[1]) / (2*(F + past_coefs[2]))
        rec_2 = -B / (2*(F + past_coefs[2]))

        a = past_coefs[0] + D + self.beta + self.h(s) - ((past_coefs[1]+E)**2)/(4*(past_coefs[2] + F))  
        b = C - B*(past_coefs[1]+E) / (2 * (past_coefs[2] + F))
        c = A - (B**2)/(4*(past_coefs[2]+F))

        return np.array([a, b, c]), rec_1, rec_2

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
                rootp = (-coefs[1] + np.sqrt(delta)) / (2 * coefs[2])
                rootm = (-coefs[1] - np.sqrt(delta)) / (2 * coefs[2])
        else:
            if coefs[1] != 0:
                rootp = -coefs[0] / coefs[1]
                rootm = -coefs[0] / coefs[1]
            else:
                if coefs[0] >= 0:
                    remove.append(idx)
                    x.append(np.inf)
                    return (x[0], remove)

        if rootp > phi_curr + eps and rootm > phi_curr + eps:
            x.append(min(rootp, rootm))
        elif rootp > phi_curr + eps:
            x.append(rootp)
        elif rootm > phi_curr + eps:
            x.append(rootm)
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
                x[tau], remove = self.get_mean_diff(phi_curr, coefs[tau],
                                                    coefs[key_curr], tau, remove)

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
            if coefs[2] == 0: # Pas de div par 0
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
        coef, rec_1, rec_2 = self.get_coefs_at_null(ckpts[1]+1)

        list_rec_1s = [rec_1]
        list_rec_2s = [rec_2]
        list_phi = []

        for i, ckpt in enumerate(ckpts[2:]):

            coef, rec_1, rec_2 = self.get_min_C(ckpts[i+1]+1, ckpt + 1, coef) 

            list_rec_1s.append(rec_1)
            list_rec_2s.append(rec_2)

        phi = -coef[1] / (2 * coef[2])
        list_phi.append(phi)
        for rec_1, rec_2 in zip(list_rec_1s[::-1], list_rec_2s[::-1]):
            list_phi.append(rec_1 + rec_2 * list_phi[-1])

        return list_phi[::-1]

    def approx_f(self, ckpts, phis):
        phis[0] += 1
        phi1 = np.array([phis[0]])
        approx = [phi1]
        for i in range(1, len(ckpts)):
            slope = (phis[i] - phis[i - 1]) / (ckpts[i] - ckpts[i - 1])
            x = np.arange(ckpts[i - 1] + 1, ckpts[i] + 1)
            y = slope * (x - ckpts[i - 1]) + phis[i - 1]
            approx.append(y)

        return np.concatenate(approx)

    def compute_approx_and_plot(self, ckpts=None, logs=False, verbose=True,
                                stride=5, title="", test=False, noticks=False):
        """Compute the phis, the approximation of y (given phis) and plot it (optionnal)
        """
        if ckpts is None:
            ckpts = self.ckpts
        if ckpts[0] != 0:
            ckpts.insert(0, 0)
        if ckpts[-1] != len(self.y)-1:
            ckpts.insert(-1, len(self.y)-1)

        self.phis = self.get_phis(ckpts)

        self.approx = self.approx_f(ckpts, self.phis)
        if verbose:
            if self.idxs is None:
                idxs = np.arange(0, len(self.y))
            else:
                idxs = self.idxs
            plt.figure(figsize=(12, 6))
            plt.plot(idxs, self.y, c="blue")
            plt.plot(idxs, self.approx, c="r", label="Approximation")
            plt.title(title)
            plt.scatter(idxs[ckpts[1:-1]], self.approx[ckpts[1:-1]], c="r")

            ax = plt.gca()
            tick_positions = range(0, len(idxs), stride)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([idxs[i] for i in tick_positions], rotation=45)

            if noticks:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.legend()
            if test:
                plt.show()

        if logs:
            return self.approx

    def _loglikelihood(self):

        log_likelihood = 0

        for t in range(self.n):

            log_likelihood += -0.5 * (np.log(2 * np.pi * self.sigma**2) + ((self.y[t] - self.approx[t]) ** 2) / self.sigma**2)

        return log_likelihood

    def BIC(self):
        return -2*self._loglikelihood() + len(self.phis) * np.log(self.n)

    def mBIC(self):
        scnd_term = sum(math.log((self.ckpts[i] - self.ckpts[i-1]) / self.n) for i in range(1, len(self.ckpts)))
        return -2*self._loglikelihood() + 3 * len(self.phis) * np.log(self.n) + scnd_term

    def AIC(self):
        return -2*self._loglikelihood() + 2*len(self.phis)

    def criterion(self, criterion="BIC"):
        if criterion == "BIC":
            return self.BIC()
        elif criterion == "mBIC":
            return self.mBIC()
        elif criterion == "AIC":
            return self.AIC()
        else:
            return None

    def compute_max_criterion(self, beta_range=np.linspace(0.5, 20, 39), criterion="BIC",
                              verbose=True, log_n=False, upd_sigma=False, reset_sigma=True,
                              noticks=False):
        """
        For beta in beta_range, we estimate the model and select the one that minimise the criterion
        log_n is optional and multiply the values of beta by log_n. In the original article
        they show that we have asymptotic results for beta = rec_2 * log(n)
        """
        criterion_value = []
        self.list_logs = []
        if log_n:
            beta_range *= np.log(self.n)
        for i in tqdm(beta_range):

            self.beta = i

            self._reset_coefs()
            _ = self.run()

            self.compute_approx_and_plot(verbose=False)

            if upd_sigma:
                self.update_sigma()    

            criterion_value.append(self.criterion(criterion))
            self.list_logs.append((float(self.sigma), float(self.beta),
                                    float(self.criterion(criterion))))

            if reset_sigma:
                self.reset_sigma()

        idx_min = criterion_value.index(min(criterion_value))
        self.beta = beta_range[idx_min]
        self._reset_coefs()
        self.sigma = self.list_logs[idx_min][0]
        self.run()
        print(f"Beta for min {criterion}:", self.beta)
        print(f"{criterion}:", self.criterion(criterion))
        self.compute_approx_and_plot(verbose=verbose, noticks=noticks)
