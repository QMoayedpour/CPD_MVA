import numpy as np
import matplotlib.pyplot as plt
from src.cost import cost_L2


class PELT(object):
    def __init__(self, x, cost_function=cost_L2, max_changepoints=5):
        """
        Paper: https://arxiv.org/pdf/1101.1438
        """
        self.x = x
        self.cost_function = cost_function
        self.max_changepoints = max_changepoints
        self.T = None

    def compute_cost_grid(self):
        """
        Compute the cost grid for the time series.
        """
        n = len(self.x)
        cost_grid = np.zeros((n, n))
        for t in range(n):
            for s in range(t, n):
                cost_grid[t, s] = self.cost_function(self.x[t:(s + 1)])
        return cost_grid

    def dp_first_cp(self, cost_values, opt_values):
        """
        Compute the first change point and its cost.
        """
        n = len(cost_values)
        k = n - len(opt_values) + 1

        cost_array = np.full(n - k, np.inf)
        for s in range(n - k):
            cost_array[s] = cost_values[s] + opt_values[s + 1]
        t = np.argmin(cost_array)
        v = cost_array[t]

        return t, v

    def dp_all_cp(self):
        """
        Compute all change points up to the maximum number.
        """
        n = len(self.x)
        K = self.max_changepoints
        T = np.empty(K, dtype=object)

        cost_grid = self.compute_cost_grid()
        V = cost_grid[:, -1]

        for k in range(1, K):
            T[k - 1] = np.empty(n - k, dtype=int)
            for t in range(n - k):
                cost_values = cost_grid[t, t:]
                opt_values = V[t:]
                T[k - 1][t], V[t] = self.dp_first_cp(cost_values, opt_values)
                T[k - 1][t] += t
            V = V[:n - k]

        T[-1], _ = self.dp_first_cp(cost_grid[0], V)
        self.T = T

    def dp_cpd(self, k):
        """
        Retrieve the k optimal change points.
        """

        K = len(self.T)
        CP = np.empty(k, dtype=int)
        if k == K:
            CP[0] = self.T[-1]
        else:
            CP[0] = self.T[k - 1][0]

        for i in range(1, k):
            CP[i] = self.T[k - i - 1][CP[i - 1] + 1]

        return CP

    def run(self):
        """
        Run the change point detection algorithm.
        """
        self.dp_all_cp()

    def show_changepoints(self, num_changepoints, how=""):
        """
        Visualize the detected change points with segment means.
        """

        CP = self.dp_cpd(num_changepoints)

        CP = np.sort(np.concatenate(([0], CP, [len(self.x)])))

        if how == "linear":
            segment_means = []
            for i in range(len(CP) - 1):
                start, end = CP[i], CP[i + 1]
                t = np.arange(len(self.x[start:end])).reshape(-1, 1)
                model = LinearRegression()
                model.fit(t, self.x[start:end])

                segment_means.append(model.predict(t))
           
        else:
            segment_means = []
            for i in range(len(CP) - 1):
                start, end = CP[i], CP[i + 1]
                segment_means.append(np.tile(np.mean(self.x[start:end]), end-start))
        segments = np.concatenate(segment_means)

        plt.figure(figsize=(16, 8))
        plt.plot(np.arange(len(self.x)), self.x)

        plt.scatter(CP[1:-1], segments[CP[1:-1]], label=f"{num_changepoints} Change Points", c="r")
        if how=="linear":
            plt.plot(np.arange(len(self.x)), segments, c="red")
        else:
            plt.step(np.arange(len(self.x)), segments, where="post", c="red")
        plt.title("Change Point Detection with Segment Means")
        plt.legend()
        plt.show()
