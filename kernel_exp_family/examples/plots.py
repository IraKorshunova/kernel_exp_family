import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from kernel_exp_family.estimators.full.gaussian import KernelExpFullGaussian
from kernel_exp_family.kernels.kernels import gaussian_kernel_grad, gaussian_kernel_dx_dx, gaussian_kernel
from kernel_exp_family.estimators.estimator_oop import EstimatorBase


class Gaussian_KDE(EstimatorBase):
    def __init__(self, bandwidth, D):
        super(Gaussian_KDE, self).__init__(D)
        self.bandwidth = bandwidth

    def fit(self, X, y=None):
        self.X = X

    def log_pdf(self, x):
        return np.log(np.sum(gaussian_kernel(x, self.X, sigma=self.bandwidth), axis=-1) / (
            self.X.shape[0] * self.bandwidth))

    def grad(self, x):
        g = np.sum(gaussian_kernel_grad(x, self.X, sigma=self.bandwidth), axis=0) / np.sum(
            gaussian_kernel(x[None, :], self.X, sigma=self.bandwidth), axis=-1)
        return g

    def second_order_grad(self, x):
        g2 = np.sum(gaussian_kernel_dx_dx(x, self.X, sigma=self.bandwidth), axis=0) / np.sum(
            gaussian_kernel(x[None, :], self.X, sigma=self.bandwidth), axis=-1)
        g2 -= self.grad(x) ** 2
        return g2

    def objective(self, X_test):
        N_test, D = X_test.shape

        objective = 0.0

        for a, x_a in enumerate(X_test):
            g = self.grad(x_a)
            g2 = self.second_order_grad(x_a)
            objective += (0.5 * np.sum(g ** 2) + np.sum(g2)) / N_test
        return objective


def get_sigma_star(X):
    """
    From the paper: 'sigma star is the median of pairwise distances of data'
    """
    dist_matrix = pairwise_distances(X)
    idxs = np.tril_indices(n=dist_matrix.shape[0], k=-1)
    median_dist = np.median(dist_matrix[idxs])
    return median_dist


def plot1(cv_full=False, cv_kde=False):
    y_est1, y_est2 = [], []
    x_range = range(2, 20, 2)
    for d in x_range:
        N = 200  # in the paper n=500, but that's too slow
        D = d
        X = np.random.randn(N, D)

        lmbda = 0.1 * np.power(N, -1. / 3)  # from the paper
        sigma_star = get_sigma_star(X)

        sigma = 0.8 * sigma_star
        if cv_full:
            print('CV for KernelExpFullGaussian')
            cv_sigma_factors = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
            cv_obj = []
            for sf in cv_sigma_factors:
                sigma = sf * sigma_star
                est1 = KernelExpFullGaussian(sigma, lmbda, D)
                cv_obj.append(np.mean(est1.xvalidate_objective(X)))
            sigma = cv_sigma_factors[np.argmin(cv_obj)] * sigma_star
            print('sigma', sigma)

        bandwidth = 0.1 * sigma_star
        if cv_kde:
            cv_bandwidth_factors = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            cv_obj = []
            for sf in cv_bandwidth_factors:
                bandwidth = sf * sigma_star
                est2 = Gaussian_KDE(bandwidth, D)
                cv_obj.append(np.mean(est2.xvalidate_objective(X)))
            bandwidth = cv_bandwidth_factors[np.argmin(cv_obj)] * sigma_star
            print('bandwidth', bandwidth)

        est1 = KernelExpFullGaussian(sigma, lmbda, D)
        est2 = Gaussian_KDE(bandwidth, D)

        est1.fit(X)
        est2.fit(X)

        y_est1.append(est1.objective(X))
        y_est2.append(est2.objective(X))

        print D, y_est1[-1]
        print D, y_est2[-1]
        print('----------------------------------------')

    plt.figure()
    plt.plot(x_range, y_est1, 'r', marker='o', label='score match')
    plt.plot(x_range, y_est2, 'b', marker='o', label='kde')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot1(cv_full=False, cv_kde=True)
