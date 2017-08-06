import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from kernel_exp_family.estimators.full.gaussian import KernelExpFullGaussian
from kernel_exp_family.kernels.kernels import gaussian_kernel_grad, gaussian_kernel_dx_dx, gaussian_kernel
from kernel_exp_family.estimators.estimator_oop import EstimatorBase
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    x_range = np.arange(2, 21, 2)
    for d in x_range:
        N = 500
        X = np.random.randn(N, d)
        X_test = np.random.randn(1500, d)

        lmbda = 0.1 * np.power(N, -1. / 3)  # from the paper
        sigma_star = get_sigma_star(X)

        sigma = 2 * (0.8 * sigma_star) ** 2
        if cv_full:
            print('CV for KernelExpFullGaussian')
            cv_sigma_factors = [0.001, 0.003, 0.005, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2., 2.2,
                                2.4, 2.6, 2.8, 3.0, 3.2]
            cv_obj = []
            for sf in cv_sigma_factors:
                sigma = 2 * (sf * sigma_star) ** 2
                est1 = KernelExpFullGaussian(sigma, lmbda, d)
                cv_obj.append(np.mean(est1.xvalidate_objective(X)))
                print(sigma, cv_obj[-1])
            sigma = 2 * (cv_sigma_factors[np.argmin(cv_obj)] * sigma_star) ** 2
            print('sigma', sigma)

        bandwidth = 2 * (0.1 * sigma_star) ** 2
        if cv_kde:
            print('CV for KDE')
            cv_bandwidth_factors = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
            cv_obj = []
            for sf in cv_bandwidth_factors:
                bandwidth = 2 * (sf * sigma_star) ** 2
                est2 = Gaussian_KDE(bandwidth, d)
                cv_obj.append(np.mean(est2.xvalidate_objective(X)))
                print(bandwidth, cv_obj[-1])
            bandwidth = 2 * (cv_bandwidth_factors[np.argmin(cv_obj)] * sigma_star) ** 2
            print('bandwidth', bandwidth)

        est1 = KernelExpFullGaussian(sigma, lmbda, d)
        est2 = Gaussian_KDE(bandwidth, d)

        # est1.fit(X)
        est2.fit(X)

        y_est1.append(0.)
        # y_est1.append(est1.objective(X_test))
        y_est2.append(est2.objective(X_test))

        # print d, y_est1[-1]
        print d, y_est2[-1]
        print('----------------------------------------')

    np.savez('plot1', np.array(y_est1), np.array(y_est2), x_range)
    print('saved np arrays')


def plot_from_npz():
    d = np.load('plot1.npz')
    est1, est2, x = d['arr_0'], d['arr_1'], d['arr_2']
    print(x)
    print(est1)
    print(est2)
    est1[0] = 0.
    est1[1] = 0.
    plt.figure()
    plt.plot(x, est1, 'r', marker='o', label='score match')
    plt.plot(x, est2, 'b', marker='o', label='kde')
    plt.legend()
    plt.savefig('plot1.png', bbox_inches='tight', pad_inches=0)
    print('Saved plot')


if __name__ == '__main__':
    plot1(cv_full=False, cv_kde=True)
    plot_from_npz()
