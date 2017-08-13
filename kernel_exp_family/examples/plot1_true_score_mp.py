import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from kernel_exp_family.estimators.full.gaussian import KernelExpFullGaussian
from kernel_exp_family.estimators.kde.kde import KDEGaussianTrue
from kernel_exp_family.estimators.full.gaussian import grad as exp_full_gaussian_grad
from kernel_exp_family.tools.assertions import assert_array_shape, assert_positive_int
import matplotlib
from functools import partial
from multiprocessing import Pool
from kernel_exp_family.tools.log import SimpleLogger
import sys

matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_name = 'plot1_true_score_mp'
sys.stdout = SimpleLogger('%s.log' % script_name)
sys.stderr = sys.stdout


class ground_truth():
    def __init__(self):
        pass

    def log_pdf(self, x):
        return -0.5 * np.dot(x, x)

    def grad(self, x):
        return -0.5 * x

    def fit(self, X):
        pass

    def log_pdf_multiple(self, X):
        return np.array([self.log_pdf(x) for x in X])

    def objective(self, x):
        return 0.


class KernelExpFullGaussianTrue(KernelExpFullGaussian):
    def compute_objective(self, X_test, X_train, sigma, alpha, beta):
        true_density = ground_truth()
        N_test, D = X_test.shape
        objective = 0.0
        for a, x_a in enumerate(X_test):
            g = exp_full_gaussian_grad(x_a, X_train, sigma, alpha, beta)
            objective += np.sum((g - true_density.grad(x_a)) ** 2) / N_test
        return objective

    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        return self.compute_objective(X, self.X, self.sigma, self.alpha, self.beta)


def get_sigma_star(X):
    """
    From the paper: 'sigma star is the median of pairwise distances of data'
    """
    dist_matrix = pairwise_distances(X)
    idxs = np.tril_indices(n=dist_matrix.shape[0], k=-1)
    median_dist = np.median(dist_matrix[idxs])
    return median_dist


def gen_data_xvalidate_objective(estimator, N, d, num_folds=5):
    assert_positive_int(num_folds)
    O = []
    for i in range(num_folds):
        x_train = np.random.randn(N, d)  # not a proper CV
        x_test = np.random.randn(N, d)
        try:
            estimator.fit(x_train)
        except Exception, e:
            print(N, d, e)
            i -= 1
        O.append(estimator.objective(x_test))
    return np.mean(O)


def compute_for_d(d, cv_full, cv_kde):
    N = 500
    X = np.random.randn(N, d)
    X_test = np.random.randn(1500, d)

    lmbda = 0.1 * np.power(N, -1. / 3) / 2.  # from the paper
    print('Lambda', lmbda)
    sigma_star = get_sigma_star(X)

    sigma = 2 * (0.8 * sigma_star) ** 2
    if cv_full:
        print('CV for KernelExpFullGaussian')
        cv_sigma_factors = [0.005, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2., 2.2,
                            2.4, 2.6, 2.8, 3.0, 3.2, 10, 100]
        cv_obj = []
        for sf in cv_sigma_factors:
            sigma = 2 * (sf * sigma_star) ** 2
            est1 = KernelExpFullGaussianTrue(sigma, lmbda, d)
            obj = np.mean(est1.xvalidate_objective(X))
            obj = np.inf if np.isnan(obj) else obj
            cv_obj.append(obj)
            print('Full', d, sf, sigma, cv_obj[-1])
        sigma = 2 * (cv_sigma_factors[np.argmin(cv_obj)] * sigma_star) ** 2
        print('sigma', sigma)

    bandwidth = 2 * (0.1 * sigma_star) ** 2
    if cv_kde:
        print('CV for KDE')
        cv_bandwidth_factors = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
        cv_obj = []
        for sf in cv_bandwidth_factors:
            bandwidth = 2 * (sf * sigma_star) ** 2
            est2 = KDEGaussianTrue(bandwidth, d, ground_truth=ground_truth())
            cv_obj.append(np.mean(est2.xvalidate_objective(X)))
            print('KDE', d, sf, bandwidth, cv_obj[-1])
        bandwidth = 2 * (cv_bandwidth_factors[np.argmin(cv_obj)] * sigma_star) ** 2
        print('bandwidth', bandwidth)

    est1 = KernelExpFullGaussianTrue(sigma, lmbda, d)
    est2 = KDEGaussianTrue(bandwidth, d, ground_truth=ground_truth())

    est1.fit(X)
    est2.fit(X)

    print(d, est1.objective(X_test), est2.objective(X_test))
    return d, est1.objective(X_test), est2.objective(X_test)


def plot1(cv_full=False, cv_kde=False):
    x_range = list(np.arange(2, 21, 2))
    y_est1, y_est2 = [0] * len(x_range), [0] * len(x_range)
    pool = Pool(processes=6)
    res = pool.map(partial(compute_for_d, cv_full=cv_full, cv_kde=cv_kde), x_range)

    print(res)
    for (d, e1, e2) in res:
        i = x_range.index(d)
        y_est1[i] = e1
        y_est2[i] = e2

    np.savez(script_name, np.array(y_est1), np.array(y_est2), np.array(x_range))
    print('saved np arrays')


def plot_from_npz():
    d = np.load('%s.npz' % script_name)
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
    plt.savefig('%s.png' % script_name, bbox_inches='tight', pad_inches=0)
    print('Saved plot')


if __name__ == '__main__':
    plot1(cv_full=True, cv_kde=True)
    plot_from_npz()
