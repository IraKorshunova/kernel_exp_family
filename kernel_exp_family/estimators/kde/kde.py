from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.kernels.kernels import gaussian_kernel_grad, gaussian_kernel
import numpy as np


class KDEGaussianTrue(EstimatorBase):
    def __init__(self, bandwidth, D, ground_truth):
        super(KDEGaussianTrue, self).__init__(D)
        self.bandwidth = bandwidth
        self.ground_truth = ground_truth

    def fit(self, X):
        self.X = X

    def log_pdf(self, x):
        return np.log(np.sum(gaussian_kernel(x, self.X, sigma=self.bandwidth), axis=-1) / (
            self.X.shape[0] * self.bandwidth))

    def grad(self, x):
        if x.ndim == 1:
            g = np.sum(gaussian_kernel_grad(x, self.X, sigma=self.bandwidth), axis=0) / np.sum(
                gaussian_kernel(x[None, :], self.X, sigma=self.bandwidth), axis=-1)
            return g
        else:
            grads = []
            for i in xrange(x.shape[0]):
                g_i = self.grad(x[i])
                grads.append(g_i)
            return np.asarray(grads)

    def objective(self, X_test):
        N_test, D = X_test.shape

        objective = 0.0
        for a, x_a in enumerate(X_test):
            g = self.grad(x_a)
            g_is_nan = np.any(np.isnan(g))
            if not g_is_nan:
                objective += np.sum((g - self.ground_truth.grad(x_a)) ** 2)
            else:
                N_test -= 1
        try:
            objective = objective / N_test
        except:
            objective = np.inf
        return objective

    def get_name(self):
        return self.__class__.__name__

    def get_parameter_names(self):
        return ['bandwidth']
