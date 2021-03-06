from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussianAdaptive
from kernel_exp_family.estimators.parameter_search_bo import plot_bayesopt_model_1d
from kernel_exp_family.examples.tools import visualise_fit_2d
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    """
    This simple demo demonstrates how to automatically tune parameters for the lite
    estimator, without user interaction.
    Note that this procedure is "hot-started", i.e. the second fit call uses the
    previously learned parameters to start learning parameters from.
    """
    N = 200
    D = 2

    # fit model to samples from a standard Gaussian
    X = np.random.randn(N, D)

    # fully automatic parameter tuning in every fit call
    est = KernelExpLiteGaussianAdaptive(sigma=1., lmbda=.001, D=D, N=N,
                                        # these paramters are all optional, to controll Bayesian opt.
                                        num_initial_evaluations=3, num_evaluations=3, minimum_size_learning=100,
                                        # these depend on how much data changes between the "fit" calls
                                        num_initial_evaluations_relearn=3, num_evaluations_relearn=3,
                                        # this can be used to adjust search spaces or include more parameters
                                        # by default, only sigma is optimised
                                        param_bounds={'sigma': [-3, 3]}
                                        )

    # automatically sets parameters
    est.fit(X)

    # only for illustration purpose
    plt.figure()
    plot_bayesopt_model_1d(est.bo)
    plt.title("Original fit")

    visualise_fit_2d(est, X)
    plt.suptitle("Original fit")

    # now change data, with different length scale
    X = np.random.randn(N, D) * .1

    # re-learns parameters, but starts from previous ones
    est.fit(X)

    visualise_fit_2d(est, X)
    plt.suptitle("New fit")

    # only for illustration purpose
    plt.figure()
    plot_bayesopt_model_1d(est.bo)
    plt.title("New fit")

    plt.show()
