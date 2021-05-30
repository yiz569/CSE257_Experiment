import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

np.set_printoptions(precision=3, linewidth=240, suppress=True)
np.random.seed(1993)

############################################## Logistic Regression ###############################################


def sigmoid(z):
    return 1. / (1. + np.exp(-np.clip(z, -15, 15)))

# features is an [n x d] matrix of features (each row is one data point)
# labels is an n-dimensional vector of labels (0/1)


def logistic_loss(x, features, labels):
    n = features.shape[0]
    probs = sigmoid(np.dot(features, x))
    return (-1./n) * (np.dot(labels, np.log(1e-12 + probs)) + np.dot(1-labels, np.log(1e-12 + 1-probs)))


def logistic_loss_full_gradient(x, features, labels):
    return np.dot(np.transpose(features), sigmoid(np.dot(features, x)) - labels) / features.shape[0]


def logistic_loss_stochastic_gradient(x, features, labels, minibatch_size):
    idxs = np.random.randint(0, features.shape[0], minibatch_size)
    fts = features[idxs, :]
    res = sigmoid(np.dot(fts, x)) - labels[idxs]
    return np.dot(res.reshape(1, minibatch_size), fts).reshape(len(x))/minibatch_size


def logistic_loss_hessian(x, features, labels):
    s = sigmoid(np.dot(features, x))
    s = s * (1 - s)
    return np.dot(np.transpose(features) * s, features) / features.shape[0]

##################################################################################################################


def one_inner_outer_iteration(x_start, M, K, stepsize):
    grads = np.zeros_like(x_start)
    for _ in range(M):
        x = x_start.copy()
        for _ in range(K):
            g = objective_stochastic_gradient(x, 1)
            grads += g / M
            x -= stepsize * g
    return grads


def inner_outer_sgd(x0_len, M, K, R, inner_stepsize, outer_stepsize, loss_freq, avg_window=8):
    losses = []
    iterates = [np.zeros(x0_len)]
    for r in range(R):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        direction = one_inner_outer_iteration(
            iterates[-1], M, K, inner_stepsize)
        iterates.append(iterates[-1] - outer_stepsize * direction)
        if (r+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates, axis=0)))
            print(
                'Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(r+1, R, losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
    print('')
    return losses


def local_sgd(x0_len, M, K, R, stepsize, loss_freq):
    return inner_outer_sgd(x0_len, M, K, R, stepsize, stepsize, loss_freq)


def minibatch_sgd(x0_len, T, batchsize, stepsize, loss_freq, avg_window=8):
    losses = []
    iterates = [np.zeros(x0_len)]
    for t in range(T):
        if len(iterates) >= avg_window:
            iterates = iterates[-(avg_window-1):]
        iterates.append(
            iterates[-1] - stepsize * objective_stochastic_gradient(iterates[-1], batchsize))
        if (t+1) % loss_freq == 0:
            losses.append(objective_value(np.average(iterates, axis=0)))
            print(
                'Iteration: {:d}/{:d}   Loss: {:f}                 \r'.format(t+1, T, losses[-1]), end='')
            if losses[-1] > 10*losses[0]:
                print('\nLoss is diverging: Loss = {:f}'.format(losses[-1]))
                return losses
    print('')
    return losses


def gradient_descent(x0_len, T, stepsize):
    x = np.zeros(x0_len)
    losses = [objective_value(x)]
    for t in range(T):
        x -= stepsize * objective_full_gradient(x)
        losses.append(objective_value(x))
    return np.array(losses)


def newtons_method(x0_len, max_iter=1000, tol=1e-6):
    x = np.zeros(x0_len)
    stepsize = 0.5
    for t in range(max_iter):
        gradient = objective_full_gradient(x)
        hessian = objective_hessian(x)
        update_direction = np.linalg.solve(hessian, gradient)
        x -= stepsize * update_direction
        newtons_decrement = np.sqrt(np.dot(gradient, update_direction))
        if newtons_decrement <= tol:
            print(
                "Newton's method converged after {:d} iterations".format(t+1))
            return objective_value(x)
    print("Warning: Newton's method failed to converge")
    return objective_value(x)

##################################################################################################################


def experiment(M, K, R, plotfile):
    loss_freq = 5
    n_reps = 5
    n_stepsizes = 10

    tt_stepsizes = [np.exp(exponent)
                    for exponent in np.linspace(-6, 0, n_stepsizes)]
    lg_stepsizes = [np.exp(exponent)
                    for exponent in np.linspace(-6, 0, n_stepsizes)]
    lc_stepsizes = [np.exp(exponent)
                    for exponent in np.linspace(-8, -1, n_stepsizes)]

    print('Doing Thumb Twiddling...')
    thumb_results = np.zeros((R//loss_freq, len(tt_stepsizes)))
    for i, stepsize in enumerate(tt_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize,
                                                   i+1, len(tt_stepsizes)))
        for rep in range(n_reps):
            thumb_results[:, i] += (minibatch_sgd(x0_len,
                                                  R, M, stepsize, loss_freq) - fstar) / n_reps

    print('Doing Large Minibatch...')
    large_results = np.zeros((R//loss_freq, len(lg_stepsizes)))
    for i, stepsize in enumerate(lg_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize,
                                                   i+1, len(lg_stepsizes)))
        for rep in range(n_reps):
            large_results[:, i] += (minibatch_sgd(x0_len,
                                                  R, K*M, stepsize, loss_freq) - fstar) / n_reps

    print('Doing Local SGD...')
    local_results = np.zeros((R//loss_freq, len(lc_stepsizes)))
    for i, stepsize in enumerate(lc_stepsizes):
        print('Stepsize {:.5f}:  {:d}/{:d}'.format(stepsize,
                                                   i+1, len(lc_stepsizes)))
        for rep in range(n_reps):
            local_results[:, i] += (local_sgd(x0_len, M,
                                              K, R, stepsize, loss_freq) - fstar) / n_reps

    l0 = objective_value(np.zeros(x0_len))-fstar
    local_l = np.concatenate([[l0], np.min(local_results, axis=1)])
    thumb_l = np.concatenate([[l0], np.min(thumb_results, axis=1)])
    large_l = np.concatenate([[l0], np.min(large_results, axis=1)])

    Rs = [0] + list(range(loss_freq, R+1, loss_freq))

    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Rs, local_l, label='Local SGD')
    ax.plot(Rs, thumb_l, label='Thumb-Twiddling SGD')
    ax.plot(Rs, large_l, label='MB-SGD')
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Round of Communication')
    ax.set_ylabel('Objective Value')
    ax.set_title('M={:d}, K={:d}, R={:d}'.format(M, K, R))
    ax.legend(handles, labels, loc='upper right')
    plt.savefig(plotfile)

##################################################################################################################


dataset = 'synthetic'
scale = 12
dim = 25
N = 5000
features = scale * np.dot(np.random.randn(N, dim),
                          np.diag(np.linspace(1./dim, 1., dim)))  # [N x d]
w1 = np.random.randn(dim)/(scale*np.sqrt(dim))
w2 = np.random.randn(dim)/(scale*np.sqrt(dim))
b1 = 2*np.random.randn(1)
b2 = 2*np.random.randn(1)
prob_positive = sigmoid(np.minimum(
    np.dot(features, w1)+b1, np.dot(features, w2)+b2))
labels = np.random.binomial(1, prob_positive)
features = np.concatenate([features, np.ones((N, 1))], axis=1)  # for bias term
x0_len = features.shape[1]


loss_function = 'binary logistic loss'


def objective_value(x): return logistic_loss(
    x, features, labels)  # + 0.05*np.linalg.norm(x)**2


def objective_full_gradient(x): return logistic_loss_full_gradient(
    x, features, labels)  # + 0.1*x


def objective_stochastic_gradient(x, minibatch_size): return logistic_loss_stochastic_gradient(
    x, features, labels, minibatch_size)  # + 0.1*x


def objective_hessian(x): return logistic_loss_hessian(
    x, features, labels)  # + 0.1*np.eye(len(x))


fstar = newtons_method(x0_len)
print('Fstar = {:.5f}'.format(fstar))

experiment(M=500, K=5, R=40, plotfile='plots/bigM-smallK_2.png')
experiment(M=500, K=20, R=40, plotfile='plots/bigM-bigK_2.png')
experiment(M=50, K=20, R=40, plotfile='plots/smallM-bigK_2.png')
experiment(M=500, K=40, R=40, plotfile='plots/bigM-biggerK_2.png')
experiment(M=50, K=40, R=40, plotfile='plots/smallM-biggerK_2.png')
experiment(M=50, K=5, R=40, plotfile='plots/smallM-smallK_2.png')
