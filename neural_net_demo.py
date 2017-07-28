import numpy as np
import matplotlib.pyplot as plt


def create_data():
    # Generate three gaussian clouds as data points
    np.random.seed(1)
    n_samples = 500
    # k classes
    k = 3

    # First centered at [0, -2]
    x_1 = np.random.randn(n_samples, 2) + np.array([0, -2])
    # Second centered at [2, 2]
    x_2 = np.random.randn(n_samples, 2) + np.array([2, 2])
    # Third centered at [-2, -2]
    x_3 = np.random.randn(n_samples, 2) + np.array([-2, 2])

    x = np.vstack([x_1, x_2, x_3])

    # Create labels
    y = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)

    # One-hot encoding of y
    t = np.zeros((len(y), k))
    for i in xrange(len(y)):
        t[i, y[i]] = 1

    return x, y, t


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()


def derivative_w2(Z, T, Y):
    return Z.T.dot(T - Y)


def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)


def derivative_w1(X, Z, T, Y, W2):
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    ret1 = X.T.dot(dZ)
    return ret1


def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)


def plot_data(_x, _y):
    # Visualize data
    plt.scatter(_x[:, 0], _x[:, 1], c=_y, s=100, alpha=0.5)
    plt.show()


def classifier_rate(_y, _p):
    """
    A function to evaluate classifier accuracy rates.
    Parameters
    ----------
    _y : The input labels (actuals).
    _p : The input predictions (from the model).

    Returns
    -------
    r : The classification rate.
    """
    n_correct = 0
    n_total = len(_y)
    for i in xrange(n_total):
        if _y[i] == _p[i]:
            n_correct += 1
    r = float(n_correct) / n_total
    return r


def softmax(a):
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    """
    A function to calculate the forward propagation of data through the network.
    Parameters
    ----------
    X : The input data.
    W1 : The first set of weights.
    b1 : The first bias on the first layers.
    W2 : The second set of hidden weights.
    b2 : The bias on the hidden weight layer.
    Returns
    -------
    Y : The predicted output values.
    Z : The values at the hidden layer.
    """
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z


def backprop():
    x, y, t = create_data()
    plot_data(x, y)
    # Randomize initial weights
    d = 2
    k = 3
    m = 12
    np.random.seed(1)

    # Randomly initialize weights
    w_1 = np.random.randn(d, m)
    b_1 = np.random.randn(m)
    w_2 = np.random.randn(m, k)
    b_2 = np.random.randn(k)

    # Initialize the velocities to zero
    v_w_1 = np.zeros(w_1.shape)
    v_b_1 = np.zeros(b_1.shape)
    v_w_2 = np.zeros(w_2.shape)
    v_b_2 = np.zeros(b_2.shape)

    # Initialize caches for adaptive learning using RMSProp
    cache_w_1 = 0
    cache_b_1 = 0
    cache_w_2 = 0
    cache_b_2 = 0

    # Lots of hyperparameter choices
    decay = 0.999
    eps = 10e-7
    lr = 0.001
    reg = 0.001
    mu = 0.5
    costs = []

    for epoch in xrange(100000):
        out, hid = forward(x, w_1, b_1, w_2, b_2)
        if epoch % 100 == 0:
            c = cost(t, out)
            p = np.argmax(out, axis=1)
            r = classifier_rate(y, p)
            print('Cost is {c} at classifier rate {r}'.format(c=c,
                                                              r=r))
            costs.append(c)
        g_w_2 = derivative_w2(hid, t, out) - reg * w_2
        cache_w_2 = decay*cache_w_2 + (1 - decay)*g_w_2*g_w_2
        v_w_2 = (mu * mu * v_w_2) - (1 + mu) * lr * g_w_2 / (np.sqrt(cache_w_2) + eps)
        w_2 -= v_w_2

        g_b_2 = derivative_b2(t, out) - reg * b_2
        cache_b_2 = decay*cache_b_2 + (1 - decay)*g_b_2*g_b_2
        v_b_2 = (mu * mu * v_b_2) - (1 + mu) * lr * g_b_2 / (np.sqrt(cache_b_2) + eps)
        b_2 -= v_b_2

        g_w_1 = derivative_w1(x, hid, t, out, w_2) - reg * w_1
        cache_w_1 = decay*cache_w_1 + (1 - decay)*g_w_1*g_w_1
        v_w_1 = (mu * mu * v_w_1) - (1 + mu) * lr * g_w_1 / (np.sqrt(cache_w_1) + eps)
        w_1 -= v_w_1

        g_b_1 = derivative_b1(t, out, w_2, hid) - reg * b_1
        cache_b_1 = decay*cache_b_1 + (1 - decay)*g_b_1*g_b_1
        v_b_1 = (mu * mu * v_b_1) - (1 + mu)*lr*g_b_1 / (np.sqrt(cache_b_1) + eps)
        b_1 -= v_b_1

    plt.plot(costs)
    plt.show()


if __name__ == "__main__":
    backprop()
