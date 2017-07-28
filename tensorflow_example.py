import numpy as np
import tensorflow as tf


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


def init_weights(_shape):
    return tf.Variable(tf.random_normal(_shape, stddev=0.01))


def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2


def run_network():
    X, Y, T = create_data()
    # Randomize initial weights
    D = 2
    K = 3
    M = 8
    tf_X = tf.placeholder(tf.float32, [None, D])
    tf_Y = tf.placeholder(tf.float32, [None, K])

    W1 = init_weights([D, M])
    b1 = init_weights([M])
    W2 = init_weights([M, K])
    b2 = init_weights([K])

    py_x = forward(tf_X, W1, b1, W2, b2)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, tf_Y))

    lr = 0.05
    train_fn = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in xrange(1000):
        sess.run(train_fn, feed_dict={tf_X: X,
                                      tf_Y: T})
        pred = sess.run(predict_op, feed_dict={tf_X: X,
                                               tf_Y: T})
        if i % 10 == 0:
            print(np.mean(Y == pred))


if __name__ == "__main__":
    run_network()
