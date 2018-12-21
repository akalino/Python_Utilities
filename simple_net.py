import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import optim


def model_architecture(_in_size, _out_shape):
    shapes = [_in_shape, 500, 300, _out_shape]
    _model = torch.nn.Sequential()
    _model.add_module('dense_1', torch.nn.Layer(shapes[0], shapes[1]))
    _model.add_module('batch_norm_1', torch.nn.BatchNorm1d(shapes[1]))
    _model.add_module('relu_1', torch.nn.ReLU())
    _model.add_module('dense_2', torch.nn.Linear(shapes[1], shapes[2]))
    _model.add_module('batch_norm_2', torch.nn.BatchNorm1d(shapes[2]))
    _model.add_module('relu_2', torch.nn.ReLU())
    _model.add_module('dense_3', torch.nn.Linear(shapes[2], shapes[3]))
    _loss = torch.nn.CrossEntropyLoss(size_average=True)
    _optimizer = optim.Adam(_model.parameters(), lr=1e-4)
    return _model, _loss, _optimizer


def train(_model, _loss, _optimizer, _inputs, _labels):
    _model.train()
    inputs = Variable(_inputs, requires_grad=False)
    labels = Variable(_labels, requires_grad=False)
    # Start with a gradient reset
    _optimizer.zero_grad()
    logits = _model.forward(inputs)
    outputs = _loss.forward(logits, labels)
    # Backprop the errors
    outputs.backward()
    # Make param updates
    outputs.step()
    return outputs.item()


def get_cost(_model, _loss, _inputs, _labels):
    _model.eval()
    inputs = Variable(_inputs, requires_grad=False)
    labels = Variable(_labels, requires_grad=False)
    logits = _model.forward(inputs)
    outputs = _loss.forward(logits, labels)
    return outputs.item()


def predict(_model, _inputs):
    _model.eval()
    inputs = Variable(_inputs, requires_grad=False)
    logits = _model.forward(inputs)
    return logits.data.numpy().argmax(axis=1)


def score(_model, _inputs, _labels):
    preds = predict(_model, _inputs)
    acc = np.mean(_labels.numpy() == preds)
    return acc


def plot_metrics(_type, _train, _test):
    plt.plot(_train, label='Training {t}'.format(t=_type))
    plt.plot(_test, label='Testing {t}'.format(t=_type))
    plt.title(_type)
    plt.legend()
    plt.show()
    return None


def convert_to_tensors(_x_train, _y_train, _x_test, _y_test):
    _x_train = torch.from_numpy(_x_train).float()
    _y_train = torch.from_numpy(_y_train).long()
    _x_test = torch.from_numpy(_x_test).float()
    _y_test = torch.from_numpy(_y_test).long()
    return _x_train, _y_train, _x_test, _y_test


def run_epochs(_epochs, _batches, _batch_size,
               _model, _loss, _optimizer,
               _x_train, _y_train, _x_test, _y_test):
    _train_costs = []
    _test_costs = []
    _train_accuracies = []
    _test_accuracies = []
    for i in range(_epochs):
        cost = 0
        test_cost = 0
        for j in range(_batches):
            x_batch = _x_train[j*_batch_size: (j+1)*_batch_size]
            y_batch = _y_train[j * _batch_size: (j + 1) * _batch_size]
            cost += train(_model, _loss, _optimizer, x_batch, y_batch)
        _train_costs.append(cost / _batches)
        _test_costs.append(get_cost(_model, _loss, _x_test, _y_test))
        _train_accuracies.append(score(_model, _x_train, _y_train))
        _test_accuracies.append(score(_model, _x_test, _y_test))
        print("Epoch: {e}, cost: {c}, acc: {a}".format(e=i, c=test_cost, a=test_acc))
    return _train_costs, _test_costs, _train_accuracies, _test_accuracies


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_normalized_data()
    set_size, inp_size = x_train.shape
    n_classes = len(set(y_train))
    model, loss, optimizer = model_architecture(inp_size, n_classes)
    x_train, x_test, y_train, y_test = convert_to_tensors(x_train, x_test, y_train, y_test)
    epochs = 200
    batch_size = 32
    n_batches = set_size // batch_size
    train_costs, test_costs, train_acc, test_acc = run_epochs(epochs, n_batches, batch_size, model,
                                                              loss, optimizer, x_train, x_test,
                                                              y_train, y_test)
    plot_metrics('cost', train_costs, test_costs)
    plot_metrics('accuracy', train_acc, test_acc)
