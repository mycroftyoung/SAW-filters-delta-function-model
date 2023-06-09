import numpy as np
from delta_model import Filter, IDT

"""
IN DEVELOPMENT

"""

"""def loss(filter: Filter):
    X = filter.s_params[2]
    X_0 = x_0(filter.f_0, filter.freq, filter.bandwidth)
    W = weights(filter.f_0, filter.freq, filter.bandwidth)
    f_0 = filter.f_0

    f_0_A = np.max(X)
    delta_X = np.abs(X - f_0_A)

    loss = np.sum((delta_X - X_0) ** 2 ** W)
    return loss


def weights(f_0, freq, bandwidth):
    W = np.exp(- (freq - f_0) ** 2 / (8 * (bandwidth * f_0) ** 2)) / (2 * bandwidth * f_0 * np.sqrt(2 * np.pi))
    return W


def x_0(f_0, freq, bandwidth, deltaA=20):
    x = [(0 if x >= f_0 - bandwidth * f_0 or x <= f_0 + bandwidth * f_0 else deltaA) for x in freq]

    return x


# Метрика точности
def X0_metric(f_0, freq, deltaA, bandwidth):
    x = [(deltaA if x >= f_0 - bandwidth * f_0 or x <= f_0 + bandwidth * f_0 else 0) for x in freq]

    return x


def accuracy_3db(X, X_0_metric):
    delta = np.abs(X - X_0_metric)

    count = 0
    for x in delta:
        if x <= 3:
            count += 1

    acc = count / X.shape[0]

    return acc
"""