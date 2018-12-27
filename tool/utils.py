import numpy as np
import torch


def relative_ranks(x):
    def ranks(x):
        ranks = np.zeros(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    y = ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    return y / (x.size - 1.) - 0.5


class Schedule(object):
    def value(self, t):
        raise NotImplementedError()


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        assert self._outside_value is not None
        return self._outside_value


class Adam(object):
    def __init__(self, shape, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.stepsize, self.beta1, self.beta2, self.epsilon = stepsize, beta1, beta2, epsilon
        self.t = 0
        self.m = torch.zeros(shape)
        self.v = torch.zeros(shape)

    def step(self, g):
        self.t += 1
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        return - a * self.m / (torch.sqrt(self.v) + self.epsilon)
