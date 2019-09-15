from base import InferenceBase
import numpy as np
from util import read_data


def ibp(xs, ns):
    """
    :type xs: np.ndarray
    :type ns: np.ndarray
    :param hs:
    :return:
    """
    xs = np.array(map(float, xs))
    ns = np.array(map(float, ns))
    hs = (xs + np.finfo(float).eps) / (ns + np.finfo(float).eps)
    mean = hs.mean()
    var = hs.var()
    alpha = beta = 0
    for i in range(10000):
        # print mean, var
        b = (mean * (1 - mean) / var - 1)
        alpha = mean * b
        beta = (1 - mean) * b
        hs = (xs + alpha + np.finfo(float).eps) / (ns + alpha + beta + np.finfo(float).eps)
        mean = hs.mean()
        var = hs.var()

    d = dict(map(lambda x: ((int(x[0]), int(x[1])), x[2]), zip(xs, ns, hs)))
    # for (x, n), h in d.iteritems():
    #     print x, n, h

    return alpha, beta


def calc_alpha_beta(hs):
    mean = hs.mean()
    var = hs.var()
    b = (mean * (1 - mean) / var - 1)
    alpha = mean * b
    beta = (1 - mean) * b
    return alpha, beta


class IBP(InferenceBase):
    def __init__(self, cutoff=10, enable_cluster=True, n_max_iter=10000):
        InferenceBase.__init__(self, cutoff, enable_cluster)
        if enable_cluster:
            self.children = dict()
        self.__name__ = 'IBP' if n_max_iter == 10000 else 'BP'
        self.n_max_iter = n_max_iter

    def fit(self, ns, xs, cats=None):
        if self.enable_cluster:
            d = {}
            for n, x, cat in zip(ns, xs, cats):
                d.setdefault(cat, [])
                d[cat].append((n, x))
            for cat in d:
                ns = list(map(lambda x: x[0], d[cat]))
                xs = list(map(lambda x: x[1], d[cat]))
                self.children[cat] = IBP(self.cutoff, False, self.n_max_iter)
                self.children[cat].fit(ns, xs)
            return

        ns = np.array(list(map(float, ns)))
        xs = np.array(list(map(float, xs)))[ns >= self.cutoff]
        ns = ns[ns >= self.cutoff]
        hs = xs / (ns + np.finfo(float).eps)
        self.init_hs = hs
        alpha, beta = calc_alpha_beta(hs)
        for i in range(self.n_max_iter):
            # print mean, var
            mean = hs.mean()
            hs = (xs + alpha) / (ns + alpha + beta + np.finfo(float).eps)
            if abs(hs.mean() - mean) < 0.00000000001:
                self.rounds = i + 1
                self.current_hs = hs
                break
            alpha, beta = calc_alpha_beta(hs)
            # print mean, var, alpha, beta

        self.alpha = alpha
        self.beta = beta

    def __call__(self, n, x, cat=None):
        if not self.enable_cluster:
            mode = (self.alpha + x) / (self.alpha + self.beta + n + np.finfo(float).eps)
            if type(n) is np.ndarray:
                mode[mode > 1] = 1
            else:
                mode = 1 if mode > 1 else mode
            return mode

        # distinct_cats = set(cat)
        def _fun():
            for ni, xi, ci in zip(n, x, cat):
                yield self.children[ci](ni, xi)

        return np.array(list(_fun()))
        # for c in distinct_cats:
        #     return self.children[c](n[cat == c], x[cat == c])


class BP(InferenceBase):
    def __init__(self, observed_ns, observed_xs):
        InferenceBase.__init__(self, observed_ns, observed_xs)
        xs = np.array(map(float, observed_xs))
        ns = np.array(map(float, observed_ns))
        hs = xs / (ns + np.finfo(float).eps)
        self.init_hs = hs
        alpha, beta = calc_alpha_beta(hs)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, n, x):
        mode = (self.alpha + x) / (self.alpha + self.beta + n + np.finfo(float).eps)
        if mode > 1:
            mode = 1.0
        return mode


if __name__ == '__main__':
    xs, ns = read_data(0)
    model = IBP(ns, xs)
