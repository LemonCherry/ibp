import pickle
import random
from itertools import cycle, chain

from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression, SGDRegressor, LinearRegression
from sklearn.svm import SVR

from ibp import IBP
from svm_test import load_svmlight_file, MASK, StandardScaler
from pickle import load
import numpy as np
from base import Review
from simulated import FUNCTIONS, user_sym


class Sampler:
    def __init__(self, n_rolls=10000):
        self.arr = load_svmlight_file('data/Training-attributes.txt')
        tmp = np.loadtxt('data/Training-voting.txt')
        self.ys = tmp[:, 1] / tmp[:, 2]
        self.n_rolls = n_rolls
        self.normalizer = StandardScaler().fit(self.arr)

    def __call__(self, h):
        weights = (1 - abs((self.ys - h))) ** 10
        weights = weights / weights.sum()
        return self.arr[np.random.choice(list(range(len(weights))), p=weights)]


class LinearFeatureGenerator:
    def __init__(self, feature_generator=FUNCTIONS[1], n_features=50):
        self.weights = np.random.uniform(-1.0, 1.0, n_features)
        self.weights /= sum(abs(self.weights))
        self.random_gen = feature_generator
        try:
            self.name = feature_generator.__name__
        except:
            pass

    def get_random_review(self, n_reviews=50000):
        feature_arr = self.random_gen((n_reviews, self.weights.size)) * 2 - 1
        y = (feature_arr * self.weights).sum(1) + 0.5
        span = max(y) - min(y)
        y -= min(y)
        y /= span
        return y, feature_arr


class MixedFeatureGenerator(LinearFeatureGenerator):
    def __init__(self, feature_generators=FUNCTIONS, n_features=50):
        LinearFeatureGenerator.__init__(self, feature_generators, n_features)
        self.name = 'Mixed'

    def get_random_review(self, n_reviews=50000):
        arr = []
        for idx, gen in zip(range(self.weights.size), cycle(self.random_gen)):
            arr.append(gen(n_reviews) * 2 - 1)
        arr = np.array(arr).T
        y = (arr * self.weights).sum(1) + 0.5
        span = max(y) - min(y)
        y -= min(y)
        y /= span
        return y, arr


class RandomErrorFeatureGenerator(LinearFeatureGenerator):
    def __init__(self, distortion_generator=FUNCTIONS[1], feature_generator=FUNCTIONS[1], n_features=50):
        LinearFeatureGenerator.__init__(self, feature_generator, n_features)
        self.distortion_gen = distortion_generator

    def get_random_review(self, n_reviews=50000):
        y, arr = LinearFeatureGenerator.get_random_review(self, n_reviews)
        y += self.distortion_gen(n_reviews)
        span = max(y) - min(y)
        y -= min(y)
        y /= span
        return y, arr


n_user = 1000
batch_size = 1000


def user_sym(h):
    ps = np.ones(n_user) * 0.5
    result_ts = []
    result_ns = []
    for i in range(1, h.size / batch_size + 1):
        h_slice = np.array(h[(i - 1) * batch_size: i * batch_size])
        vote_or_not = (np.random.uniform(0, 1, size=(h_slice.size, n_user)) <= ps ** (i ** 0.5)).astype(int).T
        p_array = np.random.uniform(0, 1, size=(n_user, h_slice.size))
        result_ts.append(
            ((p_array <= h_slice) * vote_or_not).sum(0))
        result_ns.append(vote_or_not.sum(0))
    result_ts = np.array(list(chain.from_iterable(result_ts))[:h.size])
    result_ns = np.array(list(chain.from_iterable(result_ns))[:h.size])
    return h, result_ts, result_ns


def run_pipeline(method):
    clf_per = LinearRegression()
    clf_ibp = LinearRegression()
    all_reviews = method.get_random_review(110000)
    train = all_reviews[0][:100000], all_reviews[1][:100000]
    test = all_reviews[0][100000:], all_reviews[1][100000:]
    hs, ts, ns = user_sym(train[0])
    ibp = IBP(enable_cluster=False)
    y_per = ts / (ns.astype(float) + np.finfo(float).eps)
    ibp.fit(ns, ts)
    y_ibp = ibp(ns, ts)

    print method.name, ((y_per - hs) ** 2).mean(), ((y_ibp - hs) ** 2).mean(), ttest_rel(y_per, y_ibp).pvalue,
    clf_per.fit(train[1], y_per)
    clf_ibp.fit(train[1], y_ibp,  ns / 5 + 0.0001)
    # print
    y_per_hat = clf_per.predict(test[1])
    y_ibp_hat = clf_ibp.predict(test[1])
    print ((y_per_hat - test[0]) ** 2).mean(), ((y_ibp_hat - test[0]) ** 2).mean(), ttest_rel(y_per_hat,
                                                                                              y_ibp_hat).pvalue
    return (train, y_per, y_ibp, ts, ns), (test, y_per_hat, y_ibp_hat)


if __name__ == '__main__':
    # print(user_sym(FUNCTIONS[1](50000)))
    # for i in range(100):
    t1, t2 = run_pipeline(MixedFeatureGenerator(n_features=16))
    import pandas as pd

    df_train = pd.DataFrame(
        np.array([list(line[0]) + list(line)[1:] for line in zip(t1[0][1], t1[0][0], t1[1], t1[2], t1[3], t1[4])]),
        columns=['Feature_{}'.format(i) for i in range(1, 17)] + ['y', 'per', 'ibp', 'pos votes', 'total votes'])
    df_test = pd.DataFrame(
        np.array([list(line[0]) + [line[1], line[2], line[3]] for line in zip(t2[0][1], t2[0][0], t2[1], t2[2])]),
        columns=['Feature_{}'.format(i) for i in range(1, 17)] + ['y', 'per', 'ibp'])
    exw = pd.ExcelWriter('dump3.xlsx')
    df_train.to_excel(exw, 'train')
    df_test.to_excel(exw, 'test')
    exw.close()
    # gens = map(LinearFeatureGenerator, FUNCTIONS)
    # for i in gens:
    #     run_pipeline(i)
