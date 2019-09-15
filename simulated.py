import pickle
from itertools import chain, izip

from sklearn.svm import SVR

from em_production import IBP2
from ibp import IBP
import pandas as pd
from scipy.stats import ttest_rel

import numpy as np

from svm_test import MASK

SIZE = 1000


def regen(fun):
    def _fun(s):
        x = fun(size=s)
        while (x < 0).any():
            x[x < 0] = fun(size=(x < 0).astype(int).sum())
        while (x > 1).any():
            x[x > 1] = fun(size=(x > 1).astype(int).sum())
        return x

    f = _fun
    f.__name__ = fun.__name__
    return f


@regen
def uniform(size):
    return np.random.uniform(size=size)


@regen
def normal(size):
    return np.random.normal(0.5, 0.15, size=size)


@regen
def beta_0_9_0_6(size):
    return np.random.beta(0.9, 0.6, size=size)


@regen
def beta_0_6_0_9(size):
    return np.random.beta(0.6, 0.9, size=size)


@regen
def beta_1_3(size):
    return np.random.beta(1, 3, size=size)


@regen
def beta_2_2(size):
    return np.random.beta(2, 2, size=size)


@regen
def beta_2_5(size):
    return np.random.beta(2, 5, size=size)


@regen
def beta_5_1(size):
    return np.random.beta(5, 1, size=size)


@regen
def gamma_1_2(size):
    return np.random.gamma(1, 2, size=size) / 20


@regen
def gamma_2_2(size):
    return np.random.gamma(2, 2, size=size) / 20


@regen
def gamma_3_2(size):
    return np.random.gamma(3, 2, size=size) / 20


@regen
def gamma_5_1(size):
    return np.random.gamma(5, 1, size=size) / 20


@regen
def sym_gamma_1_2(size):
    return 1 - np.random.gamma(1, 2, size=size) / 20


@regen
def sym_gamma_2_2(size):
    return 1 - np.random.gamma(2, 2, size=size) / 20


@regen
def sym_gamma_3_2(size):
    return 1 - np.random.gamma(3, 2, size=size) / 20


@regen
def sym_gamma_5_1(size):
    return 1 - np.random.gamma(5, 1, size=size) / 20


FUNCTIONS = [uniform,
             normal,
             beta_0_9_0_6,
             beta_0_6_0_9,
             beta_1_3,
             beta_2_2,
             beta_2_5,
             beta_5_1,
             gamma_1_2,
             gamma_2_2,
             gamma_3_2,
             gamma_5_1,
             sym_gamma_1_2,
             sym_gamma_2_2,
             sym_gamma_3_2,
             sym_gamma_5_1]

SKIP = 100

GLOBAL_PS = np.array(list(chain.from_iterable(map(lambda x: [0.5 ** (x ** 0.5)] * SKIP, range(1, SIZE / SKIP + 1)))))


def user_sym(h, n_user=1000):
    vote_or_not = (np.random.uniform(0, 1, size=(h.size, n_user)) <= GLOBAL_PS).astype(int).T
    p_array = np.random.uniform(0, 1, size=(n_user, h.size))

    return h, ((p_array <= h) * vote_or_not).sum(0), ((p_array > h) * vote_or_not).sum(0)


def user_sym_batch(hs, n_user=1000):
    vote_or_not = (np.random.uniform(0, 1, size=(n_user, SIZE)) <= GLOBAL_PS).astype(int)
    p_array = np.random.uniform(0, 1, size=(n_user, SIZE))
    for h in hs:
        yield h, ((p_array <= h) * vote_or_not).sum(0), ((p_array > h) * vote_or_not).sum(0)


if __name__ == '__main__':
    hs = map(lambda x: x(SIZE), FUNCTIONS)
    df_total = pd.DataFrame(columns=['name', 'PER', 'IBP', 'EM', 'T.TEST PER-IBP', 'T.TEST EM-IBP', 'T.TEST PER-EM'])
    COLUMNS = ['h', 'PER', 'IBP', 'EM']
    from synthetic_features import Sampler

    clf_per = pickle.load(open('BasicInference.model'))
    clf_ibp = pickle.load(open('IBPModel.model'))
    sampler = Sampler()

    for name, (h, ts, fs) in izip(FUNCTIONS, user_sym_batch(hs)):
        # print ts, fs
        xs = ts
        ns = ts + fs
        ibp_model = IBP(enable_cluster=False)
        ibp_model.fit(ns, xs)
        # em_model = IBP2(ns, xs)
        h_per = xs / (ns + np.finfo(float).eps)
        # print h_per
        h_hat_ibp = np.array(map(lambda (n, x): ibp_model(n, x), izip(ns, xs)))
        X = np.array(list(map(sampler, h)))
        X = sampler.normalizer.transform(X)
        h_per = clf_per.predict(X[:, MASK])
        h_hat_ibp = clf_ibp.predict(X[:, MASK])
        # print h_hat_ibp
        # h_hat_em = np.array(map(lambda (n, x): em_model(n, x), izip(ns, xs)))
        print 'name = ' + name.__name__
        mse_per = ((h_per - h) ** 2).mean()
        mse_ibp = ((h_hat_ibp - h) ** 2).mean()
        # mse_em = ((h_hat_em - h) ** 2).mean()
        print 'MSE_PER = {}'.format(mse_per)
        print 'MSE_IBP = {}'.format(mse_ibp)
        print 'ttest = {}'.format(ttest_rel(h_per, h_hat_ibp).pvalue)
        # print 'MSE_EM = {}'.format(((h_hat_em - h) ** 2).mean())
        # df_total = df_total.append(
        #     dict(zip(['name', 'PER', 'IBP',
        #               # 'EM',
        #               'T.TEST PER-IBP',
        #               # 'T.TEST EM-IBP', 'T.TEST PER-EM'
        #               ],
        #              [name.__name__, mse_per, mse_ibp,
        #               # mse_em,
        #               ttest_rel(h_per, h_hat_ibp),
        #               # ttest_rel(h_per, h_hat_em),
        #               # ttest_rel(h_hat_em, h_hat_ibp)
        #               ])),
        #     ignore_index=True)
        # df = pd.DataFrame(data=np.array([h, h_per, h_hat_ibp, ttest_rel(h_per, h_hat_ibp).pvalue]), columns=['name', 'PER', 'IBP', 'T.TEST PER-IBP'])
        # df.to_csv('simulation_{}.csv'.format(name.__name__))
    # df_total.to_csv('simulation_result_total.csv')
