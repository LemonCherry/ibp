from itertools import chain

from sklearn.svm import SVR
import pandas as pd
from scipy.stats import ttest_rel
import numpy as np

from ibp import IBP
from svm_test import load_svmlight_file, MASK, StandardScaler

training_data = load_svmlight_file('data/Training-attributes.txt')
training_votes = np.loadtxt('data/Training-voting.txt').T
tsa_data = load_svmlight_file('tsa/tsa-attributes.txt')
tsa_truth = np.loadtxt('data/tsa-votes.txt')
tsa_truth = tsa_truth[:, 2] / tsa_truth[:, 3].astype(float)
tsb_data = load_svmlight_file('tsb/Testing.txt')
tsb_truth = np.loadtxt('tsb/results.txt')[:, 1]

training_cats = np.loadtxt('data/training-categories.txt')
tsa_cats = np.loadtxt('tsa/tsa-categories.txt')
tsb_cats = np.loadtxt('tsb/tsb-categories.txt')

ibp = IBP(enable_cluster=False)
ibp.fit(training_votes[2], training_votes[1])
y_per = training_votes[1] / training_votes[2].astype(float)
y_ibp = ibp(training_votes[2], training_votes[1])

scaler = StandardScaler()
training_x = scaler.fit_transform(training_data)[:, MASK]
tsa_x = scaler.transform(tsa_data)[:, MASK]
tsb_x = scaler.transform(tsb_data)[:, MASK]


def run_particular(arg):
    i, cat = arg
    X = np.array(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(training_cats[:, i], training_x))))
    X_tsa = np.array(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(tsa_cats[:, i], tsa_x))))
    y_tsa = np.array(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(tsa_cats[:, i], tsa_truth))))
    X_tsb = np.array(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(tsb_cats[:, i], tsb_x))))
    y_tsb = np.array(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(tsb_cats[:, i], tsb_truth))))

    clf_per = SVR()
    clf_ibp = SVR()
    clf_per.fit(X,
                np.array(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(training_cats[:, i], y_per)))))
    clf_ibp.fit(X,
                np.array(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(training_cats[:, i], y_ibp)))))

    tsa_y_hat_per = clf_per.predict(X_tsa)
    tsa_y_hat_ibp = clf_ibp.predict(X_tsa)

    mse_tsa_per = ((tsa_y_hat_per - y_tsa) ** 2).mean()
    mae_tsa_per = abs(tsa_y_hat_per - y_tsa).mean()
    rmse_tsa_per = mse_tsa_per ** 0.5
    mse_tsa_ibp = ((tsa_y_hat_ibp - y_tsa) ** 2).mean()
    mae_tsa_ibp = abs(tsa_y_hat_ibp - y_tsa).mean()
    rmse_tsa_ibp = mse_tsa_ibp ** 0.5

    tsb_y_hat_per = clf_per.predict(X_tsb)
    tsb_y_hat_ibp = clf_ibp.predict(X_tsb)

    mse_tsb_per = ((tsb_y_hat_per - y_tsb) ** 2).mean()
    mae_tsb_per = abs(tsb_y_hat_per - y_tsb).mean()
    rmse_tsb_per = mse_tsb_per ** 0.5
    mse_tsb_ibp = ((tsb_y_hat_ibp - y_tsb) ** 2).mean()
    mae_tsb_ibp = abs(tsb_y_hat_ibp - y_tsb).mean()
    rmse_tsb_ibp = mse_tsb_ibp ** 0.5

    print 2 ** (i + 1), cat, 'tsa', \
        mse_tsa_per, mse_tsa_ibp, (mse_tsa_per - mse_tsa_ibp) / mse_tsa_per, \
        mae_tsa_per, mae_tsa_ibp, (mae_tsa_per - mae_tsa_ibp) / mae_tsa_per, \
        rmse_tsa_per, rmse_tsa_ibp, (rmse_tsa_per - rmse_tsa_ibp) / rmse_tsa_per

    return [[2 ** (i + 1), cat, 'tsa', \
             mse_tsa_per, mse_tsa_ibp, (mse_tsa_per - mse_tsa_ibp) / mse_tsa_per, \
             mae_tsa_per, mae_tsa_ibp, (mae_tsa_per - mae_tsa_ibp) / mae_tsa_per, \
             rmse_tsa_per, rmse_tsa_ibp, (rmse_tsa_per - rmse_tsa_ibp) / rmse_tsa_per, \
             ttest_rel(tsa_y_hat_per, tsa_y_hat_ibp).pvalue],
            [2 ** (i + 1), cat, 'tsb', \
             mse_tsb_per, mse_tsb_ibp, (mse_tsb_per - mse_tsb_ibp) / mse_tsb_per, \
             mae_tsb_per, mae_tsb_ibp, (mae_tsb_per - mae_tsb_ibp) / mae_tsb_per, \
             rmse_tsb_per, rmse_tsb_ibp, (rmse_tsb_per - rmse_tsb_ibp) / rmse_tsb_per, \
             ttest_rel(tsb_y_hat_per, tsb_y_hat_ibp).pvalue]]


if __name__ == '__main__':
    from multiprocessing import Pool, freeze_support

    freeze_support()
    p = Pool(10)
    df_results = pd.DataFrame(
        list(chain.from_iterable(p.map(run_particular, [(i, cat) for i in range(2) for cat in range(2 ** (i + 1))]))),
        columns=['total_category', 'category', 'type',
                 'MSE PER', 'MSE IBP', 'Improvement % MSE',
                 'MAE PER', 'MAE IBP', 'Improvement % MAE',
                 'RMSE PER', 'RMSE IBP', 'Improvement % RMSE', 'T-Test p'])
    df_results.to_excel('results_tsa_tsb.xlsx')
