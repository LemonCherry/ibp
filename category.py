from itertools import chain

from sklearn.svm import SVR
import pandas as pd
from scipy.stats import ttest_rel
import numpy as np

from ibp import IBP
from svm_test import load_svmlight_file, MASK, StandardScaler

training_data = load_svmlight_file('data/Training-attributes.txt')
training_votes = np.loadtxt('data/Training-voting.txt').T

tsb_data = load_svmlight_file('tsb/Testing.txt')
tsb_truth = np.loadtxt('tsb/results.txt')[:, 1]

training_cats = np.loadtxt('data/training-categories.txt')
tsb_cats = np.loadtxt('tsb/tsb-categories.txt')

scaler = StandardScaler()
training_x = scaler.fit_transform(training_data)
tsb_x = scaler.transform(tsb_data)

cat_parameters = {
    'PER': {
        0: {'C': 0.51857752222309594, 'gamma': 0.0081511791481798362},
        1: {'C': 0.51857752222309594, 'gamma': 0.010816576451920346}
    },
    'IBP': {
        0: {'C': 0.64546959897370237, 'gamma': 0.0081511791481798362},
        1: {'C': 1.0, 'gamma': 0.0061425832656991797}
    }
}

def run_particular(arg):
    i, cat = arg

    ibp = IBP(enable_cluster=False)
    ibp.fit(training_votes[2], training_votes[1])
    y_per = training_votes[1] / training_votes[2].astype(float)
    y_ibp = ibp(training_votes[2], training_votes[1])

    X = np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(training_cats[:, i], training_x)))))
    X_tsb = np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(tsb_cats[:, i], tsb_x)))))
    y_tsb = np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(tsb_cats[:, i], tsb_truth)))))

    clf_per = SVR(C=1, gamma=0.001)
    clf_ibp = SVR(C=1000, gamma=0.0001)
    clf_per.fit(X,
                np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(training_cats[:, i], y_per))))))
    clf_ibp.fit(X,
                np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(training_cats[:, i], y_ibp))))))

    tsb_y_hat_per = clf_per.predict(X_tsb)
    tsb_y_hat_ibp = clf_ibp.predict(X_tsb)

    mse_tsb_per = ((tsb_y_hat_per - y_tsb) ** 2).mean()
    mae_tsb_per = abs(tsb_y_hat_per - y_tsb).mean()
    rmse_tsb_per = mse_tsb_per ** 0.5
    mse_tsb_ibp = ((tsb_y_hat_ibp - y_tsb) ** 2).mean()
    mae_tsb_ibp = abs(tsb_y_hat_ibp - y_tsb).mean()
    rmse_tsb_ibp = mse_tsb_ibp ** 0.5

    print(2 ** (i + 1), cat, 'tsb', (training_cats[:, i] == cat).astype(int).sum(),
          (tsb_cats[:, i] == cat).astype(int).sum(),
          mse_tsb_per, mse_tsb_ibp, (mse_tsb_per - mse_tsb_ibp) / mse_tsb_per,
          mae_tsb_per, mae_tsb_ibp, (mae_tsb_per - mae_tsb_ibp) / mae_tsb_per,
          rmse_tsb_per, rmse_tsb_ibp, (rmse_tsb_per - rmse_tsb_ibp) / rmse_tsb_per)

    return [[2 ** (i + 1), cat, 'tsb', (training_cats[:, i] == cat).astype(int).sum(),
             (tsb_cats[:, i] == cat).astype(int).sum(),
             mse_tsb_per, mse_tsb_ibp, (mse_tsb_per - mse_tsb_ibp) / mse_tsb_per,
             mae_tsb_per, mae_tsb_ibp, (mae_tsb_per - mae_tsb_ibp) / mae_tsb_per,
             rmse_tsb_per, rmse_tsb_ibp, (rmse_tsb_per - rmse_tsb_ibp) / rmse_tsb_per,
             ttest_rel(tsb_y_hat_per, tsb_y_hat_ibp).pvalue]]


def run_vary_cutoff(arg):
    k, cat = arg

    ibp = IBP(cutoff=k, enable_cluster=True, n_max_iter=10000)
    ibp.fit(training_votes[2], training_votes[1], cats=training_cats[:, 0])
    y_per = training_votes[1] / training_votes[2].astype(float)
    y_ibp = ibp(training_votes[2], training_votes[1], training_cats[:, 0])

    clf_per = SVR(**cat_parameters['PER'][cat])
    clf_ibp = SVR(**cat_parameters['IBP'][cat])

    X = np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(training_cats[:, 0], training_x)))))
    X_tsb = np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(tsb_cats[:, 0], tsb_x)))))
    y_tsb = np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(tsb_cats[:, 0], tsb_truth)))))

    clf_per.fit(X, np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(training_cats[:, 0], y_per))))))
    clf_ibp.fit(X, np.array(list(map(lambda x: x[1], filter(lambda x: x[0] == cat, zip(training_cats[:, 0], y_ibp))))))

    tsb_y_hat_per = clf_per.predict(X_tsb)
    tsb_y_hat_ibp = clf_ibp.predict(X_tsb)

    mse_tsb_per = ((tsb_y_hat_per - y_tsb) ** 2).mean()
    mae_tsb_per = abs(tsb_y_hat_per - y_tsb).mean()
    rmse_tsb_per = mse_tsb_per ** 0.5
    mse_tsb_ibp = ((tsb_y_hat_ibp - y_tsb) ** 2).mean()
    mae_tsb_ibp = abs(tsb_y_hat_ibp - y_tsb).mean()
    rmse_tsb_ibp = mse_tsb_ibp ** 0.5

    print(2, cat, 'tsb', (training_cats[:, 0] == cat).astype(int).sum(),
          (tsb_cats[:, 0] == cat).astype(int).sum(),
          mse_tsb_per, mse_tsb_ibp, (mse_tsb_per - mse_tsb_ibp) / mse_tsb_per,
          mae_tsb_per, mae_tsb_ibp, (mae_tsb_per - mae_tsb_ibp) / mae_tsb_per,
          rmse_tsb_per, rmse_tsb_ibp, (rmse_tsb_per - rmse_tsb_ibp) / rmse_tsb_per)

    return [[2, cat, 'tsb', (training_cats[:, 0] == cat).astype(int).sum(),
             (tsb_cats[:, 0] == cat).astype(int).sum(),
             mse_tsb_per, mse_tsb_ibp, (mse_tsb_per - mse_tsb_ibp) / mse_tsb_per,
             mae_tsb_per, mae_tsb_ibp, (mae_tsb_per - mae_tsb_ibp) / mae_tsb_per,
             rmse_tsb_per, rmse_tsb_ibp, (rmse_tsb_per - rmse_tsb_ibp) / rmse_tsb_per,
             ttest_rel(tsb_y_hat_per, tsb_y_hat_ibp).pvalue]]


if __name__ == '__main__':
    # run_particular((2, 1))
    # # run_vary_cutoff((10, 0))
    from multiprocessing import Pool, freeze_support

    # freeze_support()
    p = Pool(10)
    df_results = pd.DataFrame(list(chain.from_iterable(
        p.map(run_vary_cutoff, [(k, cat) for k in [10, 30, 50, 100, 200] for cat in range(2)]))),
        columns=['total_category', 'category', 'type',
                 '# of train', '# of tsb',
                 'MSE PER', 'MSE IBP', 'Improvement % MSE',
                 'MAE PER', 'MAE IBP', 'Improvement % MAE',
                 'RMSE PER', 'RMSE IBP', 'Improvement % RMSE', 'T-Test p'])
    df_results.to_excel('results_tsb_vary_k_ibp.xlsx')
