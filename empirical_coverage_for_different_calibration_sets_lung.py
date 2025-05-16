import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_fit_lung import X_rest,y_rest, clf

def scp_conformal_quantile(X_cali, y_cali, alpha):
    phats = clf.predict_proba(X_cali)
    scores = 1 - phats[np.arange(len(y_cali)), y_cali]
    qhat = np.quantile(scores, (1-alpha)*(1+1/(len(y_cali))))
    return qhat

def scp_compute_C(X_test, y_test, X_cali, y_cali, alpha, categories=np.array([0, 1, 2]), qhat = False):
    if qhat == False:
        qhat = scp_conformal_quantile(X_cali, y_cali, alpha)
    n = len(X_test)
    prediction_sets = np.zeros(n, dtype = object) 
    phats = clf.predict_proba(X_test)
    hits = 0
    for i in range(n):
        hypothesised_scores = 1-phats[i, categories]
        C = categories[hypothesised_scores <= qhat]
        prediction_sets[i] = C
        if y_test[i] in C:
            hits += 1
    emp_coverage = 1/n * hits
    return emp_coverage

def plot_cali_cond_cov(n_cs, rounds, alpha, X = X_rest, y = y_rest, test_size = 420):
    steps = len(n_cs)
    mean_emp_cov = np.zeros(steps)

    for i in range(rounds):
        X_cali, X_test, y_cali, y_test = train_test_split(X_rest, y_rest, stratify = y_rest, test_size = test_size, shuffle = True)
        current_coverages = np.zeros(steps)
        for j, n_c in enumerate(n_cs):
            X_cali_step = X_cali[:n_c + 1]
            y_cali_step = y_cali[:n_c + 1]
            current_coverages[j] = scp_compute_C(X_cali_step, y_cali_step, X_test, y_test, alpha)
        mean_emp_cov += current_coverages
        plt.plot(n_cs, current_coverages)
        print(i)
    mean_emp_cov = mean_emp_cov / rounds
    plt.plot(n_cs, mean_emp_cov,  linestyle = '--', label = 'Mean empirical coverage', color = 'k')
    plt.title(rf'Empirical Coverage for $\alpha={alpha}$')
    plt.xlabel(r'Size of Calibration Set, $n_c$')
    plt.grid()
    plt.legend(loc = 'upper right') 
    plt.show()

n_c = np.arange(20, 2000, 20)
plot_cali_cond_cov(n_c,10,0.1)