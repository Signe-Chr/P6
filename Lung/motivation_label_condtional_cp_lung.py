import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_fit_lung import normal_rest,covid_rest,pneumonia_rest,clf

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

def plot_alpha_motivation_label(alphas,n_runs): 
    emp_cov_normal = np.zeros_like(alphas)
    emp_cov_covid = np.zeros_like(alphas)
    emp_cov_pneu = np.zeros_like(alphas)

    n_runs = 10
    for k in range(n_runs):
        print(f'round {k+1} out of {n_runs}')
        normal_cali, normal_test = train_test_split(normal_rest, train_size = 2/3)
        covid_cali, covid_test = train_test_split(covid_rest, train_size = 2/3)
        pneumonia_cali, pneumonia_test = train_test_split(pneumonia_rest, train_size = 2/3)

        X_cali = np.concatenate( (normal_cali, covid_cali, pneumonia_cali), axis = 0)
        y_cali = np.concatenate( (np.full(len(normal_cali), 0), np.full(len(covid_cali), 1), np.full(len(pneumonia_cali), 2)), axis = 0)

        for i, alpha in enumerate(alphas):
            qhat = scp_conformal_quantile(X_cali, y_cali, alpha)

            coverage_normal = scp_compute_C(normal_test, np.full(len(normal_test), 0) , X_cali, y_cali, alpha, qhat=qhat)
            coverage_covid = scp_compute_C(covid_test, np.full(len(covid_test), 1) , X_cali, y_cali, alpha, qhat=qhat)
            coverage_pneumonia = scp_compute_C(pneumonia_test, np.full(len(pneumonia_test), 2) , X_cali, y_cali, alpha, qhat=qhat)

            emp_cov_normal[i] += coverage_normal
            emp_cov_covid[i] += coverage_covid
            emp_cov_pneu[i] += coverage_pneumonia
        print(k)

    emp_cov_normal = emp_cov_normal/n_runs
    emp_cov_covid = emp_cov_covid/n_runs
    emp_cov_pneu = emp_cov_pneu/n_runs

    plt.plot(alphas, emp_cov_normal, label = 'Healthy')
    plt.plot(alphas, emp_cov_covid, label = 'Covid-19')
    plt.plot(alphas, emp_cov_pneu, label = 'Pneumonia')
    plt.plot(alphas, 1-alphas, label = r'$1-\alpha$', linestyle='--', color='k', dashes=(5,7), linewidth=1)

    plt.legend(loc = 'upper right')
    plt.grid()
    plt.title(r'Empirical Coverage')
    plt.xlabel(r'$\alpha$')

    plt.tight_layout()
    plt.show()
    
alphas = np.linspace(0.001, 0.5, 150)
plot_alpha_motivation_label(alphas,10)
