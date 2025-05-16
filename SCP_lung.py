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
    proportion_of_setlengths = np.zeros(len(categories)+1)
    hits = 0
    for i in range(n):
        hypothesised_scores = 1-phats[i, categories]
        C = categories[hypothesised_scores <= qhat]
        prediction_sets[i] = C
        len_C = len(C)
        proportion_of_setlengths[len_C] += 1
        if y_test[i] in C:
            hits += 1
    proportion_of_setlengths = proportion_of_setlengths/n
    emp_coverage = 1/n * hits
    average_C_length = sum([len(prediction_sets[i]) for i in range(n)])/n
    return emp_coverage, average_C_length, qhat, proportion_of_setlengths

def plot_alpha_SCP(alphas,n_runs):
    empirical_coverages = np.zeros_like(alphas)
    average_C_lengths = np.zeros_like(alphas)
    average_quantiles = np.zeros_like(alphas)
    average_amounts = np.zeros((len(alphas), 4))

    for k in range(n_runs):
        print(f'round {k+1} out of {n_runs}')
        normal_cali, normal_test = train_test_split(normal_rest, train_size = 2/3)
        covid_cali, covid_test = train_test_split(covid_rest, train_size = 2/3)
        pneumonia_cali, pneumonia_test = train_test_split(pneumonia_rest, train_size = 2/3)

        X_cali = np.concatenate( (normal_cali, covid_cali, pneumonia_cali), axis = 0)
        y_cali = np.concatenate( (np.full(len(normal_cali), 0), np.full(len(covid_cali), 1), np.full(len(pneumonia_cali), 2)), axis = 0)
        X_test = np.concatenate( (normal_test, covid_test, pneumonia_test), axis = 0)
        y_test = np.concatenate( (np.full(len(normal_test), 0), np.full(len(covid_test), 1), np.full(len(pneumonia_test), 2)), axis = 0)

        for i, alpha in enumerate(alphas):
            empirical_coverage, average_C_length, qhat, proportions_of_setlengths = scp_compute_C(X_test, y_test, X_cali, y_cali, alpha)
            empirical_coverages[i] += empirical_coverage
            average_C_lengths[i] += average_C_length
            average_quantiles[i] += qhat
            average_amounts[i] += proportions_of_setlengths
            
    empirical_coverages = empirical_coverages/n_runs
    average_C_lengths = average_C_lengths/n_runs
    average_quantiles = average_quantiles/n_runs
    average_amounts = average_amounts/n_runs


    plt.subplot(2, 2, 1)
    plt.plot(alphas, empirical_coverages, label = 'Empirical Coverage')
    plt.plot(alphas, 1-alphas, label = r'$1-\alpha$', linestyle='--', color='k', dashes=(5,7), linewidth=1)
    plt.legend(loc = 'upper right')
    plt.title('Empirical Coverage')
    plt.grid()
    plt.title("Empirical Coverage")
    plt.ylim(0.45, 1.1)
    plt.xlabel(r'$\alpha$')

    plt.subplot(2, 2, 2)
    plt.plot(alphas, average_C_lengths)
    plt.title('Average Size of Prediction Set')
    plt.grid()
    plt.ylim(0, 3)
    plt.xlabel(r'$\alpha$')

    plt.subplot(2, 2, 3)
    plt.plot(alphas, average_quantiles)
    plt.title('Conformal Quantile')
    plt.grid()
    plt.ylim(0.1, 1.1)
    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.0])
    plt.xlabel(r'$\alpha$')

    plt.subplot(2, 2, 4)
    plt.plot(alphas, average_amounts[:,0], label = 'Size 0')
    plt.plot(alphas, average_amounts[:,1], label = 'Size 1')
    plt.plot(alphas, average_amounts[:,2], label = 'Size 2')
    plt.plot(alphas, average_amounts[:,3], label = 'Size 3')
    plt.legend(loc = "upper right")
    plt.grid()
    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.0])
    plt.ylim(0, 1)
    plt.xlabel(r'$\alpha$')
    plt.title('Shares of Prediction Set Sizes')

    plt.tight_layout()
    plt.savefig("fig1.pdf", format='pdf')
    plt.show()
    
alphas = np.linspace(0.001, 0.5, 150) 
plot_alpha_SCP(alphas,10)