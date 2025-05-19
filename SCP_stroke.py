import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_fit_stroke import clf,X_split,Y_split

np.random.seed(69420)

def scp_conformal_quantile(X_cali, y_cali, alpha):
    phats = clf.predict_proba(X_cali)
    scores = 1 - phats[np.arange(len(y_cali)), y_cali]
    qhat = np.quantile(scores, (1-alpha)*(1+1/(len(y_cali))))
    return qhat

def scp_compute_C(X_test, y_test, X_cali, y_cali, alpha, categories=np.array([0, 1])):
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
        average_amount_of_0 = np.zeros_like(alphas)
        average_amount_of_1 = np.zeros_like(alphas)
        average_amount_of_2 = np.zeros_like(alphas)
        for k in range(n_runs):
                print(f'round {k+1} out of {n_runs}')
                np.random.seed(k + 69)
                X_cali, X_test, Y_cali, Y_test= train_test_split(X_split,Y_split,stratify=Y_split,test_size=0.5,shuffle=True)
                for i, alpha in enumerate(alphas):
                        empirical_coverage, average_C_length, qhat, proportions_of_setlengths = scp_compute_C(X_test, Y_test, X_cali, Y_cali, alpha)
                        empirical_coverages[i] += empirical_coverage
                        average_C_lengths[i] += average_C_length
                        average_quantiles[i] += qhat
                        average_amount_of_0[i] += proportions_of_setlengths[0]
                        average_amount_of_1[i] += proportions_of_setlengths[1]
                        average_amount_of_2[i] += proportions_of_setlengths[2]

        empirical_coverages = empirical_coverages/n_runs
        average_C_lengths = average_C_lengths/n_runs
        average_quantiles = average_quantiles/n_runs
        average_amount_of_0 = average_amount_of_0/n_runs
        average_amount_of_1 = average_amount_of_1/n_runs
        average_amount_of_2 = average_amount_of_2/n_runs

        plt.subplot(2, 2, 1)
        plt.plot(alphas, empirical_coverages, label = 'Empirical Coverage')
        plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
        plt.legend(loc = "upper right")
        plt.title('Empirical Coverage')
        plt.grid()
        plt.xlabel(r'$\alpha$')

        plt.subplot(2, 2, 2)
        plt.plot(alphas, average_C_lengths)
        plt.grid()
        plt.xlabel(r'$\alpha$')
        plt.title('Average Size of Prediction Set')

        plt.subplot(2, 2, 3)
        plt.plot(alphas, average_quantiles)
        plt.grid()
        plt.xlabel(r'$\alpha$')
        plt.title(r'Conformal Quantile')

        plt.subplot(2, 2, 4)
        plt.plot(alphas, average_amount_of_0, label = 'Size 0')
        plt.plot(alphas, average_amount_of_1, label = 'Size 1')
        plt.plot(alphas, average_amount_of_2, label = 'Size 2')
        plt.legend(loc = "upper right")
        plt.grid()
        plt.xlabel(r'$\alpha$')
        plt.title('Shares of Prediction Set Sizes')

        plt.tight_layout(pad=0.2)
        plt.show()
        
alphas = np.linspace(0.01, 0.5, 150)
plot_alpha_SCP(alphas,10)