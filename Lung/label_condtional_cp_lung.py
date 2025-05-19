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

def plot_alpha_label(alphas,n_runs):
    emp_cov_normal = np.zeros_like(alphas)
    emp_cov_covid = np.zeros_like(alphas)
    emp_cov_pneu = np.zeros_like(alphas)
    set_size_normal = np.zeros_like(alphas)
    set_size_covid = np.zeros_like(alphas)
    set_size_pneu = np.zeros_like(alphas)
    quantile_normal = np.zeros_like(alphas)
    quantile_covid = np.zeros_like(alphas)
    quantile_pneu = np.zeros_like(alphas)
    share_of_sizes_normal = np.zeros((len(alphas), 4))
    share_of_sizes_covid = np.zeros((len(alphas), 4))
    share_of_sizes_pneu = np.zeros((len(alphas), 4))
    
    for k in range(n_runs):
        print(f'round {k+1} out of {n_runs}')
        
        normal_cali, normal_test = train_test_split(normal_rest, train_size = 2/3)
        covid_cali, covid_test = train_test_split(covid_rest, train_size = 2/3)
        pneumonia_cali, pneumonia_test = train_test_split(pneumonia_rest, train_size = 2/3)

        for i, alpha in enumerate(alphas):
            cov_normal, size_normal, qhat_normal, portions_normal = scp_compute_C(normal_test, np.full(len(normal_test), 0) , normal_cali, np.full(len(normal_cali), 0), alpha)
            cov_covid, size_covid, qhat_covid, portions_covid = scp_compute_C(covid_test, np.full(len(covid_test), 1) , covid_cali, np.full(len(covid_cali), 1), alpha)
            cov_pneu, size_pneu, qhat_pneu, portions_pneu = scp_compute_C(pneumonia_test, np.full(len(pneumonia_test), 2) , pneumonia_cali, np.full(len(pneumonia_cali), 2), alpha)

            emp_cov_normal[i] += cov_normal
            emp_cov_covid[i] += cov_covid
            emp_cov_pneu[i] += cov_pneu
            set_size_normal[i] += size_normal
            set_size_covid[i] += size_covid
            set_size_pneu[i] += size_pneu
            quantile_normal[i] += qhat_normal
            quantile_covid[i] += qhat_covid
            quantile_pneu[i] += qhat_pneu
            share_of_sizes_normal[i] += portions_normal
            share_of_sizes_covid[i] += portions_covid
            share_of_sizes_pneu[i] += portions_pneu

    emp_cov_normal = emp_cov_normal/n_runs
    emp_cov_covid = emp_cov_covid/n_runs
    emp_cov_pneu = emp_cov_pneu/n_runs
    set_size_normal = set_size_normal/n_runs
    set_size_covid = set_size_covid/n_runs
    set_size_pneu = set_size_pneu/n_runs
    quantile_normal = quantile_normal/n_runs
    quantile_covid = quantile_covid/n_runs  
    quantile_pneu = quantile_pneu/n_runs
    share_of_sizes_normal = share_of_sizes_normal/n_runs
    share_of_sizes_covid = share_of_sizes_covid/n_runs
    share_of_sizes_pneu = share_of_sizes_pneu/n_runs

    plt.subplot(3, 2, 1)
    plt.plot(alphas, emp_cov_normal, label= 'Healthy')
    plt.plot(alphas, emp_cov_covid, label = 'Covid-19')
    plt.plot(alphas, emp_cov_pneu, label= 'Pneumonia')
    plt.plot(alphas, 1-alphas, label = r'$1-\alpha$', linestyle='--', color='k', dashes=(5,7), linewidth=1)
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.ylim(0.45, 1.1)
    plt.title('Empirical Coverage')
    plt.xlabel(r'$\alpha$')

    plt.subplot(3, 2, 3)
    plt.plot(alphas, quantile_normal, label= 'Healthy')
    plt.plot(alphas, quantile_covid, label = 'Covid-19')
    plt.plot(alphas, quantile_pneu, label= 'Pneumonia')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.ylim(0.1, 1.1)
    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    plt.title('Conformal Quantile')
    plt.xlabel(r'$\alpha$')

    plt.subplot(3, 2, 5)
    plt.plot(alphas, set_size_normal, label= 'Healthy')
    plt.plot(alphas, set_size_covid, label = 'Covid-19')
    plt.plot(alphas, set_size_pneu, label= 'Pneumonia')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.title('Average Size of Prediction Set')
    plt.xlabel(r'$\alpha$')

    plt.subplot(3, 2, 2)
    plt.plot(alphas, share_of_sizes_normal[:,0], label = 'Size 0')
    plt.plot(alphas, share_of_sizes_normal[:,1], label = 'Size 1')
    plt.plot(alphas, share_of_sizes_normal[:,2], label = 'Size 2')
    plt.plot(alphas, share_of_sizes_normal[:,3], label = 'Size 3')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.ylim(-0.05, 1.05)
    plt.title('Shares of Prediction Set Sizes, Healthy')
    plt.xlabel(r'$\alpha$')

    plt.subplot(3, 2, 4)
    plt.plot(alphas, share_of_sizes_covid[:,0], label = 'Size 0')
    plt.plot(alphas, share_of_sizes_covid[:,1], label = 'Size 1')
    plt.plot(alphas, share_of_sizes_covid[:,2], label = 'Size 2')
    plt.plot(alphas, share_of_sizes_covid[:,3], label = 'Size 3')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.ylim(-0.05, 1.05)
    plt.title('Shares of Prediction Set Sizes, Covid-19')
    plt.xlabel(r'$\alpha$')

    plt.subplot(3, 2, 6)
    plt.plot(alphas, share_of_sizes_pneu[:,0], label = 'Size 0')
    plt.plot(alphas, share_of_sizes_pneu[:,1], label = 'Size 1')
    plt.plot(alphas, share_of_sizes_pneu[:,2], label = 'Size 2')
    plt.plot(alphas, share_of_sizes_pneu[:,3], label = 'Size 3')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.ylim(-0.05, 1.05)
    plt.title('Shares of Prediction Set Sizes, Pneumonia')
    plt.xlabel(r'$\alpha$')

    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.show()
    
alphas = np.linspace(0.005, 0.5, 150) #kan ikke starte på 0, alpha skal være mindst 1/1501 før np.quantile virker
plot_alpha_label(alphas,10)
