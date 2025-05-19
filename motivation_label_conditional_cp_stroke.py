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
    unique_classes, counts = np.unique(y_test, return_counts=True)
    qhat = scp_conformal_quantile(X_cali, y_cali, alpha)
    n = len(X_test)
    n0=counts[0]
    n1=counts[1]
    prediction_sets = np.zeros(n, dtype = object) 
    phats = clf.predict_proba(X_test)
    hits = 0
    hits_0=0
    hits_1=0
    for i in range(n):
        hypothesised_scores = 1-phats[i, categories]
        C = categories[hypothesised_scores <= qhat]
        prediction_sets[i] = C
        if y_test[i] in C:
            hits += 1
            if y_test[i]==0:
                hits_0+=1
            elif y_test[i]==1:
                hits_1+=1
    emp_coverage_overall = 1/n * hits
    emp_coverage_0=1/n0*hits_0
    emp_coverage_1=1/n1*hits_1
    emp_coverage=np.array([emp_coverage_0,emp_coverage_1,emp_coverage_overall])
    return emp_coverage

def plot_alpha_label_motivation(alphas,n_runs):
    empirical_coverages_0=np.zeros_like(alphas)
    empirical_coverages_1=np.zeros_like(alphas)
    for k in range(n_runs):
        print(f'round {k+1} out of {n_runs}')
        np.random.seed(k + 69)
        X_cali, X_test, Y_cali, Y_test= train_test_split(X_split,Y_split,stratify=Y_split,test_size=0.5,shuffle=True)
        for i,alpha in enumerate(alphas):
            emp_coverage=scp_compute_C(X_test,Y_test,X_cali,Y_cali,alpha)
            empirical_coverages_0[i]+=emp_coverage[0]
            empirical_coverages_1[i]+=emp_coverage[1]
    
    empirical_coverages_0=empirical_coverages_0/n_runs
    empirical_coverages_1=empirical_coverages_1/n_runs
    
    plt.plot(alphas,empirical_coverages_0,label='No Stroke')
    plt.plot(alphas,empirical_coverages_1,label='Stroke')
    plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
    plt.xlabel(r'$\alpha$')
    plt.title('Empirical Coverage')
    plt.grid()
    plt.legend(loc = "upper right")
    plt.show()
    
alphas = np.linspace(0.01, 0.5, 150)
plot_alpha_label_motivation(alphas,10)