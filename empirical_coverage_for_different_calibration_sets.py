import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_fit import clf,X_split,Y_split

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
    hits = 0
    for i in range(n):
        hypothesised_scores = 1-phats[i, categories]
        C = categories[hypothesised_scores <= qhat]
        prediction_sets[i] = C
        if y_test[i] in C:
            hits += 1
    emp_coverage = 1/n * hits
    return emp_coverage

def plot_emp_cov_dif_cali_set(n_c_list,n_runs,alpha):
    all_emp_cov=[]
    for i in range(n_runs):
        print(f'round {i+1} out of {n_runs}')
        np.random.seed(i+69)
        X_cali, X_test, Y_cali, Y_test= train_test_split(X_split,Y_split,stratify=Y_split,test_size=0.2,shuffle=True)
        emp_cov=[]
        for n_c in n_c_list:
            X_cali_nc=X_cali[0:n_c]
            Y_cali_nc=Y_cali[0:n_c]
            empi_cov=scp_compute_C(X_test, Y_test, X_cali_nc, Y_cali_nc, alpha)
            emp_cov.append(empi_cov)
        all_emp_cov.append(emp_cov)
        mean_emp_cov = np.mean(all_emp_cov, axis=0)
        plt.plot(n_c_list,emp_cov)
        plt.title(r'Empirical Coverage for $\alpha=0.1$')
        plt.xlabel(r'Size of Calibration Set, $n_c$')
        plt.grid(True)
    plt.plot(n_c_list,mean_emp_cov,linestyle='--',label='Mean empirical coverage',color='black')
    plt.legend(loc=4)
    
    plt.show()
    
n_c=np.arange(50,2050,50)

plot_emp_cov_dif_cali_set(n_c,10,0.1)