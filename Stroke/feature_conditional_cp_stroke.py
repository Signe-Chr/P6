import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_fit_stroke import clf,X_split,Y_split,X_df

np.random.seed(69420)

def bins_2(X,Y,feature):   
    col = X_df.columns.get_loc(feature)
    X_0 = X[X[:, col] == 0]  
    Y_0 = Y[X[:, col] == 0]   

    X_1 = X[X[:, col] == 1]  
    Y_1 = Y[X[:, col] == 1]  
    return X_1, Y_1, X_0, Y_0, col

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

def alpha_feature_plot(alphas,n_runs): 
    empirical_coverages_dia_0 = np.zeros_like(alphas)
    average_C_lengths_dia_0 = np.zeros_like(alphas)
    average_quantiles_dia_0 = np.zeros_like(alphas)
    average_amount_of_0_dia_0 = np.zeros_like(alphas)
    average_amount_of_1_dia_0 = np.zeros_like(alphas)
    average_amount_of_2_dia_0= np.zeros_like(alphas)
    
    empirical_coverages_dia_1 = np.zeros_like(alphas)
    average_C_lengths_dia_1 = np.zeros_like(alphas)
    average_quantiles_dia_1 = np.zeros_like(alphas)
    average_amount_of_0_dia_1 = np.zeros_like(alphas)
    average_amount_of_1_dia_1 = np.zeros_like(alphas)
    average_amount_of_2_dia_1= np.zeros_like(alphas)
    for k in range(n_runs):
            print(f'round {k+1} out of {n_runs}')
            np.random.seed(k + 69)
            X_cali, X_test, Y_cali, Y_test= train_test_split(X_split,Y_split,stratify=Y_split,test_size=0.5,shuffle=True)
            
            X_test_dia_1,Y_test_dia_1,X_test_dia_0,Y_test_dia_0,col5=bins_2(X_test,Y_test, 'Diabetes')
            X_cali_dia_1,Y_cali_dia_1,X_cali_dia_0,Y_cali_dia_0,col5=bins_2(X_cali,Y_cali, 'Diabetes')    
            for i, alpha in enumerate(alphas):
                empirical_coverage_dia_0, average_C_length_dia_0, qhat_dia_0, proportions_of_setlengths_dia_0 = scp_compute_C(X_test_dia_0, Y_test_dia_0, X_cali_dia_0, Y_cali_dia_0, alpha)
                empirical_coverages_dia_0[i]+=empirical_coverage_dia_0
                average_C_lengths_dia_0[i]+=average_C_length_dia_0
                average_quantiles_dia_0[i]+=qhat_dia_0
                average_amount_of_0_dia_0[i]+=proportions_of_setlengths_dia_0[0]
                average_amount_of_1_dia_0[i]+=proportions_of_setlengths_dia_0[1]
                average_amount_of_2_dia_0[i]+=proportions_of_setlengths_dia_0[2]
                
                empirical_coverage_dia_1, average_C_length_dia_1, qhat_dia_1, proportions_of_setlengths_dia_1 = scp_compute_C(X_test_dia_1, Y_test_dia_1, X_cali_dia_1, Y_cali_dia_1, alpha)
                empirical_coverages_dia_1[i] += empirical_coverage_dia_1
                average_C_lengths_dia_1[i] += average_C_length_dia_1
                average_quantiles_dia_1[i] += qhat_dia_1
                average_amount_of_0_dia_1[i] += proportions_of_setlengths_dia_1[0]
                average_amount_of_1_dia_1[i] += proportions_of_setlengths_dia_1[1]
                average_amount_of_2_dia_1[i] += proportions_of_setlengths_dia_1[2]
                
    empirical_coverages_dia_0 = empirical_coverages_dia_0 / n_runs
    average_C_lengths_dia_0 = average_C_lengths_dia_0 / n_runs
    average_quantiles_dia_0 = average_quantiles_dia_0 / n_runs
    average_amount_of_0_dia_0 = average_amount_of_0_dia_0 / n_runs
    average_amount_of_1_dia_0 = average_amount_of_1_dia_0 / n_runs
    average_amount_of_2_dia_0 = average_amount_of_2_dia_0 / n_runs
    
    empirical_coverages_dia_1 = empirical_coverages_dia_1 / n_runs
    average_C_lengths_dia_1 = average_C_lengths_dia_1 / n_runs
    average_quantiles_dia_1 = average_quantiles_dia_1 / n_runs
    average_amount_of_0_dia_1 = average_amount_of_0_dia_1 / n_runs
    average_amount_of_1_dia_1 = average_amount_of_1_dia_1 / n_runs
    average_amount_of_2_dia_1 = average_amount_of_2_dia_1 / n_runs
     
    fig=plt.figure(figsize=(15, 12))
    fig.suptitle("Diabetes", fontsize=20, y=0.98)
    
    plt.subplot(2,2,1)
    plt.plot(alphas,empirical_coverages_dia_0,label='ND')
    plt.plot(alphas,empirical_coverages_dia_1,label='D')
    plt.plot(alphas, 1-alphas, label = r'$1-\alpha$',linestyle='--',color='k',dashes=(5,7),linewidth=1)
    plt.xlabel(r'$\alpha$')
    plt.title('Empirical coverage')
    plt.grid()
    plt.legend(loc = "upper right")
    
    plt.subplot(2,2,2)
    plt.plot(alphas,average_C_lengths_dia_0,label='ND')
    plt.plot(alphas,average_C_lengths_dia_1,label='D')
    plt.xlabel(r'$\alpha$')
    plt.title('Average Size of Prediction Set')
    plt.grid()
    plt.legend(loc = "upper right")
     
    plt.subplot(2,2,3)
    plt.plot(alphas,average_quantiles_dia_0,label='ND')
    plt.plot(alphas,average_quantiles_dia_1,label='D')
    plt.xlabel(r'$\alpha$')
    plt.title(r'Conformal Quantile')
    plt.grid()
    plt.legend(loc = "upper right")
    
    plt.subplot(2,2,4)
    plt.plot(alphas,average_amount_of_0_dia_0,label='ND, Size 0')
    plt.plot(alphas,average_amount_of_1_dia_0,label='ND, Size 1')
    plt.plot(alphas,average_amount_of_2_dia_0,label='ND, Size 2')
    plt.plot(alphas,average_amount_of_0_dia_1,label='D, Size 0')
    plt.plot(alphas,average_amount_of_1_dia_1,label='D, Size 1')
    plt.plot(alphas,average_amount_of_2_dia_1,label='D, Size 2')
    plt.legend(loc = "upper right")
    plt.grid()
    plt.xlabel(r'$\alpha$')
    plt.title('Shares of Prediction Set Sizes')
    
    plt.tight_layout(pad=3.0)
    plt.show()
    
alphas = np.linspace(0.01, 0.5, 150)
alpha_feature_plot(alphas,100)
