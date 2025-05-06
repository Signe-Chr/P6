import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


np.random.seed(69420)

data = pd.read_csv(r"C:/Users/Signe Christensen/Downloads/Aalborg universitet/Matematik teknologi/6.semester/Projekt/stroke/stroke_data.csv")

categorical_cols = data.select_dtypes(include=["object", "category", "string"]).columns
numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
target_cols = ["Stroke"]

#Transform string features into discrete numeric features
enc = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
encoded_data = enc.fit_transform(data[categorical_cols])

data2 = pd.concat([data, encoded_data], axis=1).drop(columns=categorical_cols.to_list())

X = pd.concat([data, encoded_data], axis=1).drop(columns=categorical_cols.to_list()+target_cols).to_numpy()
Y = data[target_cols].to_numpy().flatten()

#Remove possible non nan indicies in the data
non_nan_indices = ~np.isnan(X).any(axis=1)
X = X[non_nan_indices]
Y = Y[non_nan_indices]

#Make balanced dataset
label_0_indices = np.where(Y == 0)[0]
label_1_indices = np.where(Y == 1)[0]
sampled_label_0_indices = np.random.choice(label_0_indices, size=len(label_1_indices), replace=False)
balanced_indices = np.concatenate([sampled_label_0_indices, label_1_indices])
np.random.shuffle(balanced_indices)
X = X[balanced_indices]
Y = Y[balanced_indices]

#Split the data into training and test data
X_train, X_split, Y_train, Y_split = train_test_split(X,Y, test_size=0.6,stratify=Y, shuffle=True)

#Fit the model to the traning data
clf = CalibratedClassifierCV(LogisticRegression(class_weight={0: 1.0, 1: 2.0}, C=1, penalty='l1', tol=0.01, solver='saga'), method='sigmoid')
clf.fit(X_train,Y_train)

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