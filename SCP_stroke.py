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