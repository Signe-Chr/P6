import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from load_data_lung import normal_all, covid_all, pneumonia_all
np.random.seed(42)
normal_train, normal_rest = train_test_split(normal_all, train_size = 0.4)
covid_train, covid_rest = train_test_split(covid_all, train_size = 0.4)
pneumonia_train, pneumonia_rest = train_test_split(pneumonia_all, train_size = 0.4)

X_train = np.concatenate((normal_train, covid_train, pneumonia_train), axis = 0)
y_train = np.concatenate((np.full(len(normal_train), 0), np.full(len(covid_train), 1), np.full(len(pneumonia_train), 2)), axis = 0)  

X_rest = np.concatenate((normal_rest, covid_rest, pneumonia_rest), axis = 0)
y_rest = np.concatenate((np.full(len(normal_rest), 0), np.full(len(covid_rest), 1), np.full(len(pneumonia_rest), 2)), axis = 0) 

clf = LogisticRegression(penalty='l1', solver='saga', class_weight='balanced', tol=0.1)
clf.fit(X_train, y_train)