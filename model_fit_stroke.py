import numpy as np
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
X_df = data2.drop(columns=target_cols)

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
X_train, X_split, Y_train, Y_split = train_test_split(
        X,Y, test_size=0.6,stratify=Y, shuffle=True)

#Fit the model to the traning data
clf = CalibratedClassifierCV(LogisticRegression(class_weight={0: 1.0, 1: 2.0}, C=1, penalty='l1', tol=0.01, solver='saga'), method='sigmoid')
clf=clf.fit(X_train,Y_train)