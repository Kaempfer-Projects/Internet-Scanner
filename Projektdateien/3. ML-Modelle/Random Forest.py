#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

file = "D:/Undersampling12000/balanced.csv"
df = pd.read_csv(
        file,
        header=0,
        dtype=float,
        sep=";",
        engine='python')

#%%

#Encode Output Class
#data['treffer'] = data['treffer'].astype('category')

#Trennung Labels (Input sind alle Spalten auser die letzte (Treffer) -> X, Output ist Treffer -> Y)
X = df.drop('treffer', axis=1)
y = df['treffer']

from sklearn import model_selection
from sklearn.model_selection import train_test_split

# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)


# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# predictions
rfc_predict = rfc.predict(X_test)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

rfc_cv_score = cross_val_score(rfc, X, y, cv=30, scoring='roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())



