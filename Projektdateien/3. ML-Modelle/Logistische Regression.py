import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.DataFrame()

#Input
# Daten Laden
file = "D:/Undersampling12000/balanced.csv"
df = pd.read_csv(
        file,
        header=0,
        dtype=float,
        sep=";",
        engine='python')


#Encode Output Class
df['treffer'] = df['treffer'].astype('category')

#Trennung Labels (Input sind alle Spalten auser die letzte (Treffer) -> X, Output ist Treffer -> Y)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

#Split in Train- und Testset
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=69)


lr=LogisticRegressionCV(max_iter=5000,random_state=1)
model=lr.fit(x_train,y_train)
y_train_pred=model.predict(x_train)
y_val_pred=model.predict(x_val)
print(accuracy_score(y_train,y_train_pred))
print(accuracy_score(y_val,y_val_pred))

#confusion matrix
print(confusion_matrix(y_val, y_val_pred))

#Classification Report
print(classification_report(y_val, y_val_pred))