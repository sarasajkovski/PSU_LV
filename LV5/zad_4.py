import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\student\\Desktop\\lV5\\occupancy_processed.csv')
X = df[['S3_Temp', 'S5_CO2']]
y = df['Room_Occupancy_Count']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print("Matrica zabune:\n", confusion_matrix(y_test, y_pred))
print("Točnost:", accuracy_score(y_test, y_pred))
print("Preciznost:", precision_score(y_test, y_pred))
print("Odziv:", recall_score(y_test, y_pred))

# Primjećujemo da taj model prikazuje visoku točnost
# Uzrok dobivenih rezultata je to što je logistička regresija linearan model i podaci su jednostavniji
