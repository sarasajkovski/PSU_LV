from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv('C:\\Users\\student\\Desktop\\lV5\\occupancy_processed.csv')
X = df[['S3_Temp', 'S5_CO2']]
y = df['Room_Occupancy_Count']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(max_depth=2, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("Matrica zabune:\n", confusion_matrix(y_test, y_pred))
print("Točnost: ", accuracy_score(y_test, y_pred))
print("Preciznost: ", precision_score(y_test, y_pred))
print("Odziv: ", recall_score(y_test, y_pred))

# a) Vizualizacija stabla odlučivanja
plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=['S3_Temp', 'S5_CO2'], class_names=['Slobodna', 'Zauzeta'])
plt.title('Stablo odlučivanja (max_depth=2)')
plt.show()

# b) Što se događa s rezultatima ako mijenjate parametar max-depth stabla odlučivanja? 
#Promjenom vrijednosti max_depth kontroliramo dubinu stabla, odnosno koliko složene odluke model može donositi

# c) Što se događa s rezultatima ako ne koristite skaliranje ulaznih veličina?
# Rezultati ostaju skoro isti
