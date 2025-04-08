import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv('C:\\Users\\student\\Desktop\\lV5\\occupancy_processed.csv')
feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']
X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(X[mask, 0], X[mask, 1], label=class_names[class_value])
plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('Zauzetost prostorije')
plt.legend()
plt.show()

# a) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# b) 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# c) 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# d) 
y_pred = knn.predict(X_test_scaled)
# a.
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", conf_matrix)
# b. 
accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost klasifikacije: {accuracy:.4f}")
# c. 
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
print(f"Preciznost po klasama: {precision}")
print(f"Odziv po klasama: {recall}")


#e) Što se događa s rezultatima ako se koristi veći odnosno manji broj susjeda?
# Ako se koristi veći broj susjeda model će biti stabilniji, dok je s manjim brojem susjeda veća preciznost

#f) Što se događa s rezultatima ako ne koristite skaliranje ulaznih veličina?
# Smanjiti će se točnost, te će perfomanse modela opasti jer je udaljenost između točaka nisu pravilno izračunate
