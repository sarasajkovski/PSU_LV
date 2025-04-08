# a)
from sklearn.model_selection import train_test_split
X = df[['S3_Temp', 'S5_CO2']].to_numpy()
y = df['Room_Occupancy_Count'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# b)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# c)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# d)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
y_pred = knn.predict(X_test_scaled)
# Matrica zabune
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", conf_matrix)
# Točnost klasifikacije
accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost: {accuracy:.4f}")
# Preciznost i odziv po klasama
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
print("Preciznost po klasama:", precision)
print("Odziv po klasama:", recall)
# TOČNOST: 95.56%

#e) Što se događa s rezultatima ako se koristi veći odnosno manji broj susjeda?
# Ako se koristi veći broj susjeda model će biti stabilniji, dok je s manjim brojem susjeda veća preciznost

#f) Što se događa s rezultatima ako ne koristite skaliranje ulaznih veličina?
# Smanjiti će se točnost, te će perfomanse modela opasti jer je udaljenost između točaka nisu pravilno izračunate
