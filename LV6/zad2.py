import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from funkcija_6_1 import generate_data

X = generate_data(n_samples=500, flagc=1)
inertia_values = []

for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.plot(range(1, 21), inertia_values, marker='o')
plt.title("Vrijednost kriterijske funkcije vs broj klastera")
plt.xlabel("Broj klastera")
plt.ylabel("Kriterijska funkcija")
plt.grid(True)
plt.show()
