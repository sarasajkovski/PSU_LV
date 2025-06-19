from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from funkcija_6_1 import generate_data

X = generate_data(n_samples=500, flagc=1)

# Različite metode: 'single', 'complete', 'average', 'ward'
Z = linkage(X, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendogram hijerarhijskog grupiranja")
plt.xlabel("Uzorci")
plt.ylabel("Udaljenost")
plt.show()
