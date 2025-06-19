import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

image = mpimg.imread('example.png')
h, w, d = image.shape
image_array = image.reshape(-1, d)

kmeans = KMeans(n_clusters=16, n_init=10)
labels = kmeans.fit_predict(image_array)
new_colors = kmeans.cluster_centers_.astype('uint8')
compressed_image = new_colors[labels].reshape(h, w, d)

plt.figure()
plt.imshow(image)
plt.title("Originalna slika")

plt.figure()
plt.imshow(compressed_image)
plt.title("Kvantizirana slika (16 klastera)")
plt.show()
