import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 1) Prikaz nekoliko slika iz train skupa
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Labela: {y_train[i]}")
    plt.axis("off")
plt.suptitle("Primjeri slika iz MNIST skupa")
plt.tight_layout()
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

# 2) Izgradnja mreže
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# 3) Definiranje procesa učenja
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4) Treniranje mreže
model.fit(x_train_s, y_train_s, epochs=5, batch_size=32, verbose=2)

# 5) Evaluacija mreže
train_loss, train_accuracy = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Točnost na skupu za učenje: {train_accuracy:.4f}")
print(f"Točnost na skupu za testiranje: {test_accuracy:.4f}")

# 6) Matrica zabune za testni i učni skup
y_train_pred = np.argmax(model.predict(x_train_s), axis=1)
y_test_pred = np.argmax(model.predict(x_test_s), axis=1)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
ConfusionMatrixDisplay(cm_train, display_labels=range(10)).plot(ax=plt.gca(), cmap="Blues")
plt.title("Matrica zabune - skup za učenje")

plt.subplot(1, 2, 2)
ConfusionMatrixDisplay(cm_test, display_labels=range(10)).plot(ax=plt.gca(), cmap="Blues")
plt.title("Matrica zabune - skup za testiranje")

plt.tight_layout()
plt.show()

# 7) Prikaz pogrešnih klasifikacija
errors = np.where(y_test != y_test_pred)[0]

plt.figure(figsize=(10, 5))
for i, idx in enumerate(errors[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Prava: {y_test[idx]}, Pred: {y_test_pred[idx]}")
    plt.axis("off")
plt.suptitle("Pogrešno klasificirani primjeri")
plt.tight_layout()
plt.show()
