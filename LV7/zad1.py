import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Učitavanje MNIST skupa podataka
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 1) 
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(str(y_train[i]))
    plt.axis("off")
plt.tight_layout()
plt.show()

# oblikovanje podataka
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255
x_train_s = x_train_s.reshape(-1, 784)
x_test_s = x_test_s.reshape(-1, 784)
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)


# 2) Izgradnja mreže
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.summary()


# 3) Kompilacija i treniranje mreže
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train_s, y_train_s, epochs=5, batch_size=32, verbose=2)

train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Točnost - učenje: {train_acc:.4f}")
print(f"Točnost - test: {test_acc:.4f}")


# 4) Matrice zabune
y_train_pred = np.argmax(model.predict(x_train_s), axis=1)
y_test_pred = np.argmax(model.predict(x_test_s), axis=1)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
ConfusionMatrixDisplay(cm_train).plot(ax=plt.gca(), cmap="Blues", colorbar=False)
plt.title("Učenje")
plt.subplot(1, 2, 2)
ConfusionMatrixDisplay(cm_test).plot(ax=plt.gca(), cmap="Blues", colorbar=False)
plt.title("Test")
plt.tight_layout()
plt.show()
