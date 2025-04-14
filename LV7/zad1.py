import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(str(y_train[i]))
    plt.axis("off")
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


# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# TODO: provedi treniranje mreze pomocu .fit()
model.fit(x_train_s, y_train_s, epochs=5, batch_size=32, verbose=2)


# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Točnost - učenje: {train_acc:.4f}")
print(f"Točnost - test: {test_acc:.4f}")


# TODO: Prikazite matricu zabune na skupu podataka za testiranje

y_train_pred = np.argmax(model.predict(x_train_s), axis=1)
y_test_pred = np.argmax(model.predict(x_test_s), axis=1)
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
ConfusionMatrixDisplay(cm_train).plot(ax=plt.gca(), cmap="Blues", colorbar=False)
plt.title("Matrica zabune - učenje")

plt.subplot(1, 2, 2)
ConfusionMatrixDisplay(cm_test).plot(ax=plt.gca(), cmap="Blues", colorbar=False)
plt.title("Matrica zabune - test")
plt.tight_layout()
plt.show()


# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala
errors = np.where(y_test != y_test_pred)[0]

plt.figure(figsize=(10, 4))
for i, idx in enumerate(errors[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Prava: {y_test[idx]}, Pred: {y_test_pred[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
