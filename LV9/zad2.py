import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing import image_dataset_from_directory


train_ds = image_dataset_from_directory(
    directory='C:\\Users\\Sara\\Desktop\\LV9\\Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset="training",
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

validation_ds = image_dataset_from_directory(
    directory='C:\\Users\\Sara\\Desktop\\LV9\\Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset="validation",
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

test_ds = image_dataset_from_directory(
    directory='C:\\Users\\Sara\\Desktop\\LV9\\Test',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48)
)

# definicija CNN modela
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')  # 43 klasa GTSRB skupa
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# spremanje najboljeg modela + TensorBoard
checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=15,
    callbacks=[checkpoint_cb, tensorboard_cb]
)


best_model = load_model("best_model.h5")

# evaluacija na testnom skupu
loss, accuracy = best_model.evaluate(test_ds)
print(f"\nTočnost na testnom skupu: {accuracy * 100:.2f}%")

# matrica zabune
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = best_model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_true, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrica zabune")
plt.show()