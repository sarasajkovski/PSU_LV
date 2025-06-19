import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# istrenirani model
model = load_model("best_model.h5")


img_path = "provjera.png" 
img_size = (48, 48)

try:
    img = image.load_img(img_path, target_size=img_size)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Ulazna slika")
    plt.show()

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    print(f"Predviđena klasa prometnog znaka: {predicted_class}")

except FileNotFoundError:
    print(f"Slika '{img_path}' nije pronađena. Provjeri putanju.")
