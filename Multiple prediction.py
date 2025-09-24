import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Prediction pictures path
folder_path = '/content/drive/MyDrive/Dataset/bird classification/images to predict'

# Take labels from training set
class_labels = list(training_set.class_indices.keys())

# Loop files
for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)

    # Load and preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi

    # prediction
    prediction = cnn.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # show result
    print(f"Gambar: {img_name} → Prediksi: {predicted_class}")