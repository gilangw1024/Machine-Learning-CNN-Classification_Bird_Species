import matplotlib.pyplot as plt

for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = cnn.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show picture and label
    plt.imshow(img)
    plt.title(f"Prediksi: {predicted_class}")
    plt.axis('off')
    plt.show()