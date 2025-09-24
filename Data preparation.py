from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

#path to drive for training set
training_set = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Dataset/bird classification/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("Count Class:", training_set.num_classes)
print("Class Label:", training_set.class_indices)