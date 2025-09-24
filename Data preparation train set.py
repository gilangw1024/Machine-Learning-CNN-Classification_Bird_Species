#path to drive for training set
training_set = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Dataset/bird classification/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("Count Class:", training_set.num_classes)
print("Class Label:", training_set.class_indices)