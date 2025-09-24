This project focuses on classifying bird species using Convolutional Neural Networks (CNN). The dataset is stored in Google Drive and consists of 20 bird classes, split into training and testing folders. To improve model performance, data augmentation techniques like rotation, zoom, flipping, and shifting are applied using TensorFlow’s ImageDataGenerator.

The CNN model is built using tf.keras.Sequential, with three convolutional blocks followed by a fully connected layer and a softmax output layer. It uses the Adam optimizer and categorical crossentropy loss, with early stopping to prevent overfitting. The best model was achieved at epoch 13, with 80% validation accuracy and a validation loss of 0.5322.

Evaluation includes a classification report, confusion matrix, and visualizations of accuracy and loss. The model also supports predictions on new images, successfully identifying species like the African Crowned Crane and American Bittern.
