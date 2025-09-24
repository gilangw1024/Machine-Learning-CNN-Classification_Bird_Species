# Create CNN Algorithm
cnn = tf.keras.models.Sequential()

# Manage Convolution Layer
cnn.add(tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPooling2D(2))
cnn.add(tf.keras.layers.Dropout(0.1))

cnn.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPooling2D(2))
cnn.add(tf.keras.layers.Dropout(0.1))

cnn.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPooling2D(2))
cnn.add(tf.keras.layers.Dropout(0.1))

from tensorflow.keras.callbacks import EarlyStopping

# Callback for early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

#fully connected layer
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dropout(0.3))

#output layer with 20 categories
cnn.add(tf.keras.layers.Dense(20, activation='softmax'))

cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
