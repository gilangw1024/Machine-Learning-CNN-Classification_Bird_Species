history = cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=30,
    callbacks=[early_stop]
)