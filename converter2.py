import tensorflow as tf

import keras

model = keras.models.load_model('./cnncow3.keras')

# Export the keras model to a saved model format
model.export("saved_model")

# Convert the saved model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
