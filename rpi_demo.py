"""
Demonstrate that the tflite model works on the Pi 5
Install libhdf5-dev for keras to work
sudo apt-get install libhdf5-dev
"""

import tflite_runtime.interpreter as tflite
import keras
import numpy as np

from keras.preprocessing import image


interpreter = tflite.Interpreter('./model.tflite')
interpreter.allocate_tensors()
FNAME = "cow.jpg" # test file

# load test file
img = image.load_img(FNAME, target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# load model
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], x)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
output_index = np.argmax(output_data[0])
result = output_data < 0.5

if result:
    print('COW DETECTED')
else:
    print('NO COW DETECTED')
