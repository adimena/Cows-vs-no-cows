"""
Inference for tflite model
"""

import glob
import numpy as np
import tensorflow as tf

from keras.preprocessing import image


interpreter = tf.lite.Interpreter('./model.tflite')
interpreter.allocate_tensors()


for item in ['cow','nocow']:
    print(item)
    print("-"*10)
    data = []
    for idx, fname in enumerate(glob.glob(f'./camdata/{item}/*.JPG')):
        img = image.load_img(fname, target_size=(150,150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        data.append(x)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_index = np.argmax(output_data[0])
        print(output_data < 0.5)
