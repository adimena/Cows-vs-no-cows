"""
Inference for my model
"""

import glob
import keras
import numpy as np

from keras.preprocessing import image


model = keras.models.load_model('./cowdropout.keras')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


NO_TESTS = 10000

for item in ['cow','nocow']:
    print(item)
    print("-"*10)
    data = []
    for idx, fname in enumerate(glob.glob(f'./camdata/{item}/*.JPG')):
        img = image.load_img(fname, target_size=(150,150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        data.append(x)
        if idx > NO_TESTS:
            break
    images = np.vstack(data)
    classes = model.predict(images)#, batch_size=10)
    results = []
    for c in classes:
        result = c[0] > 0.5
        results.append(result)
    print(sum(results)/len(results))
