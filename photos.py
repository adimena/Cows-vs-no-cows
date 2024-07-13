import os
from datetime import datetime
import time

import keras
import numpy as np
import pandas as pd

from keras.preprocessing import image


model = keras.models.load_model('./cowdropout.keras')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

resultsdict = {'timestamp': [], 'cows': []}

for i in range(2):
	timenow = datetime.now().isoformat()
	resultsdict['timestamp'].append(timenow)
	cmd = "fswebcam --device /dev/video2 -r 640x480 --jpeg 85 -D 1 photo.jpg"
	os.system(cmd)
	fname = "photo.jpg"

	img = image.load_img(fname, target_size=(150,150))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
        
	images = np.vstack([x])
	classes = model.predict(images)
	result = classes[0][0] > 0.5
	resultsdict['cows'].append(result)

	print(result)
	
	df = pd.DataFrame(resultsdict)
	df.to_csv('results.csv', index=False)
	
	time.sleep(10)
