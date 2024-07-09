
import os

import random
from shutil import copyfile

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dropout


if True:
    def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
        all_files = []

        for file_name in os.listdir(SOURCE):
            file_path = SOURCE + file_name

            if os.path.getsize(file_path):
                all_files.append(file_name)
            else:
                print('{} is zero length, so ignoring'.format(file_name))

        n_files = len(all_files)
        split_point = int(n_files * SPLIT_SIZE)

        shuffled = random.sample(all_files, n_files)

        train_set = shuffled[:split_point]
        test_set = shuffled[split_point:]

        for file_name in train_set:
            copyfile(SOURCE + file_name, TRAINING + file_name)

        for file_name in test_set:
            copyfile(SOURCE + file_name, TESTING + file_name)


    COW_SOURCE_DIR = "./cowsdata2.1/cows/"
    TRAINING_COW_DIR = "./cow-v-nocow2/training/cow/"
    TESTING_COW_DIR = "./cow-v-nocow2/testing/cow/"
    NOCOW_SOURCE_DIR = "./cowsdata2.1/nocows/"
    TRAINING_NOCOW_DIR = "./cow-v-nocow2/training/nocow/"
    TESTING_NOCOW_DIR = "./cow-v-nocow2/testing/nocow/"

    split_size = .9
    print(COW_SOURCE_DIR)
    split_data(COW_SOURCE_DIR, TRAINING_COW_DIR, TESTING_COW_DIR, split_size)
    split_data(NOCOW_SOURCE_DIR, TRAINING_NOCOW_DIR, TESTING_NOCOW_DIR, split_size)
     
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), input_shape=(150, 150, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
#        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
#        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
#        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])


TRAINING_DIR = './cow-v-nocow2/training/'
train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=64,
        class_mode='binary',
        target_size=(150, 150)
    )

VALIDATION_DIR = './cow-v-nocow2/testing/'
validation_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest'

    )
validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        batch_size=64,
        class_mode='binary',
        target_size=(150, 150)
    )


import warnings
from PIL import Image as pil_image

#warnings.filterwarnings('ignore')
#print('warnings ignored')

history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator)

model.save('./cowdropout.keras')
