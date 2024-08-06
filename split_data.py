import os



to_create = [
    './cow-v-nocow2',
    './cow-v-nocow2/training',
    './cow-v-nocow2/testing',
    './cow-v-nocow2/training/cow',
    './cow-v-nocow2/training/nocow',
    './cow-v-nocow2/testing/cow',
    './cow-v-nocow2/testing/nocow'
]

for directory in to_create:
    try:
        os.mkdir(directory)
        print(directory, 'created')
    except:
        print(directory, 'failed')
