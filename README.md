# Cows-vs-no-cows  

Detection of cows on a Raspberry Pi camera, for citizen scientists collecting data on river turbidity
![GitHub_cover_image.JPG](./GitHub_cover_image.JPG)

## Description  

Neural network which detects whether there are cows or no cows in a photo, using TensorFlow. There are four parts to this:  
* Data
* Model
* Inference test
* Application

![system_diagram.png](./system_diagram.png)

## How to use

I recommend using two different virtual environments: one for training and one for inference.
```
pip install -r training_requirements.txt
```
Download a dataset (please contact me if you wish to use my dataset) with two folders: photos of cows and photos without cows. Use a variety of photos including different angles of the cow(s), and different locations within both catagories.  
Split the data into a training and testing set, using splitdata.py.  
Train the model using cows.py. This script will arrange the data into something it can use, create a neural network, train all of the images in the training set 15 times over, and save the model.  
```
pip install -r inference_requirements.txt
```
Test the model using inference.py. This script will load data from the holdout set and feed it into the model. The output will be two numbers, showing the certainty that the "cow" images contain cows, and the "no cow" images do not contain cows.  
Test the model in the field using a camera by running photos.py. This will take photos (number of photos currently set to 100) then put them through the model. The output of the script is a csv file called results.csv which contains the timestamp of each photo and whether cows have been detected in the photo or not.
**TensorFow Lite**  
Use converter.py to convert the model to TensorFlow Lite. To do the inference, run inference_tflite.py. To test the model taking photos in real time, use photos_tflite.py.

## Context

For a school project (EPQ) of my choice I chose to make a machine learning algorithm. This is still a work in progress. I decided to make a neural network to detect the presence of cows in fields, to see if this has an effect on the water in rivers nearby. I used another project on GitHub that recognises whether there is a cat or a dog in the photo (https://github.com/0sparsh2/cats-vs-dogs-coursera-assignment/blob/main/Cats_vs_Dogs_CourseraAssignment.ipynb). Instead of training the model with the "cats vs dogs" dataset I used my own "cows vs no cows" dataset. I then took photos of cows myself to use as a holdout set, which the model recognises to an accuracy of up to 80%.  

My sources of data are:
* scraping from Bing image search
* a Kaggle dataset of cows
* videos I took of cows and no cows, then extracting images from here. This was to get a large number of my own photos quickly without taking each one individually
* photos I took of cows and no cows for the holdout set. Less overall than from videos but they are more varied

## Coming soon

Better performance using dropout layers  
Graphs showing the performance of the model  
Using the model on a Raspberry Pi camera with TensorFlow Lite
