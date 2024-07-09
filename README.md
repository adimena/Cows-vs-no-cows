# Cows-vs-no-cows
![IMG_8659.JPG](./IMG_8659.JPG)
Detection of cows on a Raspberry Pi camera, for citizen scientists collecting data on river turbidity

## Description

What your application does,  
Why you used the technologies you used,  
Some of the challenges you faced and features you hope to implement in the future.  

Neural network which detects whether there are cows or no cows in a photo, using TensorFlow. There are three parts to this:  
* Data
* Model
* Inference  
I achieved an accuracy of 80% using data from the field.

## How to use

Install the requirements in a virtual environment
```
python install -r requirements.txt
```
Download a dataset (please contact me if you wish to use my dataset) with two folders: photos of cows and photos without cows. Use a variety of photos including different angles of the cow(s), and different locations within both catagories.  
Split the data into a training and testing set, using splitdata.py.  
Train the model using cows.py. This script will arrange the data into something it can use, create a neural network, train all of the images in the training set 15 times over, and save the model.  
Test the model using inference.py. This script will load data from the holdout set and feed it into the model. The output will be a number showing the certainty that the "cow" images contain cows, and the "no cow" images do not contain cows.  


## Coming soon

Get it working with the pi camera  
Graphs showing the performance of the model  
Better performance using dropout layers  
