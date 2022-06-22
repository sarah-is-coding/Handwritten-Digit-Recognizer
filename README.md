# Handwritten-Digit-Recognizer

Train a model so that given a picture of a digit the model
will recognize the digit. Can use a neural network and/or regression. Create
a website and mobile app where someone can draw a number and the
model will predict and show what number it is most likely to be and its
confidence.

For training and testing the model, we will use python. For creating the two
UIs we plan to use Kivy to create a desktop application and then package it
for Android.

### Installing Git on Windows

https://www.computerhope.com/issues/ch001927.htm

### Using Git Bash

#### When first adding a file:

git add "filename"

git commit -m "message"

git push

#### When making edits to a file:

git add .

git commit -m "message"

git push

#### Do often to see updates made by other people:

git pull

### Install Anaconda

https://youtu.be/5mDYijMfSzs

### From the anaconda navigator launch Powershell Prompt

### Run these commands:

conda create -n tensorflow_env tensorflow

conda activate tensorflow_env

conda install -c anaconda keras

conda install -c anaconda pandas

conda install -c conda-forge matplotlib

conda install kivy -c conda-forge

pip install seaborn

conda install -c anaconda scikit-learn
