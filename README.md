# Handwritten Digit Recognition using Convolutional Neural Networks and Kivy GUI

[![Watch the video](https://i.imgur.com/zTjsQO8.png)](https://youtu.be/2qIfCcd8ozk)

This project is aimed to recognize handwritten digits using a Convolutional Neural Network (CNN) model and a Graphical User Interface (GUI) developed using Kivy.

## Requirements
* Python 3.7 or later
* TensorFlow 2.x
* Kivy 2.x
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn

## Dataset
The dataset used for training and testing the CNN model is the MNIST (Modified National Institute of Standards and Technology) dataset, which is a large database of handwritten digits. The dataset consists of 60,000 training images and 10,000 testing images.

## CNN Model
The CNN model used in this project consists of two convolutional layers, each with 32 and 64 filters, respectively, followed by two max-pooling layers and dropout layers to prevent overfitting. The output of the last convolutional layer is then flattened and fed into a fully connected layer with 512 neurons and ReLU activation function, followed by a dropout layer and a final output layer with 10 neurons, each representing a digit from 0 to 9.

The model is trained using the categorical cross-entropy loss function and the RMSprop optimizer. The accuracy of the model is evaluated on a validation set and the learning rate is dynamically reduced using a learning rate reduction callback to improve convergence.

After training, the model is saved in a file named mnist_digit_model in the current directory.

## GUI
The GUI developed using Kivy allows the user to draw a digit using the mouse or touchpad on a 28x28 grid of squares. Each square can be colored with black or white, representing the presence or absence of the digit at that location. The user can then click the "Submit" button to have the CNN model predict the digit drawn.

The predicted digit is displayed on a label below the grid. The user can also click the "Clear" button to erase the drawing and reset the prediction.

## Future Improvments
Improve look of the GUI


# Notes for Running the code:

### Installing Git on Windows

https://www.computerhope.com/issues/ch001927.htm

### Set up SSH key

https://youtu.be/_e4Xf6g_yXg

or

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

### Cloning the repo

##### In git bash run:

git clone git@github.com:sgmdoc7/Handwritten-Digit-Recognizer.git

##### This will copy the repo and files onto your computer


### Install Anaconda

https://youtu.be/5mDYijMfSzs

### From the Anaconda Navigator launch Powershell Prompt and run these commands:
#### Here we are creating a new environment with tensorflow which is needed for this project

conda create -n tensorflow_env tensorflow

conda activate tensorflow_env

conda install -c anaconda keras

conda install -c anaconda numpy

conda install -c anaconda pandas

conda install -c conda-forge matplotlib

conda install kivy -c conda-forge

pip install seaborn

conda install -c anaconda scikit-learn

### Now from the Anaconda Navigator, make sure you have selected the tensorflow_env under
### environments and launch a powershell prompt

run command 'code' to open VS Code in the tensorflow_env 

open folder containing main.py and the mnist_digit_model folder and run main.py

