# Handwritten Digit Recognition using Convolutional Neural Network and Kivy GUI

<img src="https://i.imgur.com/UOjS2aI.png" width=75% height=75%>
Demonstration video: https://youtu.be/1o999abiXUw?si=A5xtbF6QhSW6VYUH

## Requirements
* Python 3.9–3.11
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
The GUI developed using Kivy allows the user to draw a digit using the mouse or touchpad on a 28x28 grid of squares. Each square can be colored with black or white, representing the presence or absence of the digit at that location. The CNN model will then predict the digit drawn.

The predicted digit is displayed on a label below the grid. The user can also click the "Clear" button to erase the drawing and reset the prediction.

## Future Improvements Checklist

* Multiple digit recognition
* Handwritten letters or even words

---

## Getting Started

### 1. Clone the repo

```bash
git clone git@github.com:sgmdoc7/Handwritten-Digit-Recognizer.git
cd Handwritten-Digit-Recognizer
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the app

```powershell
python main.py
```

A window will open with a **28×28 drawing grid**. Draw a digit with your mouse and the model will predict it in real time. Click **Clear** to reset.

---

## Retraining the Model (Optional)

The trained model is already included in the repo (`mnist_digit_model/`), so this step is not required. If you'd like to retrain from scratch:

```powershell
python cnn.py
```

This reads from `digit-recognizer/train.csv` and saves a new model to `mnist_digit_model/`.
