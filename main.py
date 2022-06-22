import tensorflow as tf

from random import random 
from kivy.app import App 
from kivy.uix.widget import Widget 
from kivy.uix.button import Button 
from kivy.graphics import Color, Ellipse, Line
from kivy.core.window import Window

import numpy as np

model = tf.keras.models.load_model('mnist_digit_model')

global inputmatrix
inputmatrix = np.zeros((28,28), dtype=np.int0)

class MyPaintWidget(Widget): 
    def on_touch_down(self, touch): 
        with self.canvas: 
            Color(255,255,255) 
            d = 30. 
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d)) 
            touch.ud['line'] = Line(points=(touch.x, touch.y), width =20)
            self.update_output(touch)

    def on_touch_move(self, touch): 
        touch.ud['line'].points += [touch.x, touch.y]
        self.update_output(touch)

    def update_output(self, touch):
        x_ind = int((touch.x/Window.size[0])*28)
        y_ind = int((touch.y/Window.size[1])*28)
        inputmatrix[28-y_ind][x_ind] = 1

        grid = inputmatrix.reshape(-1, 28 * 28)*255
        print(np.argmax(model.predict(grid)))

class MyPaintApp(App): 
    def build(self): 
        Window.size = (600,600)

        parent = Widget() 
        self.painter = MyPaintWidget() 
        clearbtn = Button(text='Clear', pos=(0,0), size=(50,50)) 
        clearbtn.bind(on_release=self.clear_canvas) 
        parent.add_widget(self.painter) 
        parent.add_widget(clearbtn) 

        printinput = Button(text='Print', pos=(50,0), size=(50,50))
        global inputmatrix
        printinput.bind(on_release=self.print_input) 
        parent.add_widget(printinput) 

        return parent 

    def clear_canvas(self, obj): 
        self.painter.canvas.clear()

        global inputmatrix
        inputmatrix = np.zeros((28,28), dtype=np.int0)

    def print_input(self, obj):
        global inputmatrix
        print(inputmatrix)


if __name__ == '__main__': 
    MyPaintApp().run()