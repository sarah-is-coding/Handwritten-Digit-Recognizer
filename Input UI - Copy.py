from kivy.app import App 

from kivy.lang import Builder 

from kivy.uix.widget import Widget 

from kivy.properties import ColorProperty 

import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('mnist_digit_model')


class GridCell(Widget): 

    color = ColorProperty('#ffffffff') 


    # collides on press or drag (_move) 

    def on_touch_down(self, touch): 

        if self.collide_point(*touch.pos): 
            self.color = App.get_running_app().pencil_color 

            return True 

        return super().on_touch_down(touch) 

  

    def on_touch_move(self, touch): 

        if self.collide_point(*touch.pos): 
            self.color = App.get_running_app().pencil_color 

            return True 

        return super().on_touch_move(touch) 

class Buttons(Widget):

    def press():
        print('hello')

  

KV = ''' 

#:import random random.random 

# Draw rectangle at cell size/position using current color 

<GridCell>: 

    canvas: 

        Color:

            rgba: self.color 

        Rectangle: 

            pos: self.pos 

            size: self.size 


BoxLayout: 

    orientation: 'vertical' 

    # The AnchorLayout centers the grid, plus it serves to determine 

    # the available space via min(*self.parent.size) below 

    AnchorLayout: 

        GridLayout: 

            id: grid 

            rows: 28 

            cols: 28 

            size_hint: None, None 

            # Use the smallest of width/height to make a square grid. 

            # The "if not self.parent" condition handles parent=None 

            # during initial construction, it will crash otherwise 

            width: 100 if not self.parent else min(*self.parent.size) 

            height: self.width 
    
    BoxLayout: 

        orientation: 'horizontal' 

        size_hint_y: None 

        Button: 

            text: 'Clear' 

            on_release: [setattr(x, 'color', '#ffffffff') for x in grid.children] 

            on_release: app.reset_prediction()

        Label:

            id: prediction

            text: 'Prediction: _'

        Button: 

            text: 'Submit' 

            on_release: app.get_prediction(grid.children)


''' 

class PixelGridApp(App):

    pencil_color = ColorProperty('#000000') 
    model_prediction = 0
    digit_arr = np.empty((0,784), int)
    
    def reset_prediction(self):

        self.root.ids.prediction.text = 'Prediction: _'

    def get_prediction(self, grid_children):

        temp_digit_arr = [abs(int(x.color[0])-1) for x in grid_children]
        digit_arr = np.append(temp_digit_arr, np.array(temp_digit_arr), axis=0)
        digit_arr = np.flipud(digit_arr)
        digit_arr = digit_arr.reshape(-1,28,28,1)

        model_prediction = np.argmax(model.predict(digit_arr))

        self.root.ids.prediction.text = f'Prediction: {model_prediction}'

    def build(self): 

        root = Builder.load_string(KV) 

        grid = root.ids.grid 

        for i in range(grid.rows * grid.cols): 
            grid.add_widget(GridCell()) 

        return root 

  
if __name__ == '__main__': 
    PixelGridApp().run()
