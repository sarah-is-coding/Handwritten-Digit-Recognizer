from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.properties import ColorProperty
import numpy as np
import tensorflow as tf
from kivy.clock import Clock
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the pretrained model
model = tf.keras.models.load_model('mnist_digit_model')

# Define the custom widget GridCell
class GridCell(Widget):
    color = ColorProperty('#ffffffff')

    # Handle touch down event
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.color = App.get_running_app().pencil_color
            App.get_running_app().schedule_update_prediction()
            return True
        return super().on_touch_down(touch)

    # Handle touch move event
    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            self.color = App.get_running_app().pencil_color
            App.get_running_app().schedule_update_prediction()
            return True
        return super().on_touch_move(touch)


# Define the Kivy language string for UI elements
KV = '''
#:import random random.random
#:import sp kivy.metrics.sp

<GridCell>:
    canvas:
        Color:
            rgba: self.color
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [sp(4),]

BoxLayout:
    canvas.before:
        Color:
            rgba: 0.9, 0.9, 0.9, 1
        Rectangle:
            pos: self.pos
            size: self.size
    orientation: 'vertical'
    padding: sp(10)
    spacing: sp(10)
    AnchorLayout:
        GridLayout:
            id: grid
            rows: 28
            cols: 28
            spacing: sp(2)
            size_hint: None, None
            width: 100 if not self.parent else min(*self.parent.size)
            height: self.width

    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: None
        height: sp(60)
        spacing: sp(10)
        Button:
            text: 'Clear'
            font_size: sp(24)
            background_color: 0, 0, 0, 0
            background_normal: ''
            color: 1, 1, 1, 1
            size_hint_x: 0.5
            bold: True
            on_release: [setattr(x, 'color', '#ffffffff') for x in grid.children]
            on_release: app.reset_prediction()
            canvas.before:
                Color:
                    rgba: 0.2, 0.6, 0.8, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [sp(10),]

        Label:
            id: prediction
            text: 'Prediction: _'
            font_size: sp(24)
            color: 0.2, 0.2, 0.2, 1
            bold: True
            size_hint_x: 0.5

'''

# Define the main app class
class PixelGridApp(App):
    pencil_color = ColorProperty('#000000')
    model_prediction = 0
    digit_arr = np.empty((0, 784), int)

    # Reset the prediction label text
    def reset_prediction(self):
        self.root.ids.prediction.text = 'Prediction: _'

    # Get the prediction based on the drawn digit
    def update_prediction(self, *args):
        grid_children = self.root.ids.grid.children
        temp_digit_arr = [abs(int(x.color[0]) - 1) for x in grid_children]
        digit_arr = np.append(temp_digit_arr, np.array(temp_digit_arr), axis=0)
        digit_arr = np.flipud(digit_arr)
        digit_arr = digit_arr.reshape(-1, 28, 28, 1)

        model_prediction = np.argmax(model.predict(digit_arr))
        self.root.ids.prediction.text = f'Prediction: {model_prediction}'

    def schedule_update_prediction(self, *args):
        Clock.unschedule(self.update_prediction)
        Clock.schedule_once(self.update_prediction, 0.1)

    # Build the app UI
    def build(self):
        root = Builder.load_string(KV)
        grid = root.ids.grid
        for i in range(grid.rows * grid.cols):
            grid.add_widget(GridCell())
        return root

# Run the app
if __name__ == '__main__':
    PixelGridApp().run()
