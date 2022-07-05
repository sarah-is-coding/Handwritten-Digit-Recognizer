from kivy.app import App 

from kivy.lang import Builder 

from kivy.uix.widget import Widget 

from kivy.properties import ColorProperty 


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

        GridCell: 

            color: app.pencil_color  

        Button: 

            text: 'Clear' 

            on_release: [setattr(x, 'color', '#ffffffff') for x in grid.children] 

        Button: 

            text: 'Submit' 

            on_release: grid.export_to_png('out.png') 

            on_release: [app.digit_arr.append(abs(int(x.color[0])-1)) for x in grid.children]

            on_release: print(app.digit_arr)

''' 

  
class PixelGridApp(App):

    pencil_color = ColorProperty('#000000') 
    
    digit_arr = []
  

    def build(self): 

        root = Builder.load_string(KV) 

        grid = root.ids.grid 

        for i in range(grid.rows * grid.cols): 

            grid.add_widget(GridCell()) 

        return root 

  

if __name__ == '__main__': 

    PixelGridApp().run()
