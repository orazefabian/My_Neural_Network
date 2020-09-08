from kivy import Config
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.core.window import Window
import numpy as np

import My_Neural_Network.image_resizer as resizer

from My_Neural_Network.digit_recognizer import NeuralNetwork

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Window.size = (600, 800)
Window.clearcolor = (1, 1, 1, 1)


def initBrain():
    global model
    model = NeuralNetwork(lr=0.005, use_new_weights=False, save_weights=False)
    model.train_runs(1)


class GUI(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def paint(self):
        global width
        global height
        global dots
        width = 600
        height = 800
        dots = np.zeros((width, height))


class PaintTool(Widget):

    def on_touch_down(self, touch):
        global length, n_points
        with self.canvas:
            Color(0, 0, 0)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=40)
            n_points = 0
            length = 0
            dots[int(touch.x / 2), int(touch.y / 2)] = 255

    def on_touch_move(self, touch):
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x / 2)
            y = int(touch.y / 2)
            touch.ud['line'].width = 40
            dots[int(touch.x / 2) - 40: int(touch.x / 2) + 40, int(touch.y / 2) - 40: int(touch.y / 2) + 40] = 255


class Application(App):
    def build(self):
        initBrain()
        parent = GUI()
        parent.paint()
        self.painter = PaintTool()
        printbtn = Button(text="print")
        clearbtn = Button(text="clear", pos=(parent.width, 0))
        printbtn.bind(on_release=self.printbtn)
        clearbtn.bind(on_release=self.clear)
        parent.add_widget(self.painter)
        parent.add_widget(printbtn)
        parent.add_widget(clearbtn)
        return parent

    def printbtn(self, obj):
        image = resizer.resizeArr(dots)
        image = 255. - np.mean(image, axis=2).reshape(1, -1)
        print(model.predict_number(image))

    def clear(self, obj):
        global dots
        self.painter.canvas.clear()
        dots = np.zeros((width, height))


if __name__ == '__main__':
    Application().run()
