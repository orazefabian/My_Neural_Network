from kivy import Config
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.core.window import Window
import numpy as np
import os

from brain import NeuralNetwork
import image_resizer as resizer

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


class PaintTool(Widget):

    def on_touch_down(self, touch):
        global length, n_points
        with self.canvas:
            Color(0, 0, 0)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=50)
            n_points = 0
            length = 0

    def on_touch_move(self, touch):
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            touch.ud['line'].width = 50


def removeFiles():
    for file in os.listdir('.'):
        if file.startswith('digit'):
            os.remove(file)


class Application(App):
    def build(self):
        initBrain()
        parent = GUI()
        parent.paint()
        self.painter = PaintTool()
        readbtn = Button(text="read")
        clearbtn = Button(text="clear", pos=(parent.width, 0))
        self.resultlbl = Label(text="", pos=(parent.width + 800, 10), font_size=100,
                               markup=True)
        readbtn.bind(on_release=self.readbtn)
        clearbtn.bind(on_release=self.clear)
        parent.add_widget(self.painter)
        parent.add_widget(readbtn)
        parent.add_widget(clearbtn)
        parent.add_widget(self.resultlbl)
        return parent

    def readbtn(self, obj):
        Window.screenshot(name='digit.png')
        image = resizer.resize(f"digit0001.png")
        image = 255. - np.mean(image, axis=2).reshape(1, -1)
        prediction = model.predict_number(image)
        print(prediction)
        self.resultlbl.text = f"[color=3333ff][b]result: {prediction}[/b][/color]"
        removeFiles()

    def clear(self, obj):
        self.painter.canvas.clear()
        self.resultlbl.text = ""
        removeFiles()


if __name__ == '__main__':
    Application().run()
