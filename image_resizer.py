import PIL
from PIL import Image

basewidth = 28
baseheight = 28


def resize(path):
    img = Image.open(path)
    img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
    return img


def resizeArr(array):
    img = Image.fromarray(array, mode=None)
    img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
    return img
