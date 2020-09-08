import PIL
from PIL import Image
import io

basewidth = 28
baseheight = 28


def resize(path):
    img = Image.open(path)
    img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
    return img


def resizeArr(array):
    img = Image.frombytes('RGB', (600, 800), array, 'raw')
    img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)
    return img
