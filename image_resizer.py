import PIL
from PIL import Image

basewidth = 28
baseheight = 28


def reisze(path):
    img = Image.open(path)
    img = img.resize((basewidth, baseheight), PIL.Image.ANTIALIAS)

    return img
