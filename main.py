import numpy as np
from PIL import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import math

size_multiplier = 20

fontsize = 10 * size_multiplier

image_size = (10 * size_multiplier, 7 * size_multiplier)

angles_size = (2, 1)

spec_size = (image_size[0] + angles_size[0] - 1, image_size[1] + angles_size[1] - 1)


def getImageNumber(number):
    image = Image.new("RGBA", (spec_size[1], spec_size[0]), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("FreeMono.ttf", fontsize)
    draw.text((10, 0), str(number), (0, 0, 0), font=font)

    # image.show()

    array = np.asarray(image.convert('L'))
    array = array.astype(np.float)
    array = array / 255 * -.25 + .75
    return array


def getImageMovingContrast(number):
    array = np.ones(spec_size, dtype=np.float) * .5
    bound = (number + 1) / (angles_size[0] + 1)
    array[int(bound  * spec_size[0]):,:] = .75
    return array

def getImageHomogeneous(number):
    color = .5 if number % 2 == 0 else 0
    array = np.ones(spec_size, dtype=np.float) * color
    return array

def getImage(number):
    # return getImageHomogeneous(0)
    return getImageNumber(number)

def showImg(array, scaling=True):
    mmin = 0
    mmax = 1
    if scaling:
        min_ = np.min(array)
        max_ = np.max(array)
        if min_ != max_:
            mmin = min_
            mmax = max_
    image = Image.fromarray((array - mmin) / (mmax - mmin) * 255)
    image.show()

def saveImg(array, filename, scaling=False):
    array_ = array
    if scaling:
        min_ = np.min(array)
        max_ = np.max(array)
        if min_ != max_:
            array_ = (array - min_) / (max_ - min_)
    rgba = np.zeros(image_size + (4,), dtype = np.uint8)
    rgba[:,:,3] = array_ * -255 + 255
    image = Image.fromarray(rgba)
    image.save(filename)

spec = np.ones(angles_size + spec_size) * .5
for ax, ay in np.ndindex(angles_size):
    spec[ax, ay, :, :] = getImage(ax)

image_below = np.ones(image_size) * .25 # math.sqrt(2) / 2
image_above = np.ones(image_size) * .25 # math.sqrt(2) / 2


def seen(ax, ay):
    below = np.ones(spec_size)
    below[0:image_size[0], 0:image_size[1]] = image_below
    above = np.ones(spec_size)
    above[ax:image_size[0] + ax, ay:image_size[1] + ay] = image_above
    return below * above


def getder(ax, ay):
    return spec[ax, ay, :, :] - seen(ax, ay)


def getder_up(ax, ay):
    ret = np.zeros(image_size)
    d = getder(ax, ay)
    ret = d[ax:image_size[0] + ax, ay:image_size[1] + ay] * image_below
    return ret


def getder_down(ax, ay):
    ret = np.zeros(image_size)
    d = getder(ax, ay)
    ret = d[0:image_size[0], 0:image_size[1]] * image_above
    return ret


for i in range(20):
    der_up = np.zeros(image_size)
    for ax, ay in np.ndindex(angles_size):
        der_up += getder_up(ax, ay)

    der_down = np.zeros(image_size)
    for ax, ay in np.ndindex(angles_size):
        der_down += getder_down(ax, ay)

    image_below += der_down * .1
    image_above += der_up * .1

diffused_spec = np.zeros(angles_size + spec_size)
diffused_spec[:,:,:,:] = spec # copy data

def dither_below(x, y, w = 1):
    spec_ = diffused_spec[:, :, x, y]
    actual_above_ = image_above[x:x + angles_size[0], y:y + angles_size[1]]
    actual_above = np.ones(angles_size) * 1
    actual_above[0:actual_above_.shape[0], 0:actual_above_.shape[1]] = actual_above_
    diff1 = actual_above - spec_
    err1 = np.mean(np.abs(diff1))
    err0 = np.mean(np.abs(spec_))
    chosen_below = 0 if err0 < err1 else 1
    image_below[x, y] = image_below[x, y] * (1-w) + w * chosen_below
    residuals = spec_ - chosen_below * actual_above
    residuals = residuals * 1
    diffused_spec[:, :, x, y] -= residuals
    if x + 1 < spec_size[0]:
        diffused_spec[:, :, x + 1, y] += residuals * 7 / 16
    if y + 1 < spec_size[1]:
        diffused_spec[:, :, x, y + 1] += residuals * 5 / 16
    if x + 1 < spec_size[0] and y + 1 < spec_size[1]:
        diffused_spec[:, :, x + 1, y + 1] += residuals * 1 / 16
    if x - 1 >= 0 and y + 1 < spec_size[1]:
        diffused_spec[:, :, x - 1, y + 1] += residuals * 3 / 16

def dither_above(x, y, w=1):
    spec_ = np.ones(angles_size) * 1
    for ax, ay in np.ndindex(angles_size):
        if x - ax >= 0 and y - ay >= 0:
            spec_[ax, ay] = diffused_spec[ax, ay, x - ax, y - ay]
    actual_below = np.ones(angles_size) * 1
    for ax, ay in np.ndindex(angles_size):
        if 0 <= x - ax and 0 <= y - ay:
            actual_below[ax, ay] = image_below[x - ax, y - ay]
    diff1 = actual_below - spec_
    err1 = np.mean(np.abs(diff1))
    err0 = np.mean(np.abs(spec_))
    chosen_above = 0 if err0 <= err1 else 1
    image_above[x, y] = image_above[x, y] * (1-w) + w * chosen_above
    residuals = spec_ - chosen_above * actual_below
    residuals = residuals * 1
    for ax, ay in np.ndindex(angles_size):
        if x - ax >= 0 and y - ay >= 0:
            diffused_spec[ax, ay, x - ax, y - ay] -= residuals[ax,ay]
        if 0 <= x - ax + 1 < spec_size[0] and 0 <= y - ay < spec_size[1]:
            diffused_spec[ax, ay, x - ax + 1, y - ay] += residuals[ax, ay] * 7 / 16
        if 0 <= x - ax < spec_size[0] and 0 <= y - ay + 1 < spec_size[1]:
            diffused_spec[ax, ay, x - ax, y - ay + 1] += residuals[ax, ay] * 5 / 16
        if 0 <= x - ax + 1 < spec_size[0] and 0 <= y - ay + 1 < spec_size[1]:
            diffused_spec[ax, ay, x - ax + 1, y - ay + 1] += residuals[ax, ay] * 1 / 16
        if 0 <= x - ax - 1 < spec_size[0] and 0 <= y - ay + 1 < spec_size[1]:
            diffused_spec[ax, ay, x - ax - 1, y - ay + 1] += residuals[ax, ay] * 3 / 16

'''
'''
for i in range(100):
    print(i)
    for x, y in np.ndindex(image_size):
        dither_above(x, y, .05)
    diffused_spec[:,:,:,:] = spec # copy data
    for x, y in np.ndindex(image_size):
        dither_below(x, y, .05)
    diffused_spec[:,:,:,:] = spec # copy data

'''
for x, y in np.ndindex(image_size):
    dither_above(x, y, 1)
    dither_below(x, y, 1)
'''


saveImg(image_below, 'J:\\Documents\\OneDrive - Ultimaker B.V\\PhD\\holo\\export\\below.png')
saveImg(image_above, 'J:\\Documents\\OneDrive - Ultimaker B.V\\PhD\\holo\\export\\above.png')

showImg(image_below)
showImg(image_above)

for ax,ay in np.ndindex(angles_size):
    showImg(seen(ax, ay))