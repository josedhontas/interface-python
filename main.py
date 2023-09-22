from my import *

img = imread("images\lena1.jpg")
img_uint = (img*255).astype(np.uint8)
img_negative = 255 - img
imshow(contrast(img, 1, 200))

