from my import *

img = imread("images\lena1.jpg")
element = seSquare3()
#convolve_img = convolve(img, element)
erode_img = DFT(img)
imshow(erode_img)