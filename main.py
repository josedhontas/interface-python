from my import *

img = imread("images\sin_all.gif")

dft_result = dft2(img)
imshow_complex(dft_result)

