from my import *

image = imread("images\sin_all.gif")

dft_result = dft2(image)
imshow_complex(dft_result)