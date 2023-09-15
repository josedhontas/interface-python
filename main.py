from my import *

img = imread("images\sin2.gif")

dft_result = dft2(img)
imshow_complex(dft_result)

