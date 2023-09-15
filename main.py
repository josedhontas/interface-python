from my import *

img = imread("images\lena_gray.png")

dft_result = dft2(img)
imshow_complex(dft_result)

