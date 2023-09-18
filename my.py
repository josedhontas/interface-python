import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.cm as cm

def imread(filename):
    if filename.endswith('.gif'):
        im = mpimg.imread(filename)
        return im

    im = mpimg.imread(filename)
    if im.dtype == np.float32:
        im = (im * 255).astype(np.uint8)
    if len(im.shape) >= 3 and im.shape[2] > 3:
        im = im[:, :, 0:3]
    return im

def imshow(im):
    if len(im.shape) == 2:
        cmap = cm.gray
    else:
        cmap = None
    
    plot = plt.imshow(im, cmap=cmap, origin="upper")
    plot.set_interpolation('nearest')
    plt.show()

def nchannels(im):
    #print(im.shape)
    if len(im.shape) >= 3:
        return im.shape[2]
    else:
        return 1

def size(im):
    height, width = im.shape[:2]
    return [width, height]

def rgb2gray(im):
    weights = [0.299, 0.587, 0.114]
    gray_image = np.dot(im[..., :3], weights)
    return gray_image.astype(np.uint8)

def imreadgray(filename):
    im = imread(filename)
    if nchannels(im) >= 3:
        gray_image = rgb2gray(im)
    else:
        gray_image = im
        
    return gray_image

def thresh(im, threshold):
    binary_image = np.where(im >= threshold, 255, 0)
    return binary_image.astype(np.uint8)

def negative(im):
    negative_image = 255 - im
    return negative_image.astype(np.uint8)

def contrast(im, r, m):
    img = im.astype(np.float32)
    g = r * (img - m) + m
    g = np.clip(g, 0, 255)
    return g.astype(np.uint8)


def hist(im):
    if nchannels(im) == 1:
        hist_values = np.bincount(im.flatten(), minlength=256)
    else:
        hist_values = np.column_stack([np.bincount(im[:,:,i].flatten(), minlength=256) for i in range(3)])
    
    return hist_values

def showhist(hist_values, binsize=1):
    if binsize <= 0:
        raise ValueError("O tamanho dos bins deve ser positivo")
    
    hist_length = hist_values.shape[0]
    num_bins = hist_length // binsize
    binned_hist = hist_values[:num_bins * binsize].reshape(num_bins, binsize, -1).sum(axis=1)
    
    if nchannels(hist_values) == 3:
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            plt.bar(range(num_bins), binned_hist[:, i], color=color, alpha=0.7, label=color)
    else:  
        plt.bar(range(num_bins), binned_hist, color='gray', alpha=0.7, label='gray')
    
    plt.xlabel('Intervalos de Intensidade')
    plt.ylabel('Frequencia')
    plt.legend()
    plt.show()

def histeq(im):
    hist_values = hist(im)
    cdf = hist_values.cumsum()
    cdf_min = cdf.min()

    new_intensity = np.round(((cdf - cdf_min)/ (im.size - cdf_min))*255)

    equalized_image = new_intensity[im]

    return equalized_image.astype(np.uint8)

def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width, num_channels = image.shape
    output = np.zeros_like(image)
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='edge')
    
    for y in range(image_height):
        for x in range(image_width):
            for c in range(num_channels):
                region = padded_image[y:y + kernel_height, x:x + kernel_width, c]
                output[y, x, c] = np.sum(region * kernel)
    
    return output


def maskBlur():
    mask = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]])
    mask = mask / 16.0  
    return mask

def blur(image):
    blur_mask = maskBlur()
    blurred_image = convolve(image, blur_mask)
    return blurred_image

def seSquare3():
    element = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])
    return element

def seCross3():
    element = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    return element


def erode(image, element):
    element_height, element_width = element.shape
    image_height, image_width = image.shape[:2] 
    num_channels = nchannels(image)

    output = np.zeros_like(image)
    
    pad_height = element_height // 2
    pad_width = element_width // 2
    
    padding = ((pad_height, pad_height), (pad_width, pad_width))
    if num_channels > 1:
        padding += ((0, 0),) 

    padded_image = np.pad(image, padding, mode='constant', constant_values=255)
    
    for y in range(pad_height, image_height + pad_height):
        for x in range(pad_width, image_width + pad_width):
            if np.array_equal(element, seSquare3()):
                region = padded_image[y - pad_height:y + pad_height + 1, x - pad_width:x + pad_width + 1]
                output[y - pad_height, x - pad_width] = np.min(region)
            else:
                output[y - pad_height, x - pad_width] = padded_image[y, x]
    
    return output

def dilate(image, element):
    element_height, element_width = element.shape
    image_height, image_width = image.shape[:2]
    num_channels = nchannels(image)

    output = np.zeros_like(image)

    pad_height = element_height // 2
    pad_width = element_width // 2

    padding = ((pad_height, pad_height), (pad_width, pad_width))
    if num_channels > 1:
        padding += ((0, 0),)

    padded_image = np.pad(image, padding, mode='constant', constant_values=0) 

    for y in range(pad_height, image_height + pad_height):
        for x in range(pad_width, image_width + pad_width):
            if np.array_equal(element, seSquare3()):
                region = padded_image[y - pad_height:y + pad_height + 1, x - pad_width:x + pad_width + 1]
                output[y - pad_height, x - pad_width] = np.max(region) 
            else:
                output[y - pad_height, x - pad_width] = padded_image[y, x]

    return output


def dft2(image):
    M, N = image.shape
    dft_result = np.zeros((M, N), dtype=complex)
    
    for u in range(M):
        for v in range(N):
            dft_result[u, v] = np.sum(image * np.exp(-2j * np.pi * (u * np.arange(M) / M + v * np.arange(N) / N)))
    
    return dft_result

'''
def dft2(image):
    M, N = image.shape
    dft_result = np.zeros((M, N), dtype=complex)
    
    for u in range(M):
        for v in range(N):
            dft_result[u, v] = np.sum(image * np.exp(-2j * np.pi * (u * np.arange(M) / M + v * np.arange(N) / N)))
    
    return dft_result

def idft2(dft_result):
    M, N = dft_result.shape
    idft_image = np.zeros((M, N), dtype=complex)
    
    for x in range(M):
        for y in range(N):
            idft_image[x, y] = np.sum(dft_result * np.exp(2j * np.pi * (x * np.arange(M) / M + y * np.arange(N) / N))) / (M * N)
    
    return idft_image.real
 '''

def imagefft(I):
    F = np.fft.fft2(I)
    
    F = np.fft.fftshift(F)
    
    F = np.abs(F)
    
    F = np.log(F + 1)
    
    F = F / np.max(F)
    
    plt.imshow(F, cmap='gray')
    plt.show()

def idft2(dft_result):
    idft_image = np.fft.ifft2(dft_result)
    return idft_image.real

def imagefft(I):
    F = np.fft.fft2(I)
    
    F = np.fft.fftshift(F)
    
    magnitude = np.abs(F)
    
    log_magnitude = np.log(magnitude + 1)
    
    normalized_magnitude = log_magnitude / np.max(log_magnitude) * 255
    
    reconstructed_image = np.fft.ifft2(np.fft.ifftshift(F)).real
    
    plt.subplot(121)
    plt.imshow(normalized_magnitude, cmap='gray')
    plt.title('Magnitude da DFT')
    
    plt.subplot(122)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Imagem Reconstruída')
    
    plt.show()




