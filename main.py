import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def imagefft(I):
    # Calcula a DFT 2D
    F = np.fft.fft2(I)
    
    # Centraliza a DFT
    F = np.fft.fftshift(F)
    
    # Obtém a magnitude (não é estritamente necessário)
    magnitude = np.abs(F)
    
    # Aplica uma escala logarítmica (adiciona 1 para evitar log(0))
    log_magnitude = np.log(magnitude + 1)
    
    # Renormaliza a magnitude para 0-255
    normalized_magnitude = log_magnitude / np.max(log_magnitude) * 255
    
    # Calcula a IDFT 2D (inversa da DFT)
    reconstructed_image = np.fft.ifft2(np.fft.ifftshift(F)).real
    
    # Exibe a magnitude e a imagem reconstruída em tons de cinza
    plt.subplot(121)
    plt.imshow(normalized_magnitude, cmap='gray')
    plt.title('Magnitude da DFT')
    
    plt.subplot(122)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Imagem Reconstruída em Tons de Cinza')
    
    plt.show()

# Exemplo de uso:
img = mpimg.imread("images\lena1.jpg")
imagefft(img)  # Aplica a função imagefft à imagem em tons de cinza
