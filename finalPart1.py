from scipy.fftpack import fft, fftfreq, fftshift, ifft
from skimage.util import compare_images
import matplotlib.pyplot as plt
import scipy.signal as signal
from skimage import filters, feature
from scipy import ndimage
import numpy as np
import cv2

# Add this class as a header

class ImageProcessing:
    def __init__(self, image):
        self.image = image

    def ImageProcess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)

        height, width = image.shape
        height_middle = int(height/2)
        width_middle = int(width/2)

        offset = int((height_middle + width_middle) / 3)
        image = image[height_middle-offset:height_middle+offset, width_middle-offset:width_middle+offset]

        casa = np.array(image)
        casa = casa[:,:]
        casa = cv2.resize(casa, (512,512))

        return casa

    def XY(self,image):
        x = np.arange(-image.shape[0]/2, image.shape[0]/2)
        y = np.arange(-image.shape[1]/2, image.shape[1]/2)
        x, y = np.meshgrid(x, y)

        x = np.linspace(-np.pi, np.pi, image.shape[0])
        y = np.linspace(-np.pi, np.pi, image.shape[1])

        # create meshgrid
        X, Y = np.meshgrid(x, y)

        return X, Y

    def ImageShow(self, image, title):
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()

class Fournier(ImageProcessing):
    def __init__(self, image):
        ImageProcessing.__init__(self, image)

    # def GaussianFilter(self, image, sigma):
    #     kernel = cv2.getGaussianKernel(5, sigma)
    #     kernel = kernel * kernel.T
    #     image = cv2.filter2D(image, -1, kernel)
    #     return image

    # def LaplacianOfGaussianFilter(self, image, sigma):
    #     kernel = cv2.getGaussianKernel(5, sigma)
    #     kernel = kernel * kernel.T
    #     image = cv2.filter2D(image, -1, kernel)
    #     image = cv2.Laplacian(image, -1)
    #     return image

    def FournierTransform(self, image):
        f = fft(image)
        fshift = fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        return magnitude_spectrum
        
    def InverseFourierTransform(self, image):
        fshift = fftshift(image)
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft(f_ishift)
        img_back = np.abs(img_back)
        return img_back

    def fftfreq(self, n, d=1.0):
        val = 1.0/(n*d)
        results = np.empty(n, int)
        N = (n-1)//2 + 1
        p1 = np.arange(0, N, dtype=int)
        results[:N] = p1
        p2 = np.arange(-(n//2), 0, dtype=int)
        results[N:] = p2
        return results * val

class ImageFilter(ImageProcessing):
    def __init__(self, image):
        super().__init__(image)
        
    def gaussian_filter(self, image, sigma=1):
        shape = super().ImageProcess(image)
        X, Y = super().XY(image)
        h = np.exp(-(X**2 + Y**2)/(2*sigma**2))
        return h
        
    def laplacian_of_gaussian_filter(self, image, sigma=1):
        shape = super().ImageProcess(image)
        X, Y = super().XY(image)
        h = -1/(np.pi*sigma**4) * (1 - (X**2 + Y**2)/(2*sigma**2)) * np.exp(-(X**2 + Y**2)/(2*sigma**2))
        return h

    def GaussianFilter(self, image, sigma=1):
        h = self.gaussian_filter(image, sigma)
        image = super().ImageProcess(image)
        filtered_image = signal.convolve2d(image, h, mode='same', boundary='symm')
        return filtered_image

    def LaplacianOfGaussianFilter(self, image, sigma=1): # Edge detection
        h = self.laplacian_of_gaussian_filter(image, sigma)
        filtered_image = signal.convolve2d(image, h, mode='same', boundary='symm')
        return filtered_image

    def SharpeningHighPass(self, image, sigma):
        h = self.laplacian_of_gaussian_filter(image, sigma)
        image = super().ImageProcess(image)
        filtered_image = signal.convolve2d(image, h, mode='same', boundary='symm')
        filtered_image = image + filtered_image
        return filtered_image

    def HighPassFilter(self, image, sigma=1):
        h = self.gaussian_filter(image, sigma)
        image = super().ImageProcess(image)
        filtered_image = image - h
        return filtered_image

    def LowPassFilter(self, image, sigma=1):
        h = self.gaussian_filter(image, sigma)
        image = super().ImageProcess(image)
        filtered_image = signal.convolve2d(image, h, mode='same', boundary='symm')
        return filtered_image

    def MeanFilter(self, image, size=3):
        filtered_image = ndimage.uniform_filter(image, size=size)
        return filtered_image

    def CannyEdgeDetection(self, image, sigma=1):
        h = self.laplacian_of_gaussian_filter(image, sigma)
        image = super().ImageProcess(image)
        filtered_image = signal.convolve2d(image, h, mode='same', boundary='symm')
        filtered_image = feature.canny(filtered_image)
        return filtered_image
        
    def LaplacianEdgeDetection(self, image, sigma=1):
        h = self.laplacian_of_gaussian_filter(image, sigma)
        image = super().ImageProcess(image)
        filtered_image = signal.convolve2d(image, h, mode='same', boundary='symm')
        return filtered_image

    def SobelFilter(self, image):
        filtered_image = ndimage.sobel(image)
        return filtered_image

    def PrewittFilter(self, image):
        # prewitt filter
        shape = super().ImageProcess(image)
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        # apply prewitt filter
        prewitt_x = signal.convolve2d(shape, prewitt_x, mode='same', boundary='symm')
        prewitt_y = signal.convolve2d(shape, prewitt_y, mode='same', boundary='symm')
        # calculate magnitude
        prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)

        return prewitt

    def denoise(self, image, weight):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.float32(image)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        dst = cv2.fastNlMeansDenoising(image, None, weight, 7, 21)
        return dst

if __name__ == '__main__':
    a_game = ImageFilter()
    a_game.ImageShow(a_game.image, 'Original Image')
    a_game.ImageShow(a_game.GaussianFilter(a_game.image, 1), 'Gaussian Filter')
    a_game.ImageShow(a_game.LaplacianOfGaussianFilter(a_game.image, 1), 'Laplacian of Gaussian Filter')
    a_game.ImageShow(a_game.SharpeningHighPass(a_game.image, 1), 'Sharpening High Pass Filter')
    a_game.ImageShow(a_game.HighPassFilter(a_game.image, 1), 'High Pass Filter')
    a_game.ImageShow(a_game.LowPassFilter(a_game.image, 1), 'Low Pass Filter')
    a_game.ImageShow(a_game.MedianFilter(a_game.image, 3), 'Median Filter')
    a_game.ImageShow(a_game.MeanFilter(a_game.image, 3), 'Mean Filter')
    a_game.ImageShow(a_game.CannyEdgeDetection(a_game.image, 1), 'Canny Edge Detection')
    a_game.ImageShow(a_game.LaplacianEdgeDetection(a_game.image, 1), 'Laplacian Edge Detection')
    a_game.ImageShow(a_game.SobelFilter(a_game.image), 'Sobel Filter')
    a_game.ImageShow(a_game.PrewittFilter(a_game.image), 'Prewitt Filter')
    a_game.ImageShow(a_game.denoise(a_game.image, 10), 'Denoise')