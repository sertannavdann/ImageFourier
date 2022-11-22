from scipy.fftpack import fft, fftfreq, fftshift, ifft
from skimage.util import compare_images
import matplotlib.pyplot as plt
import scipy.signal as signal
from skimage import filters, feature
from scipy import ndimage
import numpy as np
import cv2
import random
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

    def convolve_image(image, filter_matrix):
        
        im_pad = np.pad(image, 5, mode='constant') 
        im_conv = np.zeros_like(image)
        
        #the filter function takes in an image and a pair of indices and performs the convolution
        def filter_function(img, i, j):
            conv = np.zeros((3,3))
            for a in range(2,-1,-1):
                for b in range(2,-1,-1):
                    conv[2-a, 2-b] = filter_matrix[a , b] * img[i+6-a,j+6-b]
                    amt = np.sum(conv, axis = 0)
                    mean = np.sum(amt, axis=0)
            return mean 

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                im_conv[i,j] = filter_function(im_pad,i,j)    
        return im_conv
    


class Fournier(ImageProcessing):
    def __init__(self, image):
        ImageProcessing.__init__(self, image)

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

    def GaussianFilter(self, image, sigma):
        kernel = cv2.getGaussianKernel(5, sigma)
        kernel = kernel * kernel.T
        image = cv2.filter2D(image, -1, kernel)
        return image

    def LaplacianOfGaussianFilter(self, image, sigma=1): # Edge detection
        kernel = cv2.getGaussianKernel(5, sigma)
        kernel = (kernel * (kernel.T * sigma))
        image = cv2.filter2D(image, -1, kernel)
        image = cv2.Laplacian(image, -1)
        return image

    def SharpeningHighPass(self, image, sigma):
        kernel_ = np.ones((5,5),np.float32)/sigma
        kernel_[2,2] = 1 - sigma
        image = cv2.filter2D(image,-1,kernel_)
        return image
        
    def HighPassFilter(self, image, sigma=1):
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(image, -1, kernel)
        dst = image - dst
        return dst

    def LowPassFilter(self, image, sigma=1):
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(image,-1,kernel)
        return dst

    def MeanFilter(self, image, size=3):
        filtered_image = ndimage.uniform_filter(image, size=size)
        return filtered_image

    def CannyEdgeDetection(self, image, sigma=1):
        edges = cv2.Canny(image,sigma*80,sigma*90)
        return edges

    def GaussianEdgeDetection(self, image, sigma):
        h = self.GaussianFilter(image, sigma)
        image = image - h
        return image

    def SobelFilter(self, image):
        image = super().ImageProcess(image)
        # sobel filter
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # apply sobel filter
        sobel_x = signal.convolve2d(image, sobel_x, mode='same', boundary='symm')
        sobel_y = signal.convolve2d(image, sobel_y, mode='same', boundary='symm')

        # calculate magnitude
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)

        return sobel

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

    def salt_pepper_noise(self, image, sigma):
        output = np.zeros(image.shape, np.uint8)
        prob = sigma/50
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def denoise(self, image, sigma):
        sigma = int(sigma*3)
        denoised_image = cv2.medianBlur(image, sigma)
        return denoised_image


if __name__ == '__main__':
    a_game = ImageFilter()
    a_game.ImageShow(a_game.image, 'Original Image')
    a_game.ImageShow(a_game.GaussianFilter(a_game.image, 1), 'Gaussian Filter')
    a_game.ImageShow(a_game.LaplacianOfGaussianFilter(a_game.image, 1), 'Laplacian of Gaussian Filter')
    a_game.ImageShow(a_game.SharpeningHighPass(a_game.image, 1), 'Sharpening High Pass Filter')
    a_game.ImageShow(a_game.HighPassFilter(a_game.image, 1), 'High Pass Filter')
    a_game.ImageShow(a_game.LowPassFilter(a_game.image, 1), 'Low Pass Filter')
    a_game.ImageShow(a_game.MeanFilter(a_game.image, 3), 'Mean Filter')
    a_game.ImageShow(a_game.CannyEdgeDetection(a_game.image, 1), 'Canny Edge Detection')
    a_game.ImageShow(a_game.GaussianEdgeDetection(a_game.image, 1), 'Gaussian Edge Detection')
    a_game.ImageShow(a_game.SobelFilter(a_game.image), 'Sobel Filter')
    a_game.ImageShow(a_game.PrewittFilter(a_game.image), 'Prewitt Filter')
    a_game.ImageShow(a_game.salt_pepper_noise(a_game.image, 1), 'Salt and Pepper Noise')
    a_game.ImageShow(a_game.denoise(a_game.image, 10), 'Denoise')