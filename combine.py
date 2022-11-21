from finalPart1 import ImageFilter
import cv2
import sys

image = cv2.imread(sys.argv[1])
filter_type = sys.argv[2]
sigma = float(sys.argv[3])
size = sigma

Filter = ImageFilter(image)

if filter_type == 'Gaussian':
    Filter.ImageShow(Filter.GaussianFilter(image, sigma), 'Gaussian Filter')
elif filter_type == 'Laplacian':
    Filter.ImageShow(Filter.LaplacianOfGaussianFilter(image, sigma), 'Laplacian of Gaussian Filter')
elif filter_type == 'Sharpening':
    Filter.ImageShow(Filter.SharpeningHighPass(image, sigma), 'Sharpening High Pass Filter')
elif filter_type == 'HighPass':
    Filter.ImageShow(Filter.HighPassFilter(image, sigma), 'High Pass Filter')
elif filter_type == 'LowPass':
    Filter.ImageShow(Filter.LowPassFilter(image, sigma), 'Low Pass Filter')
elif filter_type == 'Mean':
    Filter.ImageShow(Filter.MeanFilter(image, size), 'Mean Filter')
elif filter_type == 'Canny':
    Filter.ImageShow(Filter.CannyEdgeDetection(image, sigma), 'Canny Edge Detection')
elif filter_type == 'LaplacianEdge':
    Filter.ImageShow(Filter.LaplacianEdgeDetection(image, sigma), 'Laplacian Edge Detection')
elif filter_type == 'Sobel':
    Filter.ImageShow(Filter.SobelFilter(image), 'Sobel Filter')
elif filter_type == 'Prewitt':
    Filter.ImageShow(Filter.PrewittFilter(image), 'Prewitt Filter')
elif filter_type == 'Denoise':
    Filter.ImageShow(Filter.denoise(image, sigma), 'Denoise')
else:
    print ("Invalid filter type")

