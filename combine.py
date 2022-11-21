from finalPart1 import ImageFilter
import cv2
import sys

image = cv2.imread(sys.argv[1])
filter_type = sys.argv[2]
sigma = float(sys.argv[3])
size = sigma

Filter = ImageFilter(image)
PATH = 'images/edited'
if filter_type == 'Gaussian':
    Filter.ImageShow(Filter.GaussianFilter(image, sigma), 'Gaussian Filter')
    cv2.imwrite(PATH + '/GaussianFilter.png', Filter.GaussianFilter(image, sigma))
elif filter_type == 'Laplacian':
    Filter.ImageShow(Filter.LaplacianOfGaussianFilter(image, sigma), 'Laplacian of Gaussian Filter')
    cv2.imwrite(PATH + '/LaplacianOfGaussianFilter.png', Filter.LaplacianOfGaussianFilter(image, sigma))
elif filter_type == 'Sharpening':
    Filter.ImageShow(Filter.SharpeningHighPass(image, sigma), 'Sharpening High Pass Filter')
    cv2.imwrite(PATH + '/SharpeningHighPass.png', Filter.SharpeningHighPass(image, sigma))
elif filter_type == 'HighPass':
    Filter.ImageShow(Filter.HighPassFilter(image, sigma), 'High Pass Filter')
    cv2.imwrite(PATH + '/HighPassFilter.png', Filter.HighPassFilter(image, sigma))
elif filter_type == 'LowPass':
    Filter.ImageShow(Filter.LowPassFilter(image, sigma), 'Low Pass Filter')
    cv2.imwrite(PATH + '/LowPassFilter.png', Filter.LowPassFilter(image, sigma))
elif filter_type == 'Mean':
    Filter.ImageShow(Filter.MeanFilter(image, size), 'Mean Filter')
    cv2.imwrite(PATH + '/MeanFilter.png', Filter.MeanFilter(image, size))
elif filter_type == 'Canny':
    Filter.ImageShow(Filter.CannyEdgeDetection(image, sigma), 'Canny Edge Detection')
    cv2.imwrite(PATH + '/CannyEdgeDetection.png', Filter.CannyEdgeDetection(image, sigma))
elif filter_type == 'LaplacianEdge':
    Filter.ImageShow(Filter.LaplacianEdgeDetection(image, sigma), 'Laplacian Edge Detection')
    cv2.imwrite(PATH + '/LaplacianEdgeDetection.png', Filter.LaplacianEdgeDetection(image, sigma))
elif filter_type == 'Sobel':
    Filter.ImageShow(Filter.SobelFilter(image), 'Sobel Filter')
    cv2.imwrite(PATH + '/SobelFilter.png', Filter.SobelFilter(image))
elif filter_type == 'Prewitt':
    Filter.ImageShow(Filter.PrewittFilter(image), 'Prewitt Filter')
    cv2.imwrite(PATH + '/PrewittFilter.png', Filter.PrewittFilter(image))
elif filter_type == 'Denoise':
    Filter.ImageShow(Filter.denoise(image, sigma), 'Denoise')
    cv2.imwrite(PATH + '/Denoise.png', Filter.denoise(image, sigma))
else:
    print ("Invalid filter type")

#save the image
cv2.imwrite('output.jpg', Filter.image)