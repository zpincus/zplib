import numpy
import scipy.ndimage as ndimage
from skimage import transform

# code taken and cleaned up from skimage

def pyr_down(image, downscale):
    out_shape = numpy.ceil(numpy.array(image.shape) / float(downscale)).astype(int)
    sigma = 2 * downscale / 6.0
    # corresponds to a filter mask twice the size of the scale factor, which
    # covers more than 99% of the Gaussian distribution
    smoothed = ndimage.gaussian_filter(image.astype(numpy.float32), sigma, mode='reflect')
    return transform.resize(smoothed, out_shape, order=1, mode='reflect', preserve_range=True)

def pyr_up(image, upscale):
    out_shape = numpy.ceil(numpy.array(image.shape) * upscale).astype(int)
    sigma = 2 * upscale / 6.0
    # corresponds to a filter mask twice the size of the scale factor, which
    # covers more than 99% of the Gaussian distribution
    resized = transform.resize(image.astype(numpy.float32), out_shape, order=1, mode='reflect', preserve_range=True)
    return ndimage.gaussian_filter(resized, sigma, mode='reflect')
