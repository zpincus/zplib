import numpy
import scipy.ndimage as ndimage
from skimage import transform

def pyr_down(image, downscale, nyquist_attenuation=0.015):
    """Return an image downsampled by the requested factor.

    Parameters:
        image: numpy array
        downscale: factor by which to shrink the image. 2 is customary.
        nyquist_attenuation: controls strength of low-pass filtering (see
            documentation for downsample_sigma() for detailed description).
            Larger values = more image blurring.

    Returns: image of type float32
    """
    out_shape = numpy.ceil(numpy.array(image.shape) / float(downscale)).astype(int)
    sigma = downsample_sigma(downscale, nyquist_attenuation)
    smoothed = ndimage.gaussian_filter(image.astype(numpy.float32), sigma, mode='reflect')
    return transform.resize(smoothed, out_shape, order=1, mode='reflect', preserve_range=True)

def pyr_up(image, upscale, nyquist_attenuation=0.985):
    """Return an image upsampled by the requested factor.

    Parameters:
        image: numpy array
        upscale: factor by which to enlarge the image. 2 is customary.
        nyquist_attenuation: controls strength of low-pass filtering (see
            documentation for downsample_sigma() for detailed description).
            Larger values = more image blurring.

    Returns: image of type float32
    """
    out_shape = numpy.ceil(numpy.array(image.shape) * upscale).astype(int)
    sigma = downsample_sigma(upscale, nyquist_attenuation)
    resized = transform.resize(image.astype(numpy.float32), out_shape, order=1, mode='reflect', preserve_range=True)
    return ndimage.gaussian_filter(resized, sigma, mode='reflect')

def downsample_sigma(scale_factor, nyquist_attenuation=0.05):
    """Calculate sigma for gaussian blur that will attenuate the nyquist frequency
    of an image (after down-scaling) by the specified fraction. Surprisingly,
    attenuating by only 5-10% is generally sufficient (nyquist_attenuation=0.05
    to 0.1).
    See http://www.evoid.de/page/the-caveats-of-image-down-sampling/ .
    """
    return scale_factor * (-8*numpy.log(1-nyquist_attenuation))**0.5