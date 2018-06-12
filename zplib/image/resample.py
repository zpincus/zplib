import numpy
from scipy import ndimage

def sample_image_rect(image, size, center, rotation=0, order=3, **kwargs):
    """Return an image "swath" of a given (w, h) size, centered at a given (x, y)
    position and rotated by the specified degrees.

    Parameters:
    image: the image to sample from.
    size: (w, h) in pixels of the output image
    center: (x, y) position in the input image that the sampling region is
        centered on.
    rotation: rotation in degrees to apply to the sampling region.
        NB: for images, the positive x axis goes downward on the screen. So
        the rotation direction seems negated compared to conventional
        coordinates. (That is, a slight negative rotation will cause the
        sampling rect to be rotated a little counterclockwise.)
    order: image interpolation order for the resampling process.
        0 = nearest-neighbor interpolation
        1 = linear interpolation
        3 = cubic interpolation
    other keyword args are passed to ndimage.map_coordinates() to control the
        resampling. Useful arguments include 'mode' and 'cval' (see the
        documentaiton for ndimage.map_coordinates() for details.)

    Returns: array of the specified size containing the image pixels sampled
    from the region as defined by the parameters above."""

    size = numpy.asarray(size)
    x, y = numpy.indices(size) - (size / 2).reshape((2,1,1))
    theta = numpy.pi * rotation / 180
    rx = x*numpy.cos(theta)-y*numpy.sin(theta)
    ry = x*numpy.sin(theta)+y*numpy.cos(theta)
    coords = numpy.array([rx, ry]) + numpy.reshape(center, (2,1,1))
    sampled = ndimage.map_coordinates(image, coords, order=order, **kwargs)
    return sampled
