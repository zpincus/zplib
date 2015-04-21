import numpy
from scipy import ndimage
from ..curve import interpolate

def sample_image_along_spline(image, tck, width, length=None, width_distance=None, order=3, **kwargs):
    """Return an image "swath" of a given length and width in pixels,
    centered along a parametric spline.

    Arguments:
    image: the image to sample from.
    tck: a parametric spline giving x, y coordinates to sample the image along.
    width: number of image samples to take perpendicular to the spline.
    length: number of image samples to take along the spline. For the resampling
        to neither magnify nor minify the image, this length ought to be the arc
        length of the spline. If more samples are taken than the arc length in
        pixels, then the image is magnified in the along-spline direction, and
        if fewer, then minification is obtained. If None, then the arc length
        is calculated and used.
    width_distance: distance (in image pixels) to sample along the perpendicular
        direction on each side of the spline. If None, this defaults to width/2,
        resulting in no magnification / minification in that direction.
        If width_distance > (width / 2) there will be minification, or if
        width_distance < (width / 2) there will be magnification.
    order: image interpolation order for the resampling process.
        0 = nearest-neighbor interpolation
        1 = linear interpolation
        3 = cubic interpolation
    other keyword args are passed to ndimage.map_coordinates() to control the
        resampling. Useful arguments include 'mode' and 'cval' (see the
        documentaiton for ndimage.map_coordinates() for details.)

    Returns: array of shape (length, width) containing samples along the length
    of the spline, going perpendicular from the spline width_distance pixels
    on each side."""
    coords = get_spline_sample_positions(tck, width, length, width_distance)
    warped = ndimage.map_coordinates(image, coords, order=order, **kwargs)
    return warped

def sample_image_rect(image, size, center, rotation=0, order=3, **kwargs):
    """Return an image "swath" of a given (w, h) size, centered at a given (x, y)
    position and rotated by the specified degrees.

    Arguments:
    image: the image to sample from.
    size: (w, h) in pixels of the output image
    center: (x, y) position in the input image that the sampling region is
        centered on.
    rotation: rotation in degrees to apply to the sampling region.
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
    theta = numpy.pi * degrees / 180
    rx = x*numpy.cos(-theta)-y*numpy.sin(-theta)
    ry = x*numpy.sin(-theta)+y*numpy.cos(-theta)
    coords = numpy.array([rx, ry]) + numpy.reshape(center, (2,1,1))
    sampled = nd.map_coordinates(image, coords, order=order, **kwargs)
    return sampled

def get_spline_sample_positions(tck, width, length=None, width_distance=None):
    """Return an array of shape (2, length, width) containing the x,y positions
    (along the first axis) of points sampled along the input spline tck.

    See the documentation for sample_image_along_spline() for explanation of the
    parameters."""
    if length is None:
        length = int(round(interpolate.spline_arc_length(tck)))

    u = numpy.linspace(0, tck[0][-1], length)
    points = interpolate.spline_interpolate(tck, length)
    der = interpolate.spline_interpolate(tck, length, derivative=1)
    perpendiculars = numpy.empty_like(der)
    perpendiculars[:,0] = -der[:,1]
    perpendiculars[:,1] = der[:,0]
    perpendiculars /= numpy.sqrt((perpendiculars**2).sum(axis=1))[:,numpy.newaxis]

    coords = numpy.empty((2, length, width), float)
    coords[:] = points.T[..., numpy.newaxis]
    offsets = numpy.empty((length, width), float)
    if width_distance is None:
        width_distance = width / 2.
    offsets[:] = numpy.linspace(-width_distance, width_distance, width)
    coords += offsets * perpendiculars.T[..., numpy.newaxis]
    return coords
