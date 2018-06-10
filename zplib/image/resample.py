import numpy
from scipy import ndimage
from ..curve import interpolate
from ..curve import spline_geometry

def sample_image_along_spline(image, tck, width, length=None, width_distance=None, order=3, **kwargs):
    """Return an image "swath" of a given length and width in pixels, centered
    along a parametric spline.

    Parameters:
    image: the image to sample from.
    tck: a parametric spline giving x, y coordinates to sample the image along.
    width: number of image samples to take perpendicular to the spline.
    length: number of image samples to take along the spline. For the resampling
        to neither magnify nor minify the image, this length ought to be the arc
        length of the spline. If more samples are taken than the arc length in
        pixels, then the image is magnified in the along-spline direction, and
        if fewer, then minification is obtained. If None, then the arc length
        is calculated and used.
    width_distance: distance (in image pixels) to sample perpendicular to the
        spline. If None, this defaults to width, resulting in no magnification
        or minification in that direction. If width_distance > width there will
        be minification, or if width_distance < width there will be
        magnification.
    order: image interpolation order for the resampling process.
        0 = nearest-neighbor interpolation
        1 = linear interpolation
        3 = cubic interpolation
    other keyword args are passed to ndimage.map_coordinates() to control the
        resampling. Useful arguments include 'mode' and 'cval' (see the
        documentation for ndimage.map_coordinates() for details.)

    Returns: array of shape (length, width) containing samples along the length
    of the spline, going perpendicular from the spline width_distance pixels
    on each side."""
    points, offset_directions = get_spline_sample_positions(tck, width, length, width_distance)
    sample_coordinates = points + offset_directions
    warped = ndimage.map_coordinates(image, sample_coordinates, order=order, **kwargs)
    return warped


def warp_image_to_standard_width(image, tck, src_radius_tck, dst_radius_tck, width, length=None, width_distance=None, order=3, **kwargs):
    """Return an image swath of a given length and width in pixels, centered
    along a parametric spline, and stretched non-uniformly perpendicular to
    the spline, such that an object in the image with an outline a determined
    by the spline and a radius-profile will be warped to a new radius-profile.

    This is most useful for stretching any given worm image to match some
    standardized "unit worm" in both length and radial profile.

    Parameters: image, tck, width, length, width_distance, order, kwargs are all
        as in sample_image_along_spline()
    src_radius_tck: nonparametric spline defining the raidus of the object along
        the tck centerline spline.
    dst_radius_tck: nonparametric spline defining the desired raidus profile.

    Returns: sampled array, as in sample_image_along_spline().
    """
    points, offset_directions = get_spline_sample_positions(tck, width, length, width_distance)
    length = points.shape[1]
    src_widths = interpolate.spline_interpolate(src_radius_tck, num_points=length)
    dst_widths = interpolate.spline_interpolate(dst_radius_tck, num_points=length)
    zero_width = dst_widths == 0
    dst_widths[zero_width] = 1 # don't want to divide by zero below
    width_ratios = src_widths / dst_widths # shape = (length,)
    width_ratios[zero_width] = 0 # this will enforce dest width of zero at these points
    sample_coordinates = points + offset_directions * width_ratios[:, numpy.newaxis]
    warped = ndimage.map_coordinates(image, sample_coordinates, order=order, **kwargs)
    return warped


def make_mask_for_sampled_spline(length, width, radius_tck, width_distance=None):
    """Given a the length and width of an image produced by sample_image_along_spline(),
    and a nonparametric spline giving the radius of an object (i.e. worm) along
    that spline, return a boolean mask that is True in the interior of that object.

    Parameters:
    length, width: the shape of the image produced by sample_image_along_spline()
        or similar.
    radius_tck: a nonparametric spline giving the radius (i.e. half-width) of
        the object at every point along the spline used for image sampling, in
        pixel units of the sampled image.
    width_distance: same as 'width_distance' parameter to sample_image_along_spline():
        this refers to the distance in source-image pixels sampled along the
        width of the output image. If None, assume this is the same as 'width'.

    Returns: boolean array of shape (length, width) that is True for every pixel
        that is within the radius of the object at each point along the image.
    """
    radius_at_each_pixel = interpolate.spline_interpolate(radius_tck, num_points=length)
    if width_distance is None:
        width_distance = width
    distance_from_image_centerline = numpy.abs(numpy.linspace(-width_distance/2, width_distance/2, width))
    # radius_at_each_pixel is 'length' pixels long,
    # distance_from_image_centerline is 'width' pixles wide
    # we want an image where element (i, j) is the result of
    # radius_at_each_pixel[i] > distance_from_image_centerline[j]
    # (i.e. 'True' where for all pixels that are closer to the centerline than
    # the edge of the object). Some thinking should convince you that this is
    # the "outer product" of the greater-than operator...
    return numpy.greater.outer(radius_at_each_pixel, distance_from_image_centerline)


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


def get_spline_sample_positions(tck, width, length=None, width_distance=None):
    """Calculate positions along the spline and vector offsets from those points
    for each sample position along the perpendiculars to that spline.

    See the documentation for sample_image_along_spline() for explanation of the
    parameters.

    Returns: points, offset_directions
        points.shape = (2, length, 1)
        offset_directions.shape = (2, length, width)
        (Note: points's shape is such that it can simply be added to the
        offset_directions array, via numpy broadcasting.)
    """
    if length is None:
        length = int(round(spline_geometry.arc_length(tck)))
    if width_distance is None:
        width_distance = width

    perpendiculars = spline_geometry.perpendiculars(tck, length).T # shape = (2, length)
    offsets = numpy.linspace(-width_distance/2, width_distance/2, width) # distances along each perpendicular across the width of the sample swath
    offset_directions = numpy.multiply.outer(perpendiculars, offsets) # shape = (2, length, width)
    points = interpolate.spline_interpolate(tck, length) # shape = (length, 2)
    points = points.T[..., numpy.newaxis] # now points.shape = (2, length, 1)
    return points, offset_directions
