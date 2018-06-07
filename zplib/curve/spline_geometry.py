import numpy

from . import geometry
from . import interpolate

def arc_length(tck, num_points=None):
    """Approximate the arc-length of spline (t,c,k) by evaluating it at num_points
    positions and calculating the length of the resulting polyline.
    If num_points is None, try to guess a sane default."""
    points = _get_points(tck, num_points)
    return numpy.sqrt(((points[:-1] - points[1:])**2).sum(axis=1)).sum()


def perpendiculars(tck, num_points=None, unit=True):
    """Return vectors perpendicular to a 2D parametric spline.

    Parameters:
    tck: parametric spline tuple
    num_points: number of equally-spaced points to evaluate perpendiculars at,
        or None, which causes the code to try to guess a good number of points.
    unit: normalize prependiculars to unit length.

    Returns: array of shape (num_points, 2), containing num_points different
        2D vectors describing each perpendicular.
    """
    der = _get_points(tck, num_points, derivative=1) # shape = (length, 2)
    perpendiculars = numpy.empty_like(der)
    perpendiculars[:,0] = -der[:,1]
    perpendiculars[:,1] = der[:,0]
    if unit:
        perpendiculars /= numpy.sqrt((perpendiculars**2).sum(axis=1))[:,numpy.newaxis]
    return perpendiculars


def outline(tck, radius_tck, num_points=None):
    """Given a shape defined by a centerline spline and a radial profile spline,
    return a polygonal outline defined by sampling the centerline num_points
    along its profile and displacing it left and right of the centerline by an
    amount defined by the radius_tck.

    If num_points is None, try to guess a sane default.

    Returns: left, right, outline, where left and right are the displaced points
    (shape=(num_points, 2)) and outline is the full polygon (shape=(2*num_points, 2))
    """
    points, radii = _get_points_and_radii(tck, radius_tck, num_points)
    perps = perpendiculars(tck, num_points=len(points))
    offsets = perps * radii[:,numpy.newaxis]
    left = points + offsets
    right = points - offsets
    outline = numpy.concatenate([left, right[::-1]], axis=0)
    return left, right, outline


def area(tck, radius_tck, num_points=None):
    """Given a shape defined by a centerline spline and a radial profile spline,
    estimate its area by converting to a polygon at num_points along the centerline
    and applying the standard polygon area formula.

    If num_points is None, try to guess a sane default.
    """
    polygon = outline(tck, radius_tck, num_points)
    xs = polygon[:,0]
    ys = polygon[:,1]
    y_forward = numpy.roll(ys, -1, axis = 0)
    y_backward = numpy.roll(ys, 1, axis = 0)
    return numpy.absolute(numpy.sum(xs * (y_backward - y_forward)) / 2.0)


def volume_and_surface_area(tck, radius_tck, num_points=None):
    """Given a shape defined by a centerline spline and a radial profile spline,
    estimate the volume and surface area of a 3D form constructed by revolution
    around the centerline, by sampling the positions and radii at num_points
    along the centerline, assuming that the shape of revolution is made of
    and conical frustrums defined by these radii, and applying the standard
    formulae for their surface area and volume.

    If num_points is None, try to guess a sane default.

    Returns: volume, surface_area"""
    points, radii = _get_points_and_radii(tck, radius_tck, num_points)
    lengths = numpy.sqrt(((points[:-1] - points[1:])**2).sum(axis=1))
    # formulae from http://en.wikipedia.org/wiki/Frustum
    r1 = radii[:-1]
    r2 = radii[1:]
    r12 = r1**2
    r22 = r2**2
    h = lengths
    volume = numpy.pi/3 * (h * (r12 + r22 + r1*r2)).sum()
    surface_area = numpy.pi * (numpy.sqrt((r12-r22)**2 + (h*(r1+r2))**2).sum() + r12[0] + r22[-1])
    return volume, surface_area


def length_and_max_width(tck, radius_tck, num_points=None):
    """Given a shape defined by a centerline spline and a radial profile spline,
    estimate its length and maximum width by sampling at num_points along the
    spline.

    If num_points is None, try to guess a sane default.

    Returns: length, max_width
    """
    points, radii = _get_points_and_radii(tck, radius_tck, num_points)
    return numpy.sqrt(((points[:-1] - points[1:])**2).sum(axis=1)).sum(), radii.max()


def _get_points(tck, num_points=None, derivative=0):
    """Evaluate a spline (or its derivative) at a given number of points.

    If the number of points is not specified, make a sane guess such that the
    resulting curve will be approximately as smooth as the spline.
    Specifically, use the maximum parameter value of the spline or 100,
    whichever is greater. The maximum parameter value makes sense only if the
    curve is parameterized with something close to the 'natural parameter',
    which interpolate.fit_spline() tries to guarantee."""
    if num_points is None:
        num_points = max(100, int(round(tck[0].max())))
    points = interpolate.spline_interpolate(tck, num_points, derivative)
    return points


def _get_points_and_radii(tck, radius_tck, num_points=None):
    if tck[1].ndim != 2 and tck[1].shape[1] != 2:
        raise ValueError('tck must be a two-dimensional parametric spline')
    if radius_tck[1].ndim != 1:
        raise ValueError('radius_tck must be a non-parametric spline')
    points = _get_points(tck, num_points)
    radii = _get_points(radius_tck, num_points=len(points))
    return points, radii
