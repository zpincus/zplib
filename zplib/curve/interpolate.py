import numpy
from scipy.interpolate import _fitpack_impl as fitpack

from . import geometry

def linear_resample_polyline(points, num_points):
    """Resample a piecewise linear curve to contain a given number of
    equally-spaced points, using linear interpolation.

    Parameters:
    points: array of n points x,y; shape=(n,2)
    num_points: number of output points in array.

    Returns a resampled array, of shape (num_points,2)"""
    points = numpy.asarray(points)
    distances = geometry.cumulative_distances(points, unit=True)
    sample_positions = numpy.linspace(0, 1, num_points)
    x = numpy.interp(sample_positions, distances, points[:,0])
    y = numpy.interp(sample_positions, distances, points[:,1])
    return numpy.transpose([x,y])

def spline_resample_polyline(points, num_points):
    """Resample a piecewise linear curve to contain a given number of
    equally-spaced points, using spline interpolation with automatically calculated
    smoothing.

    Parameters:
    points: array of n points x,y; shape=(n,2)
    num_points: number of output points in array.

    Returns a resampled array, of shape (num_points,2), and the spline parameters
    used for the resampling"""
    tck = fit_spline(points)
    points_out = spline_interpolate(tck, num_points)
    return points_out, tck

def fit_spline(points, smoothing=None, order=None, force_endpoints=True, periodic=False):
    """Fit a parametric smoothing spline to a given set of x,y points. (Fits
    x(p) and y(p) as functions for some parameter p.)

    Parameters:
    points: array of n points x,y; shape=(n,2)
    smoothing: smoothing factor: 0 requires perfect interpolation of the
        input points, at the cost of potentially high noise. Very large values
        will result in a low-order polynomial fit to the points. If None, an
        appropriate value based on the scale of the points will be selected.
        Specifically, the fitting function guarantees that sum of the distances
        between the input points and the spline will be less than or equal to
        this smoothing parameter.
    order: The desired order of the spline. If None, will be 1 if there are
        three or fewer input points, and otherwise 3.
    force_endpoints: if True (default), the endpoints of the spline will be set
        to match the positions of the input data, regardless of smoothing.
        NB: With large smoothing, this can dramatically influence the position
        of the entire curve. With smaller smoothing, changing the endpoints
        will have a more local effect.
    periodic: if True, create a periodic spline. This will implicitly add an
        additional point at the end of the points list equal to the first point.
        DO NOT manually add this point. If periodic is true, force_endpoints
        will be disabled.

    Returns a spline tuple (t,c,k) consisting of:
        t: the knots of the spline curve
        c: the x and y b-spline coefficients for each knot (shape (m, 2))
        k: the order of the spline.

    Note: the smoothing factor is an upper bound on the sum of all the squared
    distances between the original x,y points and the matching points on the
    smoothed spline representation."""
    points = numpy.asarray(points)
    if periodic:
        points = numpy.concatenate([points, points[:1]], axis=0)
        force_endpoints = False
    l = len(points)
    w = numpy.ones(l, float)
    # choose input parameter values for the curve as the distances along the polyline:
    # this gives something close to the "natural parameterization" of the curve.
    # (i.e. a parametric curve with first-derivative close to unit magnitude: the curve
    # doesn't accelerate/decelerate, so points along the curve in the x,y plane
    # don't "bunch up" with evenly-spaced parameter values.)
    distances = geometry.cumulative_distances(points, unit=False)
    if numpy.any(numpy.isclose(distances[1:] - distances[:-1], 0)):
        raise ValueError("Repeated input points are not allowed.")
    if order is None:
        if l < 4:
            k = 1
        else:
            k = 3
    else:
        k = order

    if smoothing is None:
        smoothing = l * distances[-1] / 600.

    if force_endpoints:
        w[[0,-1]] = l

    t, c, ier, msg = splprep(distances, points, s=smoothing, k=k, w=w, per=periodic)

    if ier > 3:
        raise RuntimeError(msg)
    if  force_endpoints:
        # endpoints should already be close, so this shouldn't distort the spline too much...
        c[[0,-1]] = points[[0,-1]]

    return t, c, k

def fit_nonparametric_spline(x, y, smoothing=None, order=None, force_endpoints=True, periodic=False):
    """Fit a non-parametric smoothing spline to x,y points. (Fits a function
    f(x) = y, or approximately so.)

    Parameters:
    x: array of shape=(n,). Must be monotonic
    y: array of shape=(n,)
    smoothing: smoothing factor: 0 requires perfect interpolation of the
        input points, at the cost of potentially high noise. Very large values
        will result in a low-order polynomial fit to the points. If None, an
        appropriate value based on the scale of the points will be selected.
        Specifically, the fitting function guarantees that sum of the distances
        between the input y values and the spline at the same x values will be
        less than or equal to this smoothing parameter.
    order: The desired order of the spline. If None, will be 1 if there are
        three or fewer input points, and otherwise 3.
    force_endpoints: if True (default), the endpoints of the spline will be set
        to match the positions of the input data, regardless of smoothing.
        NB: With large smoothing, this can dramatically influence the position
        of the entire curve. With smaller smoothing, changing the endpoints
        will have a more local effect.

    Returns a spline tuple (t,c,k) consisting of:
        t: the knots of the spline curve
        c: the b-spline coefficient for each knot
        k: the order of the spline.

    Note: the smoothing factor is an upper bound on the sum of all the squared
    distances between the original y values and the matching points on the
    smoothed spline representation."""

    x = numpy.asarray(x)
    y = numpy.asarray(y)
    l = len(x)
    w = numpy.ones(l, float)

    if periodic:
        force_endpoints = False

    if order is None:
        if l < 4:
            k = 1
        else:
            k = 3
    else:
        k = order
    if smoothing is None:
        smoothing = l * abs(x[0] - x[-1]) / 600.
    if force_endpoints:
        w[[0,-1]] = l

    t, c, ier, msg = splrep(x, y, s=smoothing, k=k, w=w, per=periodic)

    if ier > 3:
        raise RuntimeError(msg)
    if force_endpoints:
        # endpoints should already be close, so this shouldn't distort the spline too much...
        c[[0,-1]] = y[[0,-1]]

    return t, c, k

def spline_interpolate(tck, num_points, derivative=0):
    """Return num_points equally spaced along the given spline.

    If derivative=0, then the points themselves will be given; if derivative>0
    then the derivatives at those points will be returned."""
    # (t[0], t[-1]) gives the range of permissible values for the input position
    t, c, k = tck
    output_positions = numpy.linspace(t[0], t[-1], num_points)
    return spline_evaluate(tck, output_positions, derivative)

def spline_evaluate(tck, positions, derivative=0):
    """Evaluate a spline at the given positions.

    If derivative=0, then the points themselves will be given; if derivative>0
    then the derivatives at those points will be returned."""
    t, c, k = tck
    if c.ndim == 1:
        evaluate = splev # non-parametric
    else:
        evaluate = splpev # parametric
    return evaluate(positions, t, c, k, der=derivative)

def smooth_spline(tck, num_points, smoothing=1):
    """Smooth a spline by interpolating and then re-fitting with smoothing.

    Parameters:
        tck: parametric or nonparametric spline
        num_points: number of points to interpolate spline to.
        smoothing: the average distance between the original and smoothed splines
            will be less than this factor. Larger values generate smoother splines.

    Returns: smoothed tck
    """
    t, c, k = tck
    points = spline_interpolate(tck, num_points)
    smoothing *= num_points
    if c.ndim == 1:
        # non-parametric
        x = numpy.linspace(t[0], t[-1], num_points)
        return fit_nonparametric_spline(x, points, smoothing=smoothing)
    else:
        # parametric
        return fit_spline(points, smoothing=smoothing)

def reparameterize_spline(tck, num_points=None, smoothing=None):
    """Produce a spline where parameter values represent distance along the spline.

    Get as close as possible to the "natural parameterization" of the curve,
    where the parameter value represents the arc-length along the curve up to
    that point.

    Parameters:
        tck: parametric spline
        num_points: to compute the arc lengths for re-parameterization, the
            spline is first resampled. More points will produce a better
            reparameterization. If None, use a default that makes sense in the
            common case that 1 unit in space is on the order of the input error.
        smoothing: lower smoothing values give better reparameterizations, at
            the cost of more spline knots. Smoothing controls the total error
            of the reparameterization summed across all the points -- so if
            a smoothing is specified, make sure to scale it relative num_points.
            If None, use a sensible default.

    Returns: new tck spline.
    """
    if num_points is None:
        t_max = tck[0].max()
        num_points = int(t_max * 2)
    if smoothing is None:
        smoothing = num_points / 5000
    return fit_spline(spline_interpolate(tck, num_points), smoothing=smoothing)

def reverse_spline(tck):
    """Reverse the direction of a spline (parametric or nonparametric),
    without changing the range of the t (parametric) or x (nonparametric) values."""
    t, c, k = tck
    rt = t[-1] - t[::-1]
    rc = c[::-1]
    return rt, rc, k

def insert_control_points(tck, num_points):
    """Return an equivalent spline with additional control points added for
    improved editability.

    Parameters:
    tck: spline tuple
    num_points: total number of spline knots in returned spline tuple

    Returns: a new tck tuple"""
    t, c, k = tck
    if len(c.shape) > 1:
        # parametric spline: use pinsert
        insert_func = pinsert
    else:
        # non-parametric: use normal insert
        insert_func = insert
    while len(t) < num_points:
        spans = t[1:] - t[:-1]
        p = spans.argmax()
        new_t = t[p:p+2].mean()
        t, c, k = insert_func(t, c, k, new_t)
    return t, c, k

def spline_to_bezier(tck):
    """Convert a parametric spline into a sequence of Bezier curves of the same degree.

    Returns a list of Bezier curves of degree k that is equivalent to the input spline.

    Each Bezier curve is an array of shape (k+1,d) where d is the dimension of the
    space; thus the curve includes the starting point, the k-1 internal control
    points, and the endpoint, where each point is of d dimensions."""
    t, c, k = tck
    t = numpy.asarray(t)
    old_t = t
    try:
        c[0][0]
    except:
        # I can't figure out a simple way to convert nonparametric splines to
        # parametric splines. Oh well.
        raise TypeError("Only parametric splines are supported.")
    # the first and last k+1 knots are identical in the non-periodic case, so
    # no need to consider them when increasing the knot multiplicities below
    knots_to_consider = numpy.unique(t[k+1:-k-1])
    # For each unique knot, bring its multiplicity up to the next multiple of k+1
    # This removes all continuity constraints between each of the original knots,
    # creating a set of independent Bezier curves.
    desired_multiplicity = k+1
    for x in knots_to_consider:
        current_multiplicity = numpy.sum(old_t == x)
        remainder = current_multiplicity%desired_multiplicity
        if remainder != 0:
            # add enough knots to bring the current multiplicity up to the desired multiplicity
            number_to_insert = desired_multiplicity - remainder
            t, c, k = pinsert(t, c, k, x, number_to_insert)
    # group the points into the desired bezier curves
    return numpy.split(c, len(c) // desired_multiplicity)

def pinsert(t, c, k, u, m=1):
    """Insert m control points in spline (t,c,k) at parametric position u."""
    c = c.T
    out = numpy.empty((c.shape[1]+m, c.shape[0]))
    per=False
    for i, cc in enumerate(c):
        tt, ccc, ier = fitpack._fitpack._insert(per, t, cc, k, u, m)
        out[:,i] = ccc[:-k-1]
        if ier == 10: raise ValueError("Invalid input data")
        if ier: raise TypeError("An error occurred")
    return tt, out, k

def insert(t, c, k, x, m=1):
    """Insert m control points in spline (t,c,k) at parametric position x."""
    per=False
    t, c, ier = fitpack._fitpack._insert(per, t, c, k, x, m)
    if ier == 10: raise ValueError("Invalid input data")
    if ier: raise TypeError("An error occurred")
    return t, c, k

def splrep(x, y, s, k, w=None, per=False):
    """Return degree-k spline representation (t,c,k) of x, y points, with smoothing parameter s and weights w.

    Smoothing parameter guarantee:
    ((w * (splev(x, *tck) - y))**2).sum() <= s

    """
    m = x.shape[0]
    if w is None:
        w = numpy.ones(m, float)
    xb, xe = x[[0, -1]]
    if not (1 <= k <= 5): raise TypeError('1<=k=%d<=5 must hold'%(k))
    if (m != len(y)):
            raise TypeError('Lengths of the first two must be equal')
    if m <= k: raise TypeError('m>k must hold')
    if per:
        nest = m + 2 * k
        wrk = numpy.empty(m*(k + 1) + nest*(8 + 5*k), float)
    else:
        nest = m + k + 1
        wrk = numpy.empty(m*(k+1)+nest*(7+3*k), float)

    t = numpy.empty(nest, float)
    iwrk = numpy.empty(nest, numpy.int32)
    task = 0
    if per:
        n, c, fp, ier = fitpack.dfitpack.percur(task, x, y, w, t, wrk, iwrk, k, s)
    else:
        n, c, fp, ier = fitpack.dfitpack.curfit(task, x, y, w, t, wrk, iwrk, xb, xe, k, s)
    return t[:n], c[:n-k-1], ier, fitpack._iermess[ier][0]

def splprep(u, x, s, k, w=None, per=False):
    """Return spline representation (t,c,k) of parametric curve x(u), with smoothing parameter s and weights w.

    Parameters:
    u: array of shape (n) containing parametric positions
    x: array of shape (n, m) containing n points in m dimensions, which are the positions of the
       curve x(u) at each parametric value in the array u.
    s: smoothing parameter (see below)
    k: degree of output spline
    w: weights for each point
    per: construct a periodic spline

    Returns spline tuple (t,c,k)

    Smoothing parameter guarantee:
    ((w * (splpev(u, *tck) - x))**2).sum() <= s"""

    m, idim = x.shape
    if w is None:
        w = numpy.ones(m, float)
    ub, ue = u[[0, -1]]
    if not (1 <= k <= 5): raise TypeError('1<=k=%d<=5 must hold'%(k))
    if not len(u) == m:
            raise TypeError('Mismatch of input dimensions')
    if m <= k: raise TypeError('m>k must hold')
    if per:
        nest = m + 2 * k
    else:
        nest = m + k + 1
    t = numpy.array([], float)
    wrk = numpy.array([], float)
    iwrk = numpy.array([], numpy.int32)
    task = 0
    ipar = True
    t,c,o = fitpack._fitpack._parcur(x.ravel(), w, u, ub, ue, k, task, ipar, s, t, nest, wrk, iwrk, per)
    ier, fp, n = o['ier'], o['fp'], len(t)
    c.shape = (idim, n-k-1)
    return t, c.T, ier, fitpack._iermess[ier][0]

def splev(x, t, c, k, der=0):
    "Evaluate spline (t,c,k) or nth-order derivative thereof at position x."
    c = c.T
    out, ier = fitpack._fitpack._spl_(x, der, t, c, k, 0)
    if ier==10: raise ValueError("Invalid input data")
    if ier: raise TypeError("An error occurred")
    return out

def splpev(u, t, c, k, der=0):
    "Evaluate parametric spline (t,c,k) or nth-order derivative thereof at position u."

    c = c.T
    out = numpy.empty((len(u), c.shape[0]))
    for i, cc in enumerate(c):
        out[:,i], ier = fitpack._fitpack._spl_(u, der, t, cc, k, 0)
        if ier==10: raise ValueError("Invalid input data")
        if ier: raise TypeError("An error occurred")
    return out