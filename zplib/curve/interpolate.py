import numpy
from scipy.interpolate import fitpack

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
    tck = fit_splines(points)
    points_out = spline_interpolate(tck, num_points)
    return points_out, tck


def fit_spline(points, smoothing=None, order=None):
    """Fit a parametric smoothing spline to a given set of x,y points.

    Parameters:
    points: array of n points x,y; shape=(n,2)
    smoothing: smoothing factor: 0 requires perfect interpolation of the
        input points, at the cost of potentially high noise. Very large values
        will result in a low-order polynomial fit to the points. If None, an
        appropriate value based on the scale of the points will be selected.
    order: The desired order of the spline. If None, will be 1 if there are
        three or fewer input points, and otherwise 3.

    Returns a spline tuple (t,c,k) consisting of:
        t: the knots of the spline curve
        c: the x and y b-spline coefficients for each knot
        k: the order of the spline.

    Note: smoothing factor "s" is an upper bound on the sum of all the distances
    between the original x,y points and the matching points on the smoothed
    spline representation."""
    points = numpy.asarray(points)
    l = len(points)
    if order is None:
        if l < 4:
            k = 1
        else:
            k = 3
    else:
        k = order
    # choose input parameter values for the curve as the distances along the polyline:
    # this gives something close to the "natural parameterization" of the curve.
    # (i.e. a parametric curve with first-derivative close to unit magnitude: the curve
    # doesn't accelerate/decelerate, so points along the curve in the x,y plane
    # don't "bunch up" with evenly-spaced parameter values.)
    distances = geometry.cumulative_distances(points, unit=False)

    if smoothing is None:
        smoothing = l * distances[-1] / 600.

    t, c, ier, msg = splprep(distances, points, s=smoothing, k=k)
    if ier > 3:
        raise RuntimeError(msg)
    c[[0,-1]] = points[[0,-1]]
    return t,c,k


def spline_interpolate(tck, num_points, derivative=0):
    """Return num_points equally spaced along the given spline.

    If derivative=0, then the points themselves will be given; if derivative>0
    then the derivatives at those points will be returned."""
    t, c, k = tck
    # t[-1] gives the maximum parameter value for the parametric curve
    output_positions = numpy.linspace(0, t[-1], num_points)
    if c.ndim == 1:
        evaluate = splev # non-parametric
    else:
        evaluate = splpev # parametric
    points = evaluate(output_positions, t, c, k, der=derivative)
    return points


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
    # strip off the last k+1 knots, as they are redundant after knot insertion
    bezier_points = numpy.transpose(c)[:-desired_multiplicity]
    # group the points into the desired bezier curves
    return numpy.split(bezier_points, len(bezier_points) / desired_multiplicity, axis=0)


def pinsert(t, c, k, u, m=1):
    """Insert m control points in spline (t,c,k) at parametric position u."""
    c = c.T
    out = numpy.empty((c.shape[1]+m, c.shape[0]))
    per=False
    for i, cc in enumerate(c):
        tt, ccc, ier = fitpack._fitpack._insert(per, t, cc, k, u, m)
        out[:,i] = ccc[:-k-1]
        if ier==10: raise ValueError("Invalid input data")
        if ier: raise TypeError("An error occurred")
    return tt, out, k


def insert(t, c, k, x, m=1):
    """Insert m control points in spline (t,c,k) at parametric position x."""
    per=False
    t, c, ier = fitpack._fitpack._insert(per, t, c, k, x, m)
    if ier==10: raise ValueError("Invalid input data")
    if ier: raise TypeError("An error occurred")
    return t, c, k


def splrep(x, y, s, k):
    """Return degree-k spline representation (t,c,k) of x,y points, with smoothing parameter s.

    Smoothing parameter guarantee:
    numpy.absolute(splev(x, *tck) - y).sum() <= s"""
    m = x.shape[0]
    w = numpy.ones(m,float)
    xb, xe = x[[0, -1]]
    if not (1<=k<=5): raise TypeError('1<=k=%d<=5 must hold'%(k))
    if (m != len(y)):
            raise TypeError('Lengths of the first two must be equal')
    if m<=k: raise TypeError('m>k must hold')
    nest=m+k+1
    t = numpy.empty((nest,),float)
    wrk = numpy.empty((m*(k+1)+nest*(7+3*k),), float)
    iwrk = numpy.empty((nest,), numpy.int32)
    task = 0
    n, c, fp, ier = fitpack.dfitpack.curfit(task, x, y, w, t, wrk, iwrk, xb, xe, k, s)
    return t[:n], c[:n-k-1], ier, fitpack._iermess[ier][0]


def splprep(u, x, s, k):
    """Return spline representation (t,c,k) of parametric curve x(u), with smoothing parameter s.

    Parameters:
    u: array of shape (n) containing parametric positions
    x: array of shape (n, m) containing n points in m dimensions, which are the positions of the
       curve x(u) at each parametric value in the array u.
    s: smoothing parameter (see below)
    k: degree of output spline

    Returns spline tuple (t,c,k)

    Smoothing parameter guarantee:
    numpy.linalg.norm(splpev(u, *tck) - x).sum() <= s"""

    m, idim = x.shape
    w = numpy.ones(m,float)
    ub, ue = u[[0, -1]]
    if not (1<=k<=5): raise TypeError('1<=k=%d<=5 must hold'%(k))
    if not len(u)==m:
            raise TypeError('Mismatch of input dimensions')
    if m<=k: raise TypeError('m>k must hold')
    nest=m+k+1

    t = numpy.array([],float)
    wrk = numpy.array([],float)
    iwrk = numpy.array([],numpy.int32)
    task = per = 0
    ipar = True
    t,c,o = fitpack._fitpack._parcur(x.ravel(),w,u,ub,ue,k,task,ipar,s,t,nest,wrk,iwrk,per)
    ier, fp, n = o['ier'], o['fp'], len(t)
    c.shape=idim,n-k-1
    return t, c.T, ier, fitpack._iermess[ier][0]


def splev(x, t, c, k, der=0):
    "Evaluate spline (t,c,k) or nth-order derivative thereof at position x."
    c = c.T
    try:
        out, ier = fitpack._fitpack._spl_(x, der, t, c, k, 0)
    except:
        out, ier = fitpack._fitpack._spl_(x, der, t, c, k)
    if ier==10: raise ValueError("Invalid input data")
    if ier: raise TypeError("An error occurred")
    return out


def splpev(u, t, c, k, der=0):
    "Evaluate parametric spline (t,c,k) or nth-order derivative thereof at position u."

    c = c.T
    out = numpy.empty((len(u), c.shape[0]))
    for i, cc in enumerate(c):
        try:
            out[:,i], ier = fitpack._fitpack._spl_(u, der, t, cc, k, 0)
        except:
            out[:,i], ier = fitpack._fitpack._spl_(u, der, t, cc, k)
        if ier==10: raise ValueError("Invalid input data")
        if ier: raise TypeError("An error occurred")
    return out

