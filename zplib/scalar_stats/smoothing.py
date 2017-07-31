import numpy

def weighted_mean_and_std(x, w):
    """Return the mean and standard deviation of the data points x, weighted by
    the weights in w (which do not need to sum to 1)."""
    w = numpy.array(w, dtype=float)
    w /= w.sum()
    x = numpy.asarray(x)
    weighted_mean = (w*x).sum()
    squared_diff = (x - weighted_mean)**2
    weighted_var = (w * squared_diff).sum()
    return weighted_mean, numpy.sqrt(weighted_var)

def weighted_mean(x, w):
    """Return the mean of the data points x, weighted by the weights in w
    (which do not need to sum to 1)."""
    w = numpy.array(w, dtype=float)
    w /= w.sum()
    x = numpy.asarray(x)
    return (w*x).sum()

def _gaussian(x, mu=0, sigma=1):
  return ( (1/numpy.sqrt(2 * numpy.pi * sigma**2) * numpy.exp(-0.5 * ((numpy.asarray(x)-mu)/sigma)**2)) )

def gaussian_mean(x, y, p, std=1):
    """Given a set of positions x where values y were observed, calculate
    the gaussian-weighted mean of those values at a set of new positions p,
    where the gaussian has a specied standard deviation.
    """
    return numpy.array([weighted_mean(y, _gaussian(x, mu=pp, sigma=std)) for pp in p])


def savitzky_golay(data, kernel=11, order=4):
    """Apply Savitzky-Golay smoothing to the input data.

    http://en.wikipedia.org/wiki/Savitzky-Golay_filter
    """
    kernel = abs(int(kernel))
    order = abs(int(order))
    if kernel % 2 != 1 or kernel < 1:
        raise TypeError("kernel size must be a positive odd number, was: %d" % kernel)
    if kernel < order + 2:
        raise TypeError("kernel is to small for the polynomals\nshould be > order + 2")
    order_range = range(order+1)
    half_window = (kernel-1) // 2
    m = numpy.linalg.pinv([[k**i for i in order_range] for k in range(-half_window, half_window+1)])[0]
    window_size = len(m)
    half_window = (window_size-1) // 2
    offsets = range(-half_window, half_window+1)
    offset_data = list(zip(offsets, m))
    smooth_data = list()
    data = numpy.concatenate((numpy.ones(half_window)*data[0], data, numpy.ones(half_window)*data[-1]))
    for i in range(half_window, len(data) - half_window):
        value = 0.0
        for offset, weight in offset_data:
            value += weight * data[i + offset]
        smooth_data.append(value)
    return numpy.array(smooth_data)

def lowess(x, y, f=2/3., iters=3):
    """Apply LOWESS to fit a nonparametric regression curve to a scatterplot.

    http://en.wikipedia.org/wiki/Local_regression

    Parameters:
        x, y: 1-d arrays containing data points in x and y.
        f: smoothing parameter in range [0, 1]. Lower values = less smoothing.
        iter: number of robustifying iterations (after each of which outliers
            are detected and excluded). Larger numbers = more robustness, but
            slower run-time.

    Returns: array of smoothed y-values for the input x-values.
    """
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    r = max(4, int(numpy.ceil(f*(len(x)-1))))
    # below hogs RAM for large input, without much speed gain.
    # h = [numpy.sort(numpy.abs(x - xv))[r] for xv in x]
    # w = numpy.clip(numpy.abs(numpy.subtract.outer(x, x) / h), 0, 1)
    # w = (1 - w**3)**3
    delta = 1
    max_dists = numpy.empty_like(x)
    for it in range(iters):
        y_est = []
        for i, xv in enumerate(x): # for xv, wv in zip(x, w.T):
            x_dists = numpy.abs(x - xv)
            if it == 0:
                max_dist = numpy.partition(x_dists, r)[r]
                max_dists[i] = max_dist
            else:
                 max_dist = max_dists[i]
            wv = numpy.clip(x_dists/max_dist, 0, 1)
            wv = (1 - wv**3)**3
            weights = delta * wv
            weights_mul_x = weights * x
            b1 = numpy.dot(weights, y)
            b2 = numpy.dot(weights_mul_x, y)
            A11 = numpy.sum(weights)
            A12 = numpy.sum(weights_mul_x)
            A21 = A12
            A22 = numpy.dot(weights_mul_x, x)
            determinant = A11 * A22 - A12 * A21
            beta1 = (A22 * b1-A12 * b2) / determinant
            beta2 = (A11 * b2-A21 * b1) / determinant
            y_est.append(beta1 + beta2 * xv)
        y_est = numpy.array(y_est)
        residuals = y - y_est
        s = numpy.median(numpy.abs(residuals))
        if s > 0:
            delta = numpy.clip(residuals / (6 * s), -1, 1)
            delta = (1 - delta**2)**2
    return numpy.array(y_est)


def robust_polyfit(x, y, degree=2, iters=3):
    """Fit a polynomial to scattered data, robust to outliers.

    Parameters:
        x, y: 1-d arrays containing data points in x and y.
        degree: degree of the polynomial to fit.
        iter: number of robustifying iterations (after each of which outliers
            are detected and excluded). Larger numbers = more robustness, but
            slower run-time.

    Returns: polynomial coefficients, array of smoothed y-values for the input x-values.
    """
    x, y = numpy.asarray(x), numpy.asarray(y)
    weights = numpy.ones(len(x), float) # delta in original formulation
    for _ in range(iters):
        cs = numpy.polynomial.polynomial.polyfit(x, y, degree, w=weights)
        y_est = numpy.polynomial.polynomial.polyval(x, cs)
        residuals = y - y_est
        s = numpy.median(numpy.abs(residuals))
        if s > 0:
            weights = (residuals / (6 * s)).clip(-1, 1)
            weights = (1 - weights**2)**2
    return cs, y_est
