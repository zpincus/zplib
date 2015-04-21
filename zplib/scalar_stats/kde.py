import numpy
from scipy.stats import kde

def kd_distribution(data, x_min=None, x_max=None, num_points=200, survival=False):
    """Use Kernel Density Estimation to estimate a density or survival function from data.

    Parameters:
        data: 1-dimensional list or array of data points
        x_min, x_max: if None, the min and max of the data points (plus a little fudge
            factor so the distributions tail off more nicely) are used. Otherwise,
            these values specify the region over which to evaluate the density.
        num_points: number of points to evaluate the density along. These will NOT
            be evenly-spaced, but will be more dense in regions where the density
            function is larger.
        survival: if True, return the survival function of the data. Otherwise return
            the estimated density function.

    Returns: xs, ys, kd_estimator
        xs, ys are 1-d arrays containing the x and y values of the estimated density.
        kd_estimator is a scipy.stats.kde.gaussian_kde object that can be used to
            estimate the density of the data (etc.) at additional points.
    """
    data = numpy.asarray(data, dtype=numpy.float32)
    kd_estimator = kde.gaussian_kde(data)
    if x_min is None or x_max is None:
        data_min = data.min()
        data_max = data.max()
        extra = 0.2*(data_max - data_min)
        data_min -= extra
        data_max += extra
        if x_min is None:
            x_min = data_min
        if x_max is None:
            x_max = data_max
    # ignore underflow -- KDE can underflow when estimating regions of very low density
    err = numpy.seterr(under='ignore')
    # use a two-tier approach to generating points along which to evaluate the density function
    # (1) take linearly spaced samples between min and max. However, this fails for very
    # narrow distributions that might not be adequately sampled so also,
    # (2) generate sample points from the distribution itself.
    linear_xs = numpy.linspace(x_min, x_max, num_points//2)
    data_samples = kd_estimator.resample(num_points//2)[0]
    data_samples = data_samples[(data_samples >= x_min) & (data_samples <= x_max)]
    xs = numpy.sort(numpy.concatenate([linear_xs, data_samples]))
    if survival:
        ys = [1 - kd_estimator.integrate_box_1d(-numpy.inf, x) for x in xs]
    else:
        ys = kd_estimator(xs)
    numpy.seterr(**err)
    return xs, ys, kd_estimator