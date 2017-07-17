import numpy
from . import smoothing

def moving_mean_std(xs, ys, points_out=300, smooth=0.2):
    """Calculate smooth trendlines for the mean and standard deviation of
    a set of observations.

    Internally, LOWESS regression is used to estimate a robust mean trend, and
    then from that mean trend, the deviation of each data point is measured and
    LOWESS is again used to estimate a smooth trendline for this deviation.

    Parameters:
        xs, ys: 1-d lists or arrays of data points. Note that xs need not be
            sorted, nor unique. That is, the data need not describe a function:
            a cloud of points is appropriate here.
        points_out: number of points to evaluate the mean and std trendlines along.
        smooth: smoothing parameter 'f' for LOWESS. See smoothing1d.lowess()

    Returns x_out, mean, std
        x_out: 1-d array of length points_out containing the x-values at which
            the mean and std outputs are evaluated.
        mean, std: y-values for the mean and std trendlines.
    """
    xs, ys = numpy.asarray(xs), numpy.asarray(ys)
    order = xs.argsort()
    xs = xs[order]
    ys = ys[order]
    y_est = smoothing.lowess(xs, ys, f=smooth, iters=3)
    y_dev = (ys - y_est)**2
    # do not want to robustify against outlier deviations -- this
    # gives bad std values. So iter=1.
    var_est = smoothing.lowess(xs, y_dev, f=smooth, iters=1)
    # sometimes due to data sparsity and/or ringing artifacts in LOWESS, the
    # estimated variances can go to zero or below. Replace these with very tiny
    # positive values...
    small_compared_to_yest = numpy.absolute(y_est)/10000
    bad_var = var_est < small_compared_to_yest
    var_est[bad_var] = small_compared_to_yest[bad_var]
    x_out = numpy.linspace(xs[0], xs[-1], points_out)
    mean = numpy.interp(x_out, xs, y_est)
    std = numpy.interp(x_out, xs, numpy.sqrt(var_est))
    return x_out, mean, std

class MovingMeanSTD(object):
    """Given x_out, mean, and std calculated by moving_mean_std(), construct an
    object which uses linear interpolation to estimate the mean and std at any
    additional x positions.
    """
    def __init__(self, x_out, mean, std):
        self.x_out = x_out
        self._mean = mean
        self._std = std

    def mean(self, value):
        """Return the mean trendline at the point (or points) in the value parameter."""
        return numpy.interp(value, self.x_out, self._mean)

    def std(self, value):
        """Return the standard deviation trendline at the point (or points) in the value parameter."""
        return numpy.interp(value, self.x_out, self._std)

    def z_line(self, sigma=0):
        """Return the trendline for mean + sigma standard deviations.

        Returns x_out, z_line, where each are a 1-d array of values.
        """
        return self.x_out, self._mean+sigma*self._std