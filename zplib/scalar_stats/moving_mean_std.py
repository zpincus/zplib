import numpy
from . import smoothing

def _est_moving_mean(xs, ys, points_out, smooth, iters, outlier_threshold):
    """sort xs and ys, fit a trendline, and return the sorted values, trendline,
    resampled x-values and resampled mean"""
    xs, ys = numpy.asarray(xs), numpy.asarray(ys)
    order = xs.argsort()
    xs = xs[order]
    ys = ys[order]
    mean_est = smoothing.lowess(xs, ys, smooth, iters, outlier_threshold)
    if isinstance(points_out, int):
        trend_x = numpy.linspace(xs[0], xs[-1], points_out)
    else:
        trend_x = points_out
    mean_trend = numpy.interp(trend_x, xs, mean_est, left=numpy.nan, right=numpy.nan)
    return xs, ys, mean_est, trend_x, mean_trend


def _est_moving_std(xs, ys, mean_est, smooth):
    y_dev = (ys - mean_est)**2
    # do not want to robustify against outlier deviations -- this
    # gives bad std values. So iter=1.
    var_est = smoothing.lowess(xs, y_dev, f=smooth, iters=1)
    # sometimes due to data sparsity and/or ringing artifacts in LOWESS, the
    # estimated variances can go to zero or below. Replace these with very tiny
    # positive values...
    small_compared_to_mean = numpy.absolute(mean_est)/10000
    bad_var = var_est < small_compared_to_mean
    var_est[bad_var] = small_compared_to_mean[bad_var]
    std_est = numpy.sqrt(var_est)
    return std_est

def moving_mean(xs, ys, points_out=300, smooth=0.2, iters=3, outlier_threshold=6):
    """Calculate smooth trendlines for the mean and standard deviation of
    a set of observations.

    Internally, LOWESS regression is used to estimate a robust mean trend, and
    then from that mean trend, the deviation of each data point is measured and
    LOWESS is again used to estimate a smooth trendline for this deviation.

    Parameters:
        xs, ys: 1-d lists or arrays of data points. Note that xs need not be
            sorted, nor unique. That is, the data need not describe a function:
            a cloud of points is appropriate here.
        points_out: number of points to evaluate the mean trendline along, or
            array of points at which to evaluate the trendline. Points outside
            the range of the input x-values will evaluate to nan: no extrapolation
            will be attempted.
        smooth: smoothing parameter 'f' for LOWESS. See smoothing.lowess().
        iters: robustifying iterations for LOWESS for calculating the mean
            trend. See smoothing.lowess().
        outlier_threshold: threshold for outlier exclusion in LOWESS. See
            smoothing.lowess().

    Returns trend_x, mean_trend
        trend_x: 1-d array (of length points_out) containing the x-values at which
            the smooth mean was calculated.
        mean_trend: mean trendlines evaluated at the trend_x positions.
    """
    xs, ys, mean_est, trend_x, mean_trend = _est_moving_mean(xs, ys, points_out, smooth, iters, outlier_threshold)
    return trend_x, mean_trend


def moving_mean_std(xs, ys, points_out=300, smooth=0.2, iters=3, outlier_threshold=6):
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
        smooth: smoothing parameter 'f' for LOWESS. See smoothing.lowess()
        iters: robustifying iterations for LOWESS for calculating the mean
            trend. See smoothing.lowess().
        outlier_threshold: threshold for outlier exclusion in LOWESS. See
            smoothing.lowess().

    Returns trend_x, mean_trend, std_trend
        trend_x: 1-d array of length points_out containing the x-values at which
            the mean and std outputs are evaluated.
        mean_trend, std_trend: y-values for the mean and std trendlines.
    """
    xs, ys, mean_est, trend_x, mean_trend = _est_moving_mean(xs, ys, points_out, smooth, iters, outlier_threshold)
    std_est = _est_moving_std(xs, ys, mean_est, smooth)
    std_trend = numpy.interp(trend_x, xs, std_est)
    return trend_x, mean_trend, std_trend


def z_transform(xs, ys, smooth=0.2, iters=3, outlier_threshold=6):
    """Calculate the z-scores of a set of observations.

    Internally, LOWESS regression is used to estimate a robust mean trend, and
    then from that mean trend, the deviation of each data point is measured and
    LOWESS is again used to estimate a smoothed standard deviation.

    The z-values returned are the number of standard deviations that each original
    observation is from the mean.

    Parameters:
        xs, ys: 1-d lists or arrays of data points. Note that xs need not be
            sorted, nor unique. That is, the data need not describe a function:
            a cloud of points is appropriate here.
        smooth: smoothing parameter 'f' for LOWESS. See smoothing.lowess()
        iters: robustifying iterations for LOWESS for calculating the mean
            trend. See smoothing.lowess().
        outlier_threshold: threshold for outlier exclusion in LOWESS. See
            smoothing.lowess().

    Returns mean_est, std_est, z_est
        mean_est, std_est: the smoothed value of the mean and standard deviation
            at each of the input data positions
        z_est: the number of standard deviations from the mean of each y-value.
    """
    xs, ys = numpy.asarray(xs), numpy.asarray(ys)
    mean_est = smoothing.lowess(xs, ys, smooth, iters, outlier_threshold)
    std_est = _est_moving_std(xs, ys, mean_est, smooth)
    return mean_est, std_est, (ys-mean_est)/std_est


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