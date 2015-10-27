import numpy
from scipy.stats import kde

def kd_distribution(data, x_min=None, x_max=None, num_points=200, CDF=False):
    """Use Kernel Density Estimation to estimate a density or CDF from data.

    Parameters:
        data: 1-dimensional list or array of data points
        x_min, x_max: if None, the min and max of the data points (plus a little fudge
            factor so the distributions tail off more nicely) are used. Otherwise,
            these values specify the region over which to evaluate the density.
        num_points: number of points to evaluate the density along. These will NOT
            be evenly-spaced, but will be more dense in regions where the density
            function is larger.
        CDF: if True, return the cumulative distribution function of the data.
            Otherwise return the estimated density function.

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
    with numpy.errstate(under='ignore'):
        # use a two-tier approach to generating points along which to evaluate the density function
        # (1) take linearly spaced samples between min and max. However, this fails for very
        # narrow distributions that might not be adequately sampled so also,
        # (2) generate sample points from the distribution itself.
        data_samples = kd_estimator.resample(num_points//2)[0]
        linear_xs = numpy.linspace(x_min, x_max, num_points//2)
        data_samples = data_samples[(data_samples >= x_min) & (data_samples <= x_max)]
        xs = numpy.sort(numpy.concatenate([linear_xs, data_samples]))
        if CDF:
            ys = numpy.array([kd_estimator.integrate_box_1d(-numpy.inf, x) for x in xs])
        else:
            ys = kd_estimator(xs)
    return xs, ys, kd_estimator

class FixedBandwidthKDE(kde.gaussian_kde):
    """Gaussian KDE class that allows for a specified, data-independent bandwidth.

    Provided a shape (d, n) array of n data points in d dimensions, or a shape
    (n,) array of n 1-d data points, use Gaussian Kernel Density Estimation to
    calculate the approximate probability distribution of those points.

    scipy.stats.kde.gaussian_kde always defines the bandwidth (e.g. standard
    deviation of the Gaussian, for the 1D case) as a scalar multiple of the data's
    variance or covariance in the d-dimensional case. This is not always useful,
    so this class allows for a fixed bandwidth to be specified in advance.

    The bandwidth may be either a scalar, which will isotropically set the
    covariance of the Gaussian to that value in all d dimensions, or a vector
    of length d, which will provide the bandwidth in each dimension separately.

    Alternately, a full covariance matrix may be specified via set_covariance()

    """
    def __init__(self, dataset, bandwidth=1):
        self.dataset = numpy.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d, self.n = self.dataset.shape
        self.set_bandwidth(bandwidth)

    def set_bandwidth(self, bandwidth):
        """Set the bandwidth to a scalar value or to a vector with one bandwidth
        parameter per dimension."""
        if numpy.isscalar(bandwidth):
            covariance = numpy.eye(self.d) * bandwidth**2
        else:
            bandwidth = numpy.asarray(bandwidth)
            if bandwidth.shape == (self.d,): # bandwidth is 1-d array of len self.d
                covariance = numpy.diag(bandwidth**2)
            else:
                raise ValueError("'bandwidth' should be a scalar, a 1-d array of length d.")
        self.set_covariance(covariance)

    def set_covariance(self, covariance):
        """Set a full covariance matrix for the Gaussian KDE."""
        self.covariance = numpy.asarray(covariance)
        if covariance.shape != (self.d, self.d):
            raise ValueError("'covariance' must be a square 2d array of shape (d, d).")
        self.inv_cov = numpy.linalg.inv(self.covariance)
        self._norm_factor = numpy.sqrt(numpy.linalg.det(2*numpy.pi*self.covariance)) * self.n

