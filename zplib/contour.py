import collections

import numpy
from scipy.stats import kde
from skimage import measure
from scipy import optimize

ContourResult = collections.namedtuple('ContourResult', ('density', 'extent', 'c_level', 'contours'))

def contour(data, fraction, samples_x=100, samples_y=100):
    """Calculate contours that enclose a given fraction of the input data points.

    Use KDE to estimate the density of a scattered 2D dataset, and calculate
    from that the contours of the density function which enclose a given fraction
    of the total density. Specifically, the KDE density will be estimated across
    an (x, y) grid, and contours of that estimated function will be calculated.

    One or more input fractions may be provided. As the density function may be
    multimodal, more than one contour may be returned for each input fraction.

    Parameters:
        data: 2D dataset, of shape (n_points, 2)
        fraction: a single value in the range (0, 1), or an array of the same.
        samples_x, samples_x: resolution of the x, y grid along which to
            estimate the data density.

    Returns:
        density: array of shape (samples_x, samples_y) containing the normalized
            density estimates. (Should sum to almost one.)
        extent: tuple (xmin, xmax, ymin, ymax) that represents the spatial extent
            of the density array.
        c_level: level of the density array that contains the given fraction of
            the density (i.e. density[density >= c_level].sum() is approximately
            the given fraction), or a list of levels if multiple fractions were
            provided.
        contours: a list of contours (if a single value is provided for fraction)
            or a list of lists of contours (one list for each fraction). Each
            contour is an array of shape (n_points, 2).

    Examples:
        import numpy
        import matplotlib.pyplot as plt
        # prepare some data
        mode1 = numpy.random.multivariate_normal(mean=[0, 0], cov=[[4, 1], [1, 7]], size=300)
        mode2 = numpy.random.multivariate_normal(mean=[8, 8], cov=[[2, 1], [1, 1]], size=300)
        data = numpy.concatenate([mode1, mode2], axis=0)

        # calculate the contours
        density, extent, c_level, contours = contour(data, [0.25, 0.5])

        # plot the data
        plt.scatter(data[:,0], data[:,1], s=12, color='blue')

        # plot the density (note imshow expects images to have shape (y, x)...)
        plt.imshow(density.T, extent=extent, origin='lower')

        # plot the contours for the fractions 0.25 and 0.5
        for level_contours, color in zip(contours, ['red', 'orange']):
            # there may be several contours for each level
            for c in level_contours:
                plt.plot(c[:,0], c[:,1], color=color)
    """
    data = numpy.asarray(data)
    assert data.ndim == 2 and data.shape[1] == 2
    kd = kde.gaussian_kde(data.T)

    # now calculate the spatial extent over which to get the KDE estimates of
    # the density function.
    # We must extend the mins and maxes a bit because KDE puts weight all around
    # each data point. So to get close to the full density function, we need to
    # evaluate the function a little ways away from the extremal data points.
    maxes = data.max(axis=0)
    mins = data.min(axis=0)
    extra = 0.2*(maxes - mins)
    maxes += extra
    mins -= extra
    xmax, ymax = maxes
    xmin, ymin = mins

    # make a grid of points from the min to max positions in two dimensions
    indices = numpy.mgrid[xmin:xmax:samples_x*1j, ymin:ymax:samples_y*1j]
    # now flatten the grid to a list of (x, y) points, evaluate the density,
    # and expand back to a grid.
    density = kd(indices.reshape(2, samples_x * samples_y)).reshape(samples_x, samples_y)
    # See what the total integral of the KDE should be in this range.
    # As gaussians have infinite support, we won't ever get quite to 1, but this
    # generally gets close
    total_in_range = kd.integrate_box([xmin, ymin], [xmax, ymax])
    # rescale the density so that it sums to the expected integral.
    density /= density.sum() / total_in_range
    fraction = numpy.asarray(fraction)

    orig_dim = fraction.ndim
    if orig_dim == 0:
        fraction = fraction.reshape(1)
    assert numpy.all((fraction > 0) & (fraction < 1))
    # sort the fractions for efficient calculation. We'll unsort later.
    order = fraction.argsort()
    fraction = fraction[order]
    if fraction[-1] > total_in_range:
        # We can't calculate a contour larger than the total integrated density
        # across the grid. This should be close to 1, but isn't always.
        raise ValueError(f'cannot calculate fraction less than {total_in_range}')

    # now use Brent's algorithm to search for the density levels which enclose
    # each given fraction of the data
    low = density.min()
    high = density.max()
    c_levels = []
    for target in fraction:
        # Brent's algorithm finds roots, so the objective needs to cross zero at
        # the desired contour level
        def objective(level):
            return density[density >= level].sum() - target
        c_level = optimize.brentq(objective, low, high, xtol=(high-low)/1000)
        high = c_level # we're going in sorted order, so the next contour up is strictly larger
        c_levels.append(c_level)
    c_levels = numpy.array(c_levels)
    c_levels = c_levels[order.argsort()] # un-sort back into original order

    # now find the contours in the density array at the desired levels
    contours = []
    for c_level in c_levels:
        level_contours = measure.find_contours(density, c_level)
        # The contours are in units of the indices into the density array.
        # Scale these to the the spatial extent of the data
        for rc in level_contours:
            rc /= [samples_x, samples_y] # first scale to [0, 1] in each dim
            rc *= (maxes - mins) # then scale out to the desired min and max
            rc += mins
        contours.append(level_contours)

    if orig_dim == 0:
        contours = contours[0]
        c_levels = c_levels[0]
    extent = [xmin, xmax, ymin, ymax]
    return ContourResult(density, extent, c_levels, contours)