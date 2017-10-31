import collections

import numpy
from scipy.stats import kde
from skimage import measure

def density_at_points(data):
    """Use KDE to calculate the probability density at each point in a dataset.

    Useful for coloring points in scatterplot by the density, to better help
    visualize crowded regions of the plot.

    Parameter:
        data: array of shape (n_data_points, n_dimensions)

    Returns:
        densities: array of shape (n_data_points)

    Example:
        import numpy
        import matplotlib.pyplot as plt
        # prepare some data
        mode1 = numpy.random.multivariate_normal(mean=[0, 0], cov=[[4, 1], [1, 7]], size=300)
        mode2 = numpy.random.multivariate_normal(mean=[8, 8], cov=[[2, 1], [1, 1]], size=300)
        data = numpy.concatenate([mode1, mode2], axis=0)

        # calculate the contours
        density = density_at_points(data)

        # plot the data
        plt.scatter(data[:,0], data[:,1], s=12, c=density, cmap='inferno')
    """
    data = numpy.asarray(data)
    kd = kde.gaussian_kde(data.T)
    return kd(data.T)


ContourResult = collections.namedtuple('ContourResult', ('density', 'extent', 'c_level', 'contours'))

def contour(data, fraction, fraction_of='density', samples_x=100, samples_y=100):
    """Calculate contours that enclose a given fraction of the input data.

    Use KDE to estimate the density of a scattered 2D dataset, and calculate
    from that the contours of the density function which enclose a given fraction
    of the total density, or a given fraction of the input data points.

    By design, KDE places density beyond the data points. Thus a contour
    containing a specified fraction of the density will be larger than a
    contour containing the same fraction of the data points. Indeed, the former
    contour may well contain rather more of the data points.

    One or more input fractions may be provided. As the density function may be
    multimodal, more than one contour may be returned for each input fraction.

    Parameters:
        data: 2D dataset, of shape (n_points, 2)
        fraction: a single value in the range (0, 1), or an array of the same.
        fraction_of: either 'density' or 'points' (see above).
        samples_x, samples_x: resolution of the x, y grid along which to
            estimate the data density.

    Returns:
        density: array of shape (samples_x, samples_y) containing density
            estimates. (Will not sum to one, as these are point estimates at the
            centers of each pixel, not the integral over each pixel's area.)
        extent: tuple (xmin, xmax, ymin, ymax) that represents the spatial extent
            of the density array.
        c_level: level or a list of levels if multiple fractions were provided.
            If fraction_of='density', the following approximates each fraction:
                density[density >= c_level].sum() / density.sum()
            If fraction_of='points', the following approximates each fraction:
                (data_density >= c_level).sum() / len(data_density)
            Where data_density is the KDE estimate of the density at each data
            point. (The function density_at_points can calculate this.)
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

    fraction = numpy.asarray(fraction)
    orig_dim = fraction.ndim
    if orig_dim == 0:
        fraction = fraction.reshape(1)
    assert numpy.all((fraction > 0) & (fraction < 1))

    assert fraction_of in ('density', 'points')

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

    kd = kde.gaussian_kde(data.T)
    # now flatten the grid to a list of (x, y) points, evaluate the density,
    # and expand back to a grid.
    density = kd(indices.reshape(2, samples_x * samples_y)).reshape(samples_x, samples_y)

    if fraction_of == 'density':
        # find density levels where a given fraction of the total density is above
        # the level.
        density_values = numpy.sort(density.flat)
        density_below_value = ordered_density.cumsum()
        total_density = density_below_value[-1]
        # we now have a mapping between density values and the total amount of density
        # below that value. To find the desired density levels (where a given fraction
        # of the total density is above that level), just use that mapping:
        c_levels = numpy.interp((1-fraction)*total_density, density_below_value, density_values)
    else:
        # find density levels where a given fraction of the input data points are
        # above the level
        data_density = kd(data.T)
        c_levels = numpy.percentile(data_density, (1-fraction)*100)

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
