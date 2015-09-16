import numpy
from scipy import ndimage

def find_local_maxima(image, min_distance):
    """Find maxima in an image.

    Finds the highest-valued points in an image, such that each point is
    separted by at least min_distance.

    If there are flat regions that are all at a maxima, the enter of mass of
    the region is reported. Large flat regions of more than min_distance in
    radius will be erroneously returned as maxima even if they are not. Further
    filtering should be performed to exclude these if needed.

    Returns the position of the maxima and the value at each maximum.

    Parameters:
        image: image of arbitrary dimensionality
        min_distance: maxima found will be at least this many pixels apart

    Returns:
        centroids: list of centers of each maxima
        values: image value at each maxima

  """
    image_max = ndimage.maximum_filter(image, size=2*min_distance+1, mode='constant')
    peak_mask = (image == image_max)
    # NB: some maxima might be marked by multiple contiguous pixels if the image
    # has "plateaus". So we need to label the mask and get the centroids
    # of each of the labeled regions.
    labeled_image, num_regions = ndimage.label(peak_mask)
    label_indices = numpy.arange(1, num_regions+1)
    centroids = ndimage.center_of_mass(peak_mask, labeled_image, label_indices)
    values = ndimage.mean(image, labeled_image, label_indices)
    return numpy.array(centroids), values