import numpy
from scipy import ndimage

def hysteresis_threshold(array, low_threshold, high_threshold, structure=None):
    """Create a mask that is True for regions in the input array which are
    entirely larger than low_threshold and which contain at least one element
    larger than high_threshold."""
    high_mask = array > high_threshold
    low_mask = array > low_threshold
    return ndimage.binary_propagation(high_mask, mask=low_mask, structure=structure)

def remove_small_radius_objects(mask, max_radius):
    """Remove objects from the mask up to max_radius (in terms of number of erosion
    iterations required to remove the object.)

    Returns a new mask with the small objects removed."""
    eroded = ndimage.binary_erosion(mask, iterations=max_radius)
    return ndimage.binary_propagation(eroded, mask=mask)

def remove_edge_objects(mask):
    """Remove objects from the mask that are in contact with the edge.

    Returns a new mask with the edge objects removed."""
    edge_objects = ndimage.binary_propagation(numpy.zeros_like(mask), mask=mask,
        border_value=1)
    return mask & ~edge_objects

def fill_small_radius_holes(mask, max_radius):
    """Fill holes in the mask that are up to max_radius (in terms of the number
    of dilation iterations required to fill them).

    See also ndimage.binary_fill_holes if you want to fill all holes regardless
    of size.

    Returns a new mask with the small holes filled."""
    mask = mask.astype(bool)
    outside = ndimage.binary_propagation(numpy.zeros_like(mask), mask=~mask, border_value=1)
    holes = ~(mask | outside)
    large_hole_centers = ndimage.binary_erosion(holes, iterations=max_radius+1)
    large_holes = ndimage.binary_propagation(large_hole_centers, mask=holes)
    small_holes = holes ^ large_holes
    return mask | small_holes

def get_background(mask, offset_radius, background_radius):
    """Given a mask, return a mask containing a 'band' of True values
    'background_radius' wide, offset outward from the masked regions by
    'offset_radius'.

    Useful for calculating the local image background intensity near, but not
    too near, to an object. (Due to blur and imperfect thresholding, it is often
    the case that pixels directly adjacent to an identified bright object are
    somewhat above the background intensity. So a moderate offeset is often useful.)
    """
    offset = ndimage.binary_dilation(mask, iterations=offset_radius)
    background = ndimage.binary_dilation(offset, iterations=background_radius)
    return background ^ offset

def get_areas(mask, structure=None):
    """Find the areas of the specific regions in a mask.

    Returns labels, region_indices, areas
        labels: array where each contiguous region in the mask is 'labeled' with
            a unique positive integer. (Output of scipy.ndimage.label.)
        region_indices: list of each of these positive integers.
        areas: number of pixels in each region, in the order specified by
            'region_indices'.
    """
    labels, num_regions = ndimage.label(mask, structure=structure)
    region_indices = numpy.arange(1, num_regions + 1)
    areas = ndimage.sum(numpy.ones_like(mask), labels=labels, index=region_indices)
    return labels, region_indices, areas

def get_largest_object(mask, structure=None):
    """Return a mask containing the single largest region in the input mask."""
    labels, region_indices, areas = get_areas(mask, structure)
    if len(region_indices) == 0:
        # no regions in the first place...
        return numpy.zeros(mask.shape, dtype=bool)
    largest = region_indices[areas.argmax()]
    return labels == largest

def remove_small_area_objects(mask, max_area, structure=None):
    """Remove objects from the mask that are smaller than 'max_area' (in terms of
    pixel-wise area).

    Returns a new mask with the small objects removed."""
    labels, region_indices, areas = get_areas(mask, structure)
    keep_labels = areas > max_area
    keep_labels = numpy.concatenate(([False], keep_labels))
    return keep_labels[labels]

def fill_small_area_holes(mask, max_area):
    """Fill holes in the mask that are up to 'max area' (in terms of pixel-wise
    area).

    See also ndimage.binary_fill_holes if you want to fill all holes regardless
    of size.

    Returns a new mask with the small holes filled."""
    mask = mask.astype(bool)
    outside = ndimage.binary_propagation(numpy.zeros_like(mask), mask=~mask, border_value=1)
    holes = ~(mask | outside)
    labels, region_indices, areas = get_areas(holes)
    kill_labels = areas <= max_area
    kill_labels = numpy.concatenate(([False], kill_labels))
    small_holes = kill_labels[labels]
    return mask | small_holes
