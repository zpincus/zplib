import numpy
from scipy import ndimage

def hysteresis_threshold(array, low_threshold, high_threshold):
    """Create a mask with all regions in the array with values > low_threshold
    and which contain at least one value > high_threshold."""
    high_mask = array > high_threshold
    low_mask = array > low_threshold
    return ndimage.binary_dilation(high_mask, mask=low_mask, iterations=-1)

def remove_small_radius_objects(mask, max_radius):
    """Remove objects from the mask up to max_radius (in terms of number of erosion
    iterations required to remove the object.)"""
    eroded = ndimage.binary_erosion(mask, iterations=max_radius)
    return ndimage.binary_dilation(eroded, mask=mask, iterations=-1)

def remove_edge_objects(mask):
    edge_objects = ndimage.binary_dilation(numpy.zeros_like(mask), mask=mask,
        border_value=1, iterations=-1)
    return binary_mask & ~edge_objects

def fill_small_radius_holes(mask, max_radius):
    outside = ndimage.binary_dilation(numpy.zeros_like(mask), mask=~mask, iterations=-1, border_value=1)
    holes = ~(mask | outside)
    large_hole_centers = ndimage.binary_erosion(holes, iterations=max_radius)
    large_holes = ndimage.binary_dilation(large_hole_centers, mask=holes, iterations=-1)
    small_holes = holes ^ large_holes
    return mask | small_holes

def get_background(mask, offset_radius, background_radius):
    offset = ndimage.binary_dilation(mask, iterations=offset_radius)
    background = ndimage.binary_dilation(offset, iterations=background_radius)
    return background ^ offset

def get_areas(mask):
    labels, num_regions = ndimage.label(mask)
    region_indices = numpy.arange(1, num_regions + 1)
    areas = ndimage.sum(numpy.ones_like(mask), labels=labels, index=region_indices)
    return labels, region_indices, areas

def get_largest_object(mask):
    labels, region_indices, areas = get_areas(mask)
    largest = region_indices[areas.argmax()]
    return labels == largest

def remove_small_area_objects(mask, max_area):
    labels, region_indices, areas = get_areas(mask)
    keep_labels = areas > max_area
    keep_labels = numpy.concatenate(([0], keep_labels))
    return keep_labels[labels]

def fill_small_area_holes(mask, max_area):
    outside = ndimage.binary_dilation(numpy.zeros_like(mask), mask=~mask, iterations=-1, border_value=1)
    holes = ~(mask | outside)
    labels, region_indices, areas = get_areas(holes)
    keep_labels = areas <= max_area
    keep_labels = numpy.concatenate(([0], keep_labels))
    small_holes = keep_labels[labels]
    return mask | small_holes
