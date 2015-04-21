import numpy
from . import _sample_texture

sample_texture = _sample_texture.sample_texture
sample_ar_texture = _sample_texture.sample_ar_texture

def subsample_mask(mask, max_points):
    """Return a mask containing at most max_points 'True' values, each of which
    is located somewhere within the original mask.

    This is useful for sampling textures where it is neither necessary nor practical
    to sample EVERY pixel of potential interest. Instead, a random subset of the
    pixels of interest is selected.
    """
    mask = numpy.asarray(mask) > 0
    num_points = mask.sum()
    if num_points > max_points:
        z = numpy.zeros(num_points, dtype=bool)
        z[:max_points] = 1
        mask = mask.copy()
        mask[mask] = numpy.random.permutation(z)
    return mask
