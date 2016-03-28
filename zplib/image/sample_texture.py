import numpy
from sklearn import cluster

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


def bin_by_texture_class(image, num_classes, mask=None, size=3):
    """Return an image where pixels are replaced by the "texture class" that
    that pixel belongs to.

    Textures are sampled using sample_ar_texture and then clustered with k-means
    clustering. An image is returned where each pixel represents the label of
    its texture cluster.

    Parameters:
        image: 2-dimensional numpy array of type uint8, uint16, or float32
        num_classes: number of clusters to identify with k-means
        mask: optional mask for which pixels to examine
        size: size of the ar feature window (see sample_ar_texture)
    """
    texture_samples = sample_ar_texture(image, mask, size)
    kmeans = cluster.MiniBatchKMeans(n_clusters=64, max_iter=300)
    kmeans.fit(texture_samples)
    dtype = numpy.uint16 if num_classes > 256 else numpy.uint8
    labeled_image = numpy.zeros(image.shape, dtype)
    # if not image.flags.fortran:
    #     labeled_image = labeled_image.T
    #     if mask is not None:
    #         mask = mask.T
    if mask is not None:
        labeled_image[mask] = kmeans.labels_
    else:
        labeled_image.flat = kmeans.labels_
    # if not image.flags.fortran:
    #     labeled_image = labeled_image.T
    return labeled_image