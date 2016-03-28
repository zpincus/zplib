import numpy
from scipy import ndimage

from . import neighborhood

def canny(image, sigma, low_threshold, high_threshold):
    smoothed, magnitude, sobel = prepare_canny(image, sigma)
    local_maxima = canny_local_maxima(magnitude, sobel)
    return canny_hysteresis(local_maxima, magnitude, low_threshold, high_threshold)

def prepare_canny(image, sigma):
    if sigma > 0:
        smoothed = ndimage.gaussian_filter(image.astype(numpy.float32), sigma)
    else:
        smoothed = image.astype(numpy.float32)
    xsobel = ndimage.sobel(smoothed, axis=0)
    ysobel = ndimage.sobel(smoothed, axis=1)
    magnitude = numpy.hypot(xsobel, ysobel)
    return smoothed, magnitude, (xsobel, ysobel)

def canny_hysteresis(local_maxima, magnitude, low_threshold, high_threshold):
    # Hysteresis threshold
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)
    s = numpy.ones((3,3), dtype=bool)
    return ndimage.binary_propagation(high_mask, mask=low_mask, structure=s)

def canny_local_maxima(magnitude, sobel):
    # code taken and cleaned up from skimage
    xsobel, ysobel = sobel
    abs_xsobel = numpy.abs(xsobel)
    abs_ysobel = numpy.abs(ysobel)
    has_grad = magnitude > 0
    xpos = xsobel >= 0
    xneg = xsobel <= 0
    ypos = ysobel >= 0
    yneg = ysobel <= 0
    # Below directions refer to orientation of *edge normal*, not edge.
    diagonal = ((xpos & ypos) | (xneg & yneg)) & has_grad
    anti_diagonal = ((xpos & yneg) | (xneg & ypos)) & has_grad
    horizontal = (abs_xsobel >= abs_ysobel) & has_grad
    vertical = (abs_xsobel <= abs_ysobel) & has_grad

    magnitude_neighborhood = neighborhood.make_neighborhood_view(magnitude)

    #--------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = numpy.zeros(magnitude.shape, bool)

    # Normal is 0-45 degrees: interpolate magnitude between right and up-and-to-the-right
    # (and left/down-and-left)
    pts = diagonal & horizontal
    _interp_maxima(pts, (1, 0), (1, 1), magnitude_neighborhood, magnitude, local_maxima,
        abs_xsobel, abs_ysobel)
    # Normal is 45-90 degrees: interpolate magnitude between up and up-and-to-the-right
    # (and down/down-and-left)
    pts = diagonal & vertical
    _interp_maxima(pts, (0, 1), (1, 1), magnitude_neighborhood, magnitude, local_maxima,
        abs_xsobel, abs_ysobel)

    # Normal is 90-135 degrees: interpolate magnitude between up and up-and-to-the-left
    # (and down/down-and-right)
    pts = anti_diagonal & vertical
    _interp_maxima(pts, (0, 1), (-1, 1), magnitude_neighborhood, magnitude, local_maxima,
        abs_xsobel, abs_ysobel)

    # Normal is 135-180 degrees: interpolate magnitude between left and up-and-to-the-left
    # (and right/down-and-right)
    pts = anti_diagonal & horizontal
    _interp_maxima(pts, (-1, 0), (-1, 1), magnitude_neighborhood, magnitude, local_maxima,
        abs_xsobel, abs_ysobel)

    return local_maxima

def _interp_maxima(pts, offset1, offset2, magnitude_neighborhood, magnitude, local_maxima,
    abs_xsobel, abs_ysobel):
    """Assuming that the gradient normal is between offset1 (a cardinal direction)
    and offset2 (a diagonal direction), interpolate the gradient magnitude in
    that direction and compare it to the center-pixel gradient magnitude.
    If the center pixel has a larger magnitude than the neighboring values
    along the normal (in both positive and negative directions), then it is
    a local maxima."""
    ox1, oy1 = offset1
    ox2, oy2 = offset2
    assert ox1 == ox2 or oy1 == oy2
    assert ox1 == 0 or oy1 == 0
    # Note: index (1, 1) is the center of the neighborhood.
    c1 = magnitude_neighborhood[pts, 1+ox1, 1+oy1]
    c2 = magnitude_neighborhood[pts, 1+ox2, 1+oy2]
    m = magnitude[pts]
    if ox1 == ox2: # which means that oy1 == 0 and oy2 is nonzero, thus:
        # We're comparing directions that differ in the y direction, so the
        # weighting for c2 is the y portion of the gradient magnitude.
        # NB: no worries of div/0 because if we get to this point, we know that
        # there must be some gradient in the x dir.
        w2 = abs_ysobel[pts] / abs_xsobel[pts]
    else:
        # We're comparing directions that differ in the x direction, so the
        # weight for c2 is the x portion of the gradient magnitude.
        w2 = abs_xsobel[pts] / abs_ysobel[pts]
    w1 = 1 - w2
    c_plus = c2 * w2 + c1 * w1 <= m
    c1 = magnitude_neighborhood[pts, 1-ox1, 1-oy1]
    c2 = magnitude_neighborhood[pts, 1-ox2, 1-oy2]
    c_minus = c2 * w2 + c1 * w1 <= m
    local_maxima[pts] = c_plus & c_minus
