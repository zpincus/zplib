import numpy
from matplotlib import cm

def scale(array, min=None, max=None, gamma=1, output_max=255):
    """Return an array with values in the range [0, output_max].

    If 'min' and/or 'max' are specified, these represent the values in the input
    that will be mapped to 0 and 'output_max' in the output, respectively.
    If not specified, the min and/or max values from the input will be used.

    If gamma is specified, a gamma-transform will be applied to the final array.
    """
    if min is None:
        min = array.min()
    if max is None:
        max = array.max()
    if min >= max:
        return numpy.zeros_like(array)
    err = numpy.seterr(under='ignore')
    arr = ((numpy.clip(array.astype(float), min, max) - min) / (max - min))**gamma
    numpy.seterr(**err)
    return arr * output_max

def write_scaled(array, filename, min, max, gamma=1):
    import freeimage
    freeimage.write(scale(array, min, max, gamma).astype(numpy.uint8), filename)

def color_tint(array, target_color):
    """Given an array and an (R,G,B) color-tuple, return a color-tinted
    array, where the intensity ranges from (0,0,0) to (R,G,B), weighted by the
    array values. The output shape will be array.shape + (len(target_color),)

    The input array MUST be scaled [0, 1] with 'scale()' or similar."""
    array = numpy.asarray(array)
    return array[..., numpy.newaxis] * target_color

def color_map(array, spectrum_max=0.925, uint8=True, cmap='plasma'):
    """Color-map the input array on a pleasing black-body-ish black-blue-red-orange-yellow
    spectrum, using matplotlib's excellent and perceptually linear "plasma" or "inferno" colormap.

    Parameters:
        array: MUST be scaled [0, 1] with 'scale()' or similar.
        spectrum_max: controls the point along the spectrum (0 to 1)
            at which the colormap ends. A value of 1 is often too intensely
            yellow for good visualization.
        uint8: if True, return uint RGB tuples in range [0, 255], otherwise
            floats in [0, 1]
        cmap: matplotlib color map to use. Should be 'plasma' or 'inferno'...

    Output: array of shape array.shape + (3,), where color values are RGB tuples
    """
    # array scaled 0 to 1
    array = numpy.asarray(array, dtype=float) * spectrum_max
    assert array.min() >= 0 and array.max() <= 1
    rgb = cm.get_cmap(cmap)(array, bytes=numpy.uint8)[...,:3]
    return rgb

def luminance(color_array):
    """Return luminance of an RGB (or RGBA) array (shape (x, y, 3) or (x, y, 4),
    respectively) using the formula for CIE 1931 linear luminance:
    https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    """
    R, G, B = color_array.transpose((2,0,1))[:3]
    return 0.2126*R + 0.7152*G + 0.0722*B

def screen(a, b, max_possible=255):
    """Blend two arrays together using the 'screen' mode.

    Good for combining color-tinted fluorescence images with brightfield, for example.
    Parameter 'max_possible' refers to the brightest-possible value in the array.
    E.g. 1 for images scaled from 0 to 1, or 255 for images scaled from 0 to 255.
    """
    a = a.astype(float)
    b = b.astype(float)
    return max_possible - (((max_possible - a)*(max_possible - b))/max_possible)

def multi_screen(arrays, max_possible=255):
    """Screen a list of arrays together. See 'screen()' for an explanation of the
    parameters"""
    b = arrays[0]
    for a in arrays[1:]:
        b = screen(a, b, max_possible)
    return b

def alpha_blend(top, bottom, alpha):
    """Blend top image onto bottom image using the provided alpha value(s).

    Parameters:
        top, bottom: images, either of shape (x, y) or (x, y, c).
        alpha: alpha value for blending, either scalar, or (x, y) mask. Must be
            in the range [0, 1]
    """
    alpha = numpy.asarray(alpha)
    assert top.shape == bottom.shape
    if len(top.shape) == 3 and len(alpha.shape) == 2:
        # RBG image with 2d mask
        alpha = alpha[:, :, numpy.newaxis]
    return (top * alpha + bottom * (1-alpha)).astype(bottom.dtype)

def composite(bf, fl_images, fl_colors, bf_color=(255,255,255)):
    """Composite one or more fluorescence images on top of a brightfield image.

    Parameters:
        bf: brightfield image. MUST be scaled in the range [0, 1].
        fl_images: list of fluorescence images. MUST be scaled in the range [0, 1].
        fl_colors: list of RGB tuples for the color-tint of each fluorescence image.
        bf_color: RGB tuple for the color-tint of the brigtfield image. (White is usual.)

    Output: RGB image.
    """
    bf = color_tint(bf, bf_color).astype(numpy.uint8)
    fl_images = [color_tint(fl, cl).astype(numpy.uint8) for fl, cl in zip(fl_images, fl_colors)]
    return multi_screen([bf] + fl_images)

def interpolate_color(array, zero_color, one_color):
    """Make an image with colors lineraly interpolated between two RGB tuples.

    Input array MUST be in the range [0, 1]
    """
    return color_tint(array, zero_color) + color_tint(array, one_color)

def neg_pos_color_tint(array, zero_color=(0,0,0), neg_color=(255,0,76), pos_color=(0,50,255)):
    """Tint a signed array with two different sets of colors, one for negative numbers
    to zero, and one for zero to positive numbers.

    Parameters:
        array: MUST be scaled in the range [-1, 1]
        zero_color: RGB tuple of the color at the zero value
        pos_color: RBG tuple of the color that 1 in the input array should map to
        neg_color: RBG tuple of the color that -1 in the input array should map to
    """
    array = numpy.asarray(array)
    negative_mask = array < 0
    negative = array[negative_mask]
    positive = array[~negative_mask]
    neg_colors = interpolate_color(-negative, zero_color, neg_color)
    pos_colors = interpolate_color(positive, zero_color, pos_color)
    output = numpy.empty(array.shape, dtype=numpy.uint8)
    output[negative_mask] = neg_colors
    output[~negative_mask] = pos_colors
    return output


def wavelength_to_rgb(l):
    """Given a wavelength in nanometers, regurn an RGB tuple using
    the so-called "Bruton's Algorithm".
    http://www.physics.sfasu.edu/astro/color/spectra.html

    Note: wavelength parameter must be in the range [350, 780]
    """
    assert (350 <= l <= 780)
    if l < 440:
        R = (440-l)/(440-350)
        G = 0
        B = 1
    elif l < 490:
        R = 0
        G = (l-440)/(490-440)
        B = 1
    elif l < 510:
        R = 0
        G = 1
        B = (510-l)/(510-490)
    elif l < 580:
        R = (l-510)/(580-510)
        G = 1
        B = 0
    elif l < 645:
        R = 1
        G = (645-l)/(645-580)
        B = 0
    else:
        R = 1
        G = 0
        B = 0
    if l > 700:
        intensity = 0.3 + 0.7 * (780-l)/(780-700)
    elif l < 420:
        intensity = 0.3 + 0.7 * (l-350)/(420-350)
    else:
        intensity = 1
    return (255 * intensity * numpy.array([R,G,B])).astype(numpy.uint8)

