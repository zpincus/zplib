import sys
import numpy
import pkg_resources
from matplotlib import cm

def scale(array, min=None, max=None, gamma=1, output_max=255):
    """Return an array with values in the range [0, output_max].

    If 'min' and/or 'max' are specified, these represent the values in the input
    that will be mapped to 0 and 'output_max' in the output, respectively.
    If not specified, the min and/or max values from the input will be used.

    If gamma is specified, a gamma-transform will be applied to the final array.
    """
    array = numpy.asarray(array, dtype=numpy.float32)
    if min is None:
        min = array.min()
    if max is None:
        max = array.max()
    if min >= max:
        return numpy.zeros_like(array)
    with numpy.errstate(under='ignore'):
        array.clip(min, max, out=array)
        array -= min
        array /= max - min
        array **= gamma
    return array * output_max

def write_scaled(array, filename, min, max, gamma=1):
    """Write an image to disk as a uint8, after scaling with the specified
    parameters (see scale() function)."""
    import freeimage
    freeimage.write(scale(array, min, max, gamma).astype(numpy.uint8), filename)

def blend(top, bottom, top_alpha=None, bottom_alpha=None, mode='normal', input_max=1):
    """Blend two RGB[A] arrays, using normal or screen-blending mode.

    Parameters:
        top, bottom: arrays of shape (x, y, 3) or (x, y, 4) for RGB or RGBA images.
        top_alpha, bottom_alpha: if not None, ignore any alpha channel in image
            and use this value instead (slightly faster if all alpha is equal)
        mode: 'normal', 'screen', 'multiply', or 'overlay'
        input_max: maximum value for the input images and top_opacity value
            (e.g. 1, 255, 65535).

    Returns: image, alpha
        image: array of shape (x, y, 3) with dtype matching the bottom paramter
        alpha: scalar (if RGB arrays or top_alpha and bottom_alpha are specified)
            or array of shape (x, y, 1). To make a RGBA image from the latter,
            just do: numpy.concatenate([image, alpha], axis=-1)
    """
    bottom = numpy.asarray(bottom)
    dtype = bottom.dtype
    assert mode in ('normal', 'screen', 'multiply', 'overlay')
    top, top_alpha = _prepare_and_premultiply(top, top_alpha, input_max)
    bottom, bottom_alpha = _prepare_and_premultiply(bottom, bottom_alpha, input_max)
    out_alpha = top_alpha + bottom_alpha - top_alpha * bottom_alpha
    if mode == 'normal':
        out = top + bottom * (1 - top_alpha)
    elif mode == 'screen':
        out = top + bottom - top * bottom
    elif mode == 'multiply':
        out = top * bottom + top * (1 - bottom_alpha) + bottom * (1 - top_alpha)
        out.clip(0, 1, out=out)
    elif mode == 'overlay':
        mult_mask = (2 * bottom <= bottom_alpha)[:, :, 0]
        out = numpy.empty_like(bottom)
        t = top[mult_mask]
        if top_alpha.ndim > 0:
            ta = top_alpha[mult_mask]
        else:
            ta = top_alpha
        b = bottom[mult_mask]
        if bottom_alpha.ndim > 0:
            ba = bottom_alpha[mult_mask]
        else:
            ba = bottom_alpha
        out[mult_mask] = 2 * t * b + t * (1 - ba) + b * (1 - ta)

        mult_mask = ~mult_mask
        t = top[mult_mask]
        if top_alpha.ndim > 0:
            ta = top_alpha[mult_mask]
        b = bottom[mult_mask]
        if bottom_alpha.ndim > 0:
            ba = bottom_alpha[mult_mask]
        out[mult_mask] = t * (1 + ba) + b * (1 + ta) - 2 * t * b - ta * ba
        out.clip(0, 1, out=out)
    def on_invalid(err, flag):
        out[numpy.isnan(out)] = 0
    numpy.seterrcall(on_invalid)
    with numpy.errstate(invalid='call'):
        out /= out_alpha
    return (out * input_max).astype(dtype), (out_alpha * input_max).astype(dtype)

def _prepare_and_premultiply(array, alpha, input_max):
    assert array.ndim == 3 and array.shape[2] in (3, 4)
    if alpha is not None:
        assert 0 <= alpha <= input_max
        alpha /= input_max
    array = numpy.asarray(array, dtype=numpy.float32) / input_max
    if alpha is not None:
        array = array[:, :, :3] * alpha
    elif array.shape[2] == 3:
        alpha = 1
    else:
        alpha = array[:, :, 3:] # shape (x, y, 1)
        array = array[:, :, :3] * alpha # shape (x, y, 3)
    return array, numpy.asarray(alpha, dtype=numpy.float32)

def multi_blend(arrays, colors, alphas=None, modes=None, input_max=1, color_max=1):
    """Composite one or more one-channel images atop one another, colorizing each.

    The arrays will be blended in reverse order: arrays[0] will be the base and
    arrays[-1] will be the top image.

    Parameters:
        arrays: list of input images of shape (x, y).
        colors: list of RGB or RGBA tuples, one for each input array.
        alphas: opacity value for each image (if not None, overrides the alpha
            channel for each image, if present.
        modes: None, or list of 'normal', 'screen', 'multiply', or 'overlay'. If
            None, use 'normal' blend mode. Blend mode ignored for bottom-most
            image.
        input_max: maximum possible value of the input images (e.g. 1, 255, 65535)
        color_max: maximum value of of the RGB colors (e.g. 1 or 255 usually)

    Returns: image, alpha
        image: array of shape (x, y, 3) with dtype matching arrays[0]
        alpha: scalar (if RGB arrays or top_alpha and bottom_alpha are specified)
            or array of shape (x, y, 1). To make a RGBA image from the latter,
            just do: numpy.concatenate([image, alpha], axis=-1)
    """
    if alphas is None:
        alphas = [None] * len(arrays)
    if modes is None:
        modes = ['normal'] * len(arrays)
    colorized = [color_tint(array, color, input_max) for array, color in zip(arrays, colors)]
    bottom = colorized[0]
    bottom_alpha = alphas[0]
    for top, top_alpha, mode in zip(colorized[1:], alphas[1:], modes[1:]):
            bottom, bottom_alpha = blend(top, bottom, top_alpha, bottom_alpha, mode, color_max)
    return bottom, bottom_alpha

def color_tint(array, target_color, input_max=1):
    """Given a one-channel image and an RGB[A] tuple, return a color-tinted image.

    The output image values will range from (0,0,0) to (R,G,B), weighted by the
    array values (divided by the input_max). If an alpha value is provided, the
    output alpha channel will be that value (not weighted by the input image)

    Parameters:
        array: input image of shape (x, y).
        target_color: (R, G, B) tuple. If a (R, G, B, A) tuple, an alpha channel
            consisting of the A value will be added to the image.
        input_max: maximum possible value of the input image (e.g. 1, 255, 65535)

    Output: image with shape = array.shape + (len(target_color),)

    """
    channels = len(target_color)
    assert channels in (3, 4)
    assert array.ndim == 2
    array = numpy.asarray(array, dtype=numpy.float32) / input_max
    out = array[:, :, numpy.newaxis] * numpy.asarray(target_color, dtype=numpy.float32)
    # nb: seems wasteful to make the alpha channel above (weighted by the array values)
    # and then overwrite below, but this appears to be the fastest way in numpy...
    if channels == 4:
        out[:, :, 3].fill(target_color[3])
    return out

def color_map(array, spectrum_max=0.925, uint8=True, cmap='plasma', input_max=1):
    """Color-map the input array on a pleasing black-body-ish black-blue-red-orange-yellow
    spectrum, using matplotlib's excellent and perceptually linear "plasma" or "inferno" colormap.

    Parameters:
        array: input image of shape (x, y).
        spectrum_max: controls the point along the spectrum (0 to 1)
            at which the colormap ends. A value of 1 is often too intensely
            yellow for good visualization.
        uint8: if True, return uint RGB tuples in range [0, 255], otherwise
            floats in [0, 1]
        cmap: matplotlib color map to use. Should be 'plasma' or 'inferno'.
        input_max: maximum possible value of the input image (e.g. 1, 255, 65535)

    Output: array of shape array.shape + (3,), where color values are RGB tuples
    """
    array = numpy.asarray(array, dtype=numpy.float32) / input_max * spectrum_max
    assert array.min() >= 0 and array.max() <= 1
    return cm.get_cmap(cmap)(array, bytes=uint8)[...,:3]

def _gen_label_colors(seed=0):
    colors = [[0, 0, 0]]
    rs = numpy.random.RandomState(seed=seed)
    while len(colors) < 2**16:
        rgb = rs.randint(256, size=3)
        r, g, b = rgb
        luma = 0.2126*r + 0.7152*g + 0.0722*b # use CIE 1931 linear luminance
        if luma < 75:
            continue
        mr = (rgb[0] + colors[-1][0])/2
        dr, dg, db = (rgb - colors[-1])**2
        dc = 2*dr + 4*dg + 3*db + mr*(dr - db)/256
        if dc > 10000:
            colors.append(rgb)
    colors = numpy.array(colors, dtype=numpy.uint8)
    import pathlib
    numpy.save(str(pathlib.Path(__file__).parent/'_label_colors.npy'), colors)

_label_colors = None
def colorize_label_image(array, cmap=None):
    """Color-map an image consisting of labeled regions (each with different
    integer-valued label).

    Parameters:
        array: integer-valued array
        cmap: if None, use a default mapping of integers to colors; otherwise
            use the named matplotlib colormap. A qualitative colormap like 'tab20'
            is a good idea.

    Example of making a label image from a set of masks:
    label_image = numpy.zeros_like(masks[0])
    for i, mask in enumerate(masks):
        label_image[mask] = i + 1
    colorized = colorize_label_image(label_image)
    """
    if cmap is None:
        global _label_colors
        if _label_colors is None:
            _label_colors = numpy.load(pkg_resources.resource_stream(__name__, '_label_colors.npy'))
        colorized = _label_colors[array]
    else:
        colormap = cm.get_cmap(cmap)
        colorized = colormap(array%colormap.N, bytes=uint8)[...,:3]
        colorized[image == 0] = 0
    return colorized

def luminance(color_array):
    """Return luminance of an RGB (or RGBA) array (shape (x, y, 3) or (x, y, 4),
    respectively) using the formula for CIE 1931 linear luminance:
    https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    """
    R, G, B = color_array.transpose((2,0,1))[:3]
    return 0.2126*R + 0.7152*G + 0.0722*B

def interpolate_color(array, zero_color, max_color, input_max=1):
    """Make an image with colors lineraly interpolated between two RGB tuples.

    Parameters:
        array: input image of shape (x, y).
        zero_color: color of output image where input array == 0
        max_color: color of output image where input array == input_max
        input_max: maximum possible value of the input image (e.g. 1, 255, 65535)

    """
    array = numpy.asarray(array, dtype=numpy.float32) / input_max
    return (color_tint(array, zero_color) + color_tint(array, one_color)) * input_max

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


### DEPRECATED FUNCTIONS ###
import warnings

def screen(a, b, input_max=255):
    """Blend two arrays together using the 'screen' mode.

    Good for combining color-tinted fluorescence images with brightfield, for example.
    Parameter 'max_possible' refers to the brightest-possible value in the array.
    E.g. 1 for images scaled from 0 to 1, or 255 for images scaled from 0 to 255.
    """
    warnings.warn('screen function is deprecated: use blend', FutureWarning)
    a = a.astype(numpy.float32)
    b = b.astype(numpy.float32)
    return input_max - (((input_max - a)*(input_max - b))/input_max)

def multi_screen(arrays, max_possible=255):
    """Screen a list of arrays together. See 'screen()' for an explanation of the
    parameters"""
    warnings.warn('multi_screen function is deprecated: use multi_blend', FutureWarning)
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
    warnings.warn('alpha_blend function is deprecated: use blend', FutureWarning)
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
    warnings.warn('composite function is deprecated: use multi_blend', FutureWarning)
    bf = color_tint(bf, bf_color).astype(numpy.uint8)
    fl_images = [color_tint(fl, cl).astype(numpy.uint8) for fl, cl in zip(fl_images, fl_colors)]
    return multi_screen([bf] + fl_images)
