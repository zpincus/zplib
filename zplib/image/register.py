import numpy
import scipy.ndimage as ndimage
import scipy.optimize as optimize

from . import pyramid

## Image comparison metrics
def abs_diff(i1, i2):
    '''Return the absolute differences between two images.'''
    return numpy.abs(i1 - i2)

def square_diff(i1, i2):
    '''Return the squared differences between two images.'''
    return (i1 - i2)**2

def pyr_register(fixed_image, moving_image, levels=3, initial_shift=(0,0), diff_function=abs_diff, iterate=True):
    '''Register two images via downsampling, computing a shift, and refining on the original images.

    At each downsampling ("pyramid level"), the images are shrunk two-fold. The maximum shift
    that can be identified at each level is ~5 px, so for a 3 level pyramid (original size
    images and two consecutive downsamplings, for a 4-fold maximum shrinking), a 20 pixel
    shift in the original images can be detected.

    Parameters:
        fixed_image: image for comparison
        moving_image: image that will be shifted to match the fixed_image
        levels: number of levels of the pyramid (counting the original size)
        initial_shift: (x, y) shift to begin search at.
        diff_function: image-comparison metric: one of abs_diff or square_diff
        iterate: if true, use iterate_register at each pyramid level (slower, but more accurate).
            If False, use register.
    Returns: (x, y) shift.

    To apply the identified shift to the moving image, call e.g.:
        shifted = ndimage.shift(moving_image, shift, order=1)
    '''
    pf, pm = [fixed_image], [moving_image]
    for i in range(levels-1):
        pf.append(pyramid.pyr_down(pf[-1]))
        pm.append(pyramid.pyr_down(pm[-1]))
    initial_shift = numpy.asarray(initial_shift) / 2**(steps - 1)
    for f, m in reversed(list(zip(pf, pm))):
        if iterate:
            func = iterate_register
        else:
            func = register
        shift = func(f, m, initial_shift=initial_shift, diff_function=diff_function)
        initial_shift = shift * 2
    return shift

def iterate_register(fixed_image, moving_image, initial_shift=(0,0), search_bounds=5, diff_function=abs_diff, max_iters=10, tol=0.25, eps=1):
    '''Register two images iteratively.

    The 'register()' function is repeatedly applied to find the best registration
    between two images. Iteration ceases after max_iters, or if the change in shift
    is less than tol pixels.

    Parameters:
        fixed_image: image for comparison
        moving_image: image that will be shifted to match the fixed_image
        max_iters: maximum number of times register() will be called.
        initial_shift: (x, y) shift to begin search at.
        search_bounds: largest distance from initial_shift to examine.
        diff_function: image-comparison metric: one of abs_diff or square_diff
        tol: if the shift changes by less than this amount in x and y, then
            iteration will be ceased.
        eps: the comparison metric's gradients are estimated by finite differences
            with eps-sized offsets. For most images, values around 1 work well.
    Returns: (x, y) shift.

    To apply the identified shift to the moving image, call e.g.:
        shifted = ndimage.shift(moving_image, shift, order=1)
    '''
    cache = {}
    for i in range(max_iters):
        shift = register_images(fixed_image, moving_image, initial_shift, search_bounds, diff_function, tol=0.5, eps=eps, cache=cache)
        shift = numpy.round(shift, 2)
        if numpy.abs(shift - initial_shift).max() < tol:
            break
        initial_shift = shift
    return shift

def register(fixed_image, moving_image, initial_shift=(0,0), search_bounds=5, diff_function=abs_diff, tol=0.5, eps=1, cache=None):
    '''Register two images by numerical optimization.

    Parameters:
        fixed_image: image for comparison
        moving_image: image that will be shifted to match the fixed_image
        initial_shift: (x, y) shift to begin search at.
        search_bounds: largest distance from initial_shift to examine.
        diff_function: image-comparison metric: one of abs_diff or square_diff
        tol: if the shift changes by less than this amount in x and y, then
            iteration will be ceased.
        eps: the comparison metric's gradients are estimated by finite differences
            with eps-sized offsets. For most images, values around 1 work well.
        cache: used to cache results for given shift values between runs
            (useful if this function will be called repeatedly).
    Returns: (x, y) shift.

    To apply the identified shift to the moving image, call e.g.:
        shifted = ndimage.shift(moving_image, shift, order=1)
    '''
    if cache is None:
        cache = {}
    args = fixed_image, moving_image, diff_function, cache
    bounds = numpy.array([-search_bounds, search_bounds]) + initial_shift
    bounds = [bounds, bounds] # one for x and one for y
    result = optimize.minimize(compare_images, initial_shift, args=args, method='TNC', options={'xtol':xtol, 'eps':eps}, bounds=bounds)
    return result.x

def compare_images(shift, fixed_image, moving_image, diff_function, cache=None):
    '''Return distance metric between two images after applying a shift.

    Parameters:
        shift: (x, y) amount to shift the moving image by
        fixed_image: image for comparison
        moving_image: image that will be shifted to match the fixed_image
        diff_function: image-comparison metric: one of abs_diff or square_diff
        cache: used to cache results for given shift values between runs
            (useful if this function will be called repeatedly).
    Returns: sum of the diff function, over all pixels except those in regions
    at image edges that were trimmed off by the shift.
    '''
    shift = numpy.round(shift, 2)
    hash_shift = tuple(shift)
    if cache is not None and hash_shift in cache:
        return cache[hash_shift]
    moving_image = ndimage.shift(moving_image, shift, output=numpy.float32, cval=numpy.nan, order=1)
    diff = diff_function(fixed_image, moving_image)
    result = diff[numpy.isfinite(diff)].sum()
    if cache is not None:
        cache[hash_shift] = result
    return result
