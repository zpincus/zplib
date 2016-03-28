import cython
cimport numpy
import numpy
import random

ctypedef fused ARR:
    numpy.float32_t
    numpy.uint16_t
    numpy.uint8_t

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_texture(ARR[:,:] image, mask=None, int size=1):
    """Return texture samples of a given size from an image.

    A region around every specified image pixel will be sampled and returned
    as an array of shape (num_pixels, sample_size) for texture analysis.

    The image boundary condition is treated as a clamp: pixels off the edge will
    have the same value as the nearest pixel at the edge.

    Parameters:
        image: 2-dimensional array of type float32, uint16, or uint8
        mask: if None, every pixel will be sampled. If not none, must be a
            booelan array that specifies pixels to sample. Note that the sampling
            region may extend outside of the mask -- only the central pixel must
            belong to the mask.
        size: the 'radius' of the sampling region: any pixel within 'size'
            pixels of the center pixel, along any axis, will be sampled. (This
            corresponds to a square or 8-connected image region.)

    Returns: array of shape (num_pixels, sample_size), where num_pixels is either
    the number of nonzero pixels in the mask, or the total number of image pixels
    if a mask is not specified, and sample_size = (2 * size + 1)**2.
    """
    cdef:
        ARR[:,:] image_cy = image
        numpy.uint8_t[:,:] mask_cy
        ARR[:,:] textures_cy
        unsigned int stop_x, stop_y, skip_mask
        unsigned int i, j, ti, tj
        short ii, jj
        int io, jo
    image_py = numpy.asarray(image)
    stop_x, stop_y = image_py.shape
    if mask is not None:
        mask_cy = mask_py = (numpy.asarray(mask) > 0).astype(numpy.uint8)
        assert mask_py.shape == image_py.shape
        skip_mask = 0
        tex_size = mask_py.sum()
    else:
        mask_cy = numpy.array([[0]], dtype=numpy.uint8) # suppress unbound local checking below. Don't worry, we won't ever try to index into this
        skip_mask = 1
        tex_size = image_py.size
    textures_cy = textures_py = numpy.empty((tex_size, (2*size+1)**2), dtype=image_py.dtype)
    ti = 0
    for i in range(stop_x):
        for j in range(stop_y):
            if skip_mask or mask_cy[i, j]:
                tj = 0
                for ii in range(-size,size+1):
                    for jj in range(-size,size+1):
                        io = i+ii
                        jo = j+jj
                        if io < 0: io = 0
                        if io >= stop_x: io = stop_x-1
                        if jo < 0: jo = 0
                        if jo >= stop_y: jo = stop_y-1
                        textures_cy[ti, tj] = image_cy[io, jo]
                        tj += 1
                ti += 1
    return textures_py

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_ar_texture(ARR[:,:] image, mask=None, int size=4):
    """Return autoregressive texture samples of a given size from an image.

    As per 'Kwang In Kim et al. Support vector machines for texture
    classification. PAMI (2002)', dense texture features contain many redundencies
    due to local autocorrelation in image intensities. This function samples
    image intensity values sparsely within a given window, which allows for a
    larger window size for a given number of sampled points. Often this improves
    texture classification.

    A region around every specified image pixel will be sampled and returned
    as an array of shape (num_pixels, sample_size) for texture analysis.

    The image boundary condition is treated as a clamp: pixels off the edge will
    have the same value as the nearest pixel at the edge.

    Parameters:
        image: 2-dimensional array of type float32, uint16, or uint8
        mask: if None, every pixel will be sampled. If not none, must be a
            booelan array that specifies pixels to sample. Note that the sampling
            region may extend outside of the mask -- only the central pixel must
            belong to the mask.
        size: the 'radius' of the sampling region: pixels within 'size'
            pixels of the center pixel, along any axis, are candidates for
            sampling. The precise pattern of sampling within is those pixels on
            the central axes or along the diagonals of the window.

    Returns: array of shape (num_pixels, sample_size), where num_pixels is either
    the number of nonzero pixels in the mask, or the total number of image pixels
    if a mask is not specified, and sample_size = 8*size+1.
    """
    cdef:
        ARR[:,:] image_cy = image
        numpy.uint8_t[:,:] mask_cy
        ARR[:,:] textures_cy
        unsigned int stop_x, stop_y, skip_mask
        unsigned int i, j, ti, tj
        short ii, jj
        int io, jo
    image_py = numpy.asarray(image)
    stop_x, stop_y = image_py.shape
    if mask is not None:
        mask_cy = mask_py = (numpy.asarray(mask) > 0).astype(numpy.uint8)
        assert mask_py.shape == image_py.shape
        skip_mask = 0
        tex_size = mask_py.sum()
    else:
        mask_cy = numpy.array([[0]], dtype=numpy.uint8) # suppress unbound local checking below. Don't worry, we won't ever try to index into this
        skip_mask = 1
        tex_size = image_py.size
    textures_cy = textures_py = numpy.empty((tex_size, 8*size+1), dtype=image_py.dtype)
    ti = 0
    for i in range(stop_x):
        for j in range(stop_y):
            if skip_mask or mask[i, j]:
                tj = 0
                for ii in range(-size,size+1):
                    for jj in range(-size,size+1):
                        if ii == jj or ii == -jj or ii == 0 or jj == 0 :
                            io = i+ii
                            jo = j+jj
                            if io < 0: io = 0
                            if io >= stop_x: io = stop_x-1
                            if jo < 0: jo = 0
                            if jo >= stop_y: jo = stop_y-1
                            textures_cy[ti, tj] = image_cy[io, jo]
                            tj += 1
                ti += 1
    return textures_py