import numpy

def make_neighborhood_view(image, pad_mode='edge', **pad_kws):
    padding = [(1, 1), (1, 1)] + [(0,0) for _ in range(image.ndim - 2)]
    padded = numpy.pad(image, padding, mode=pad_mode, **pad_kws)
    shape = image.shape[:2] + (3,3) + image.shape[2:]
    strides = padded.strides[:2]*2 + padded.strides[2:]
    return numpy.ndarray(shape, padded.dtype, buffer=padded, strides=strides)
