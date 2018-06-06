'''
Image
-----
Functions for basic image processing.
 - image.colorize: color-map and color-tint grayscale images to RGB color, and combine color images. Also convert wavelengths to colors.
 - image.fft: construct simple Butterworth filters for filtering image frequencies by FFT.
 - image.fast\_fft: use pyfftw to speed up FFT image filtering.
 - image.mask: basic mask-processing functions to demonstrate the uses of scipy.ndimage's binary morphology operators.
 - image.polyfit: fit image intensities to a low-order polynomial as a way of estimating the image background.
 - image.resample: transform an image into the frame of reference of an aribrtrary rectangle or spline (defined using curve.interpolate).
 - image.sample\_texture: sample image patches for use in texture-based classification
'''