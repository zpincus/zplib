'''
# zplib

Author: [Zachary Pincus](http://zplab.wustl.edu) <zpincus@gmail.com>

Python modules for common tasks in the Pincus lab.

Curve
-----
Functions for computations over plane curves, approximated as series of points (polylines) or parametric splines.
 - curve.geometry: basic algorithms for polyline curves.
 - curve.interpolate: methods for resampling polylines and fitting them to smoothing splines (using scipy.interpolate.fitpack).

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

Scalar\_stats
-------------
Functions for analysis of 1-dimensional data.
 - scalar\_stats.hmm: estimate paths through a HMM with the Viterbi algorithm
   and estimate HMM parameters from observed data.
 - scalar\_stats.kde: estimate the distribution or survival function of a set of data using scipy.stats.kde
 - scalar\_stats.mcd: calculate the robust mean and standard deviation of data with the univarite MCD estimator of outliers vs. inliers.
 - scalar\_stats.moving\_mean\_std: estimate smooth trenslines for mean and standard deviation from scattered data in x and y.
 - scalar\_stats.smoothing: calculate weighted means of data, or smooth scattered data with Savitzky-Golay, LOWESS, or robust polynomial fitting.

'''