'''
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