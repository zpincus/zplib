import numpy

def pca(data):
    """Perform PCA on a dataset.

    Arguments:
      data: shape (n, m) array, where n is the number of data points and m is
        the dimensionality of each data point.

    Returns: (mean, pcs, norm_pcs, variances, positions, norm_positions)
      mean: array of shape m, representing the mean of the input data.
      pcs: array of shape (p, m) containing the p principal components. Each
        component is normalized to euclidian distance of 1. The number of
        components p is <= min(n - 1, m). The relationship is exact if no data
        points are duplicate, otherwise fewer PCs may be returned.
      norm_pcs: same as the array pcs, except that each component is scaled by
      the standard deviation along each component.
      variances: length-p array containing the amount of variance explained by
        each component.
      positions: the position of each data point in the basis of the principal
        components; shape = (n, p).
      norm_positions: the position of each data point in the basis of the
        normalized principal components. Positions in this basis are in terms
        of standard deviations away from the mean, rather than at the original
        scale of the data points; shape = (n, p).

    Example:
      mean, pcs, norm_pcs, variances, positions, norm_positions = pca(data)
      synthetic_point = mean + 3*norm_pcs[0]

      Here, 'synthetic_point' is three standard deviations away from the mean
      in one direction along the first principle component.
    """
    data = numpy.asarray(data)
    mean = data.mean(axis=0)
    centered = data - mean
    # could use _pca_svd below, but that appears empirically slower...
    pcs, variances, stds, positions, norm_positions = _pca_eig(centered)
    good_pcs = len(pcs) - numpy.isclose(variances, 0).sum()
    pcs = pcs[:good_pcs]
    variances = variances[:good_pcs]
    stds = stds[:good_pcs]
    positions = positions[:, :good_pcs]
    norm_positions = norm_positions[:, :good_pcs]
    norm_pcs = pcs * stds[:, numpy.newaxis]
    return mean, pcs, norm_pcs, variances, positions, norm_positions


def pca_dimensionality_reduce(data, required_variance_explained):
    """Use PCA to reduce the dimensionality of a dataset by retaining only a
    subset of the principal components.

    Arguments:
      data: shape (n, m) array, where n is the number of data points and m is
        the dimensionality of each data point.
      required_variance_explained: fraction of the total variance which must be
        explained by the retained principal components. Must be in the range
        (0,1]

    Returns: (mean, pcs, norm_pcs, variances, total_variance, positions, norm_positions)
      mean: array of shape m, representing the mean of the input data.
      pcs: array of shape (p, m) containing the p principal components. Each
        component is normalized to euclidian distance of 1. The number of
        components p is <= min(n, m).
      norm_pcs: same as the array pcs, except that each component is scaled by
      the standard deviation along each component.
      variances: length-p array containing the amount of variance explained by
        each component.
      total_variance: the total variance in the original dataset, of which
        variances.sum() is explained by the given principal components.
      positions: the position of each data point in the basis of the principal
        components; shape = (n, p).
      norm_positions: the position of each data point in the basis of the
        normalized principal components. Positions in this basis are in terms
        of standard deviations away from the mean, rather than at the original
        scale of the data points; shape = (n, p).
    """
    mean, pcs, norm_pcs, variances, positions, norm_positions = pca(data)
    total_variance = numpy.add.accumulate(variances / numpy.sum(variances))
    num = numpy.searchsorted(total_variance, required_variance_explained) + 1
    return mean, pcs[:num], norm_pcs[:num], variances[:num], numpy.sum(variances), positions[:,:num], norm_positions[:,:num]

def pca_decompose(data, pcs, mean):
    """Return the positions of new data points in the space spanned by a set of
    principle components.

    Arguments:
      data: shape (n, m) array, where n is the number of data points and m is
        the dimensionality of each data point.
      pcs: shape (p, m) array of p principal components. These may either be
        the unit-length 'pcs' or standard-deviation-scaled 'norm_pcs'.
      mean: the mean of the original data points from which the pcs were
        obtained.

    Returns: shape (n, p) projection of the data into the principal components
      basis.
    """
    projection = numpy.dot(data - mean, pcs.T)
    pc_lens = (pcs**2).sum(axis=1)
    if not numpy.allclose(pc_lens, numpy.ones(len(pcs))):
        # pcs are not of unit length; we must be using the normalized versions and
        # thus desire the normalized positions.
        # The normalized position is numpy.dot(data - mean, non_norm_pcs.T) / standard_deviations
        # and norm_pcs = non_norm_pcs * standard_deviations, so
        # we want numpy.dot(data - mean, norm_pcs.T) / standard_deviations**2
        # Note that the squared length of the norm_pcs is exactly standard_deviations**2
        projection /= pc_lens
    return projection

def pca_reconstruct(positions, pcs, mean):
    """Given positions in some PCA basis, return data points in the original
    data space.

    Arguments:
      positions: shape (n, p) array containing the projection of n data points
      into the basis of p principal components.
      pcs: shape (p, m) array of p principal components. These may either be
        the unit-length 'pcs' or standard-deviation-scaled 'norm_pcs'.
      mean: the mean of the original data points from which the pcs were
        obtained.

    Returns: shape (n, m) array of data points in the original m-dimensional
      space.
    """
    return mean + numpy.dot(positions, pcs)

def _pca_eig(data):
    """Perform PCA on a dataset using a symmetric eigenvalue decomposition."""
    values, vectors = _symm_eig(data)
    pcs = vectors.T
    variances = values / len(data)
    stds = numpy.sqrt(variances)
    positions = numpy.dot(data, vectors)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        norm_positions = positions / stds
    norm_positions[~numpy.isfinite(norm_positions)] = 0
    return pcs, variances, stds, positions, norm_positions

def _symm_eig(a):
    """Return the eigenvectors and eigenvalues of the symmetric matrix a'a. If
    a has more columns than rows, then that matrix will be rank-deficient,
    and the non-zero eigenvalues and eigenvectors can be more easily extracted
    from the matrix aa'. From the properties of the SVD:
        if matrix a of shape (m,n) has SVD u*s*v', then:
            a'a = v*s's*v'
            aa' = u*ss'*u'
        let s_hat, an array of shape (m,n), be such that s * s_hat = I(m,m)
        and s_hat * s = I(n,n). Thus, we can solve for u or v in terms of the other:
            v = a'*u*s_hat'
            u = a*v*s_hat
    """
    m, n = a.shape
    if m >= n:
        # just return the eigenvalues and eigenvectors of a'a
        vals, vecs = _eigh(numpy.dot(a.T, a))
        # if elements are < 0 due to numerical instabilities, set to 0
        vals[vals < 0] = 0
        return vals, vecs
    else:
        # figure out the eigenvalues and vectors based on aa', which is smaller
        sst_diag, u = _eigh(numpy.dot(a, a.T))
        # if elements are < 0 due to numerical instabilities, set to 0
        sst_diag[sst_diag < 0] = 0
        # now get the inverse square root of the diagonal, which will form the
        # main diagonal of s_hat
        with numpy.errstate(divide='ignore', invalid='ignore'):
            s_hat_diag = 1/numpy.sqrt(sst_diag)
        s_hat_diag[~numpy.isfinite(s_hat_diag)] = 0
        # s_hat_diag is a list of length m, a'u is shape (n, m), so we can just use
        # numpy's broadcasting instead of matrix multiplication, and only create
        # the upper m x m block of a'u, since that's all we'll use anyway...
        v = numpy.dot(a.T, u[:,:m]) * s_hat_diag
        return sst_diag, v

def _eigh(m):
    """Return eigenvalues and eigenvectors of hermetian matrix m, sorted in
    by largest eigenvalue first."""
    values, vectors = numpy.linalg.eigh(m)
    order = values.argsort()[::-1]
    return values[order], vectors[:,order]

def _pca_svd(data):
    """Perform PCA on a dataset using the singular value decomposition."""
    u, s, vt = numpy.linalg.svd(data, full_matrices=0)
    pcs = vt
    data_count = len(data)
    variances = s**2 / data_count
    root_data_count = numpy.sqrt(data_count)
    stds = s / root_data_count
    positions = u * s
    norm_positions = u * root_data_count
    return pcs, variances, stds, positions, norm_positions
