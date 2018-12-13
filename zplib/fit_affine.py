import numpy

def fit_affine(points, reference, weights=None, allow_reflection=False, find_scale=True, find_translation=True):
    """Find the rigid transformation that optimally aligns a given set of points
    to corresponding reference points in a least-squares sense.

    Parameters:
        points, reference: arrays of shape (n, d), containing n points in d-
            dimensional points.
        weights: if not None, the least-squares fit is weighted by these values.
        allow_reflection: if True, then the 'transformation' matrix may be a
            rotation or rotation-and-reflection, depending in which results in a
            better fit.
        find_scale: if True, then the 'scale' value may be other than unity if
            scaling the points results in a better fit.
        find_translation: if True, then the 'translation' vector may be other
            than a zero-vector, if translating the points results in a better fit.

    Returns: (rotation, scale, translation, new_points)
        rotation: rotation matrix; shape (d, d)
        scale: scale facto; scalar
        translation: translation vector; shape (d,)
        new_points: transformed input points; shape (n, d)

    To transform additional points:
        points_out = c * numpy.dot(points_in, rotation) + translation
    """
    # notational conventions after "Generalized Procrustes Analysis and its Applications in Photogrammetry"
    # by Devrim Akca
    A = numpy.matrix(points)
    B = numpy.matrix(reference)
    assert points.shape == reference.shape
    p = A.shape[0]
    k = A.shape[1]
    j = numpy.matrix(numpy.ones((p, 1)))
    if weights is not None:
        Q = numpy.matrix(numpy.sqrt(weights)).T
        # here use numpy-array element-wise multiplication with broadcasting:
        A = numpy.multiply(A, Q)
        B = numpy.multiply(B, Q)
        j = numpy.multiply(j, Q)
    jjt = j * j.T
    jtj = j.T * j
    I = numpy.matrix(numpy.eye(p))
    At_prod = A.T * (I - jjt / jtj)
    S = At_prod * B
    V, D, Wt = numpy.linalg.svd(S)
    if not allow_reflection:
        if numpy.allclose(numpy.linalg.det(V), -1):
            V[:, -1] *= -1
        if numpy.allclose(numpy.linalg.det(Wt), -1):
            Wt[-1, :] *= -1
    T = numpy.dot(V, Wt)
    if find_scale:
        c = numpy.trace(T.T * S) / numpy.trace(At_prod * A)
    else:
        c = 1
    new_A = c * A * T
    if find_translation:
        t = (B - new_A).T * (j / jtj)
        # now unpack t from a 2d matrix-vector into a normal numpy 1d array-vector
        t = numpy.asarray(t)[:, 0]
    else:
        t = numpy.zeros(k)
    if weights is not None:
        new_A = numpy.divide(new_A, Q)
    new_A += t
    return numpy.asarray(T), c, t, numpy.asarray(new_A)