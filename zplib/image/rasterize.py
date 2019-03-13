import numpy

def barycentric_coords(vertices, coordinates):
    """Compute barycentric coordinates with respect to a given triangle on a grid.

    Parameters:
        vertices: vertices of a triangle as a list (or array) of (x, y) pairs.
            Must be convertible to a shape (3, 2) array.
        coordinates: set of (x, y) coordinates at which to calculate the barycentric
            coordinates. coordinates.shape[0] must be 2; other dimensions will be
            broadcast. E.g. if coordinates.shape = (2, 100), that corresponds
            to a list of 100 (x, y) pairs and the output will be of shape (3, 100).
            On the other hand, if coordinates.shape = (3, 100, 100), that would
            correspond to a 100x100 grid of (x, y) coordinates and the output
            would be of shape (3, 100, 100). Such a grid might come from the
            output of numpy.indices or numpy.meshgrid.

    Returns: array of shape (3, ...) containing the 3 barycentric coordinates
    of each of the pixels at position (w, h), where output[0] contains
    the coordinates with respect to vertices[0], and so forth.
    """
    vertices = numpy.asarray(vertices)
    coordinates = numpy.asarray(coordinates)
    assert vertices.shape == (3, 2)
    assert coordinates.shape[0] == 2 and coordinates.ndim >= 2
    # set up the problem in the matrix-multiplication form from
    # https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    R = numpy.ones((3, 3))
    R[:2] = vertices.T
    Rinv = numpy.linalg.inv(R)
    # but now instead of computing the full dot product of [x, y, 1] with
    # T for each x and y, we instead just compute the portion of the dot product
    # that varys with [x, y] and then add in the constant values after the fact
    # (which is rather faster)
    barycenters = numpy.einsum('ij,jk...->ik...', Rinv[:, :2], coordinates)
    barycenters += Rinv[:, 2].reshape([3] + [1]*(coordinates.ndim - 1)) # reshape for broadcasting
    return barycenters

def draw_triangle(vertices, size):
    """Return a mask containing ones for pixels in a given triangle.

    Very inefficient for a large image size and a small triangle, but provides
    an example for using barycentric coordinates.

    Parameters:
        vertices: vertices of a triangle as a list (or array) of (x, y) pairs.
            Must be convertible to a shape (3, 2) array.
        size: size of output mask
    """
    barycenters = barycentric_coords(vertices, numpy.indices(size))
    return (barycenters >= 0).all(axis=0)