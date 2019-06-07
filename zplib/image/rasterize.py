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

def draw_triangle(vertices, shape):
    """Return a mask containing ones for pixels in a given triangle.

    Very inefficient for a large image size and a small triangle, but provides
    an example for using barycentric coordinates.

    Parameters:
        vertices: vertices of a triangle as a list (or array) of (x, y) pairs.
            Must be convertible to a shape (3, 2) array.
        shape: shape of output mask.
    """
    barycenters = barycentric_coords(vertices, numpy.indices(shape))
    return (barycenters >= 0).all(axis=0)

def gouraud_triangles(triangle_strip, vertex_vals, shape):
    """Return a triangle strip Gouraud-shaded based on values at each vertex.

    Parameters:
        triangle_strip: shape (n, 2) array of vertices describing a strip of
            connected triangles (such that vertices (0,1,2) describe the first
            triangle, vertices (1,2,3) describe the second, and so forth).
        vertex_vals: shape (n,) or (n, m) array of values associated with each
            vertex. In case of shape (n, m) this indicates m distinct sets of
            values for each vertex; as such m distinct output images will be
            produced.
        shape: shape of the output image(s).

    Returns: (mask, output)
        mask: boolean image of requested shape containing the triangle pixels.
        output: single image (if vertex_vals is 1-dim) or list of images (if
            vertex_vals is > 1-dim), where each image contains the interpolation
            of the values at each vertex.
    """
    triangle_strip = numpy.asarray(triangle_strip)
    vertex_vals = numpy.asarray(vertex_vals)
    assert triangle_strip.ndim == 2 and triangle_strip.shape[1] == 2 and len(triangle_strip) > 2
    unpack_out = False
    if vertex_vals.ndim == 1:
        vertex_vals = vertex_vals[:, numpy.newaxis]
        unpack_out = True
    assert len(vertex_vals) == len(triangle_strip)
    grid = numpy.indices(shape) + 0.5 # pixel centers are at (0.5, 0.5 geometrically)
    outputs = [numpy.zeros(shape) for i in range(vertex_vals.shape[1])]
    mask = numpy.zeros(shape, dtype=bool)
    for i in range(len(triangle_strip) - 2):
        vertices = triangle_strip[i:i+3]
        vals = vertex_vals[i:i+3]
        xmn, ymn = numpy.floor(vertices.min(axis=0)).astype(int)
        xmx, ymx = numpy.ceil(vertices.max(axis=0)).astype(int) + 1
        xs, ys = slice(xmn, xmx), slice(ymn, ymx)
        b_coords = barycentric_coords(vertices, grid[:, xs, ys])
        m = (b_coords >= 0).all(axis=0)
        mask[xs, ys] |= m
        b_m = b_coords[:, m]
        for j, out in enumerate(outputs):
            out[xs, ys][m] = vals[:, j].dot(b_m)
    if unpack_out:
        outputs = outputs[0]
    return mask, outputs

def accumulate_triangles(triangle_strip, shape):
    """Return a triangle strip rasterized such that each pixel contains
    a count of the number of triangles atop it.

    Parameters:
        triangle_strip: shape (n, 2) array of vertices describing a strip of
            connected triangles (such that vertices (0,1,2) describe the first
            triangle, vertices (1,2,3) describe the second, and so forth).
        shape: shape of the output image.
    """
    triangle_strip = numpy.asarray(triangle_strip)
    assert triangle_strip.ndim == 2 and triangle_strip.shape[1] == 2 and len(triangle_strip) > 2
    grid = numpy.indices(shape) + 0.5 # pixel centers are at (0.5, 0.5 geometrically)
    output = numpy.zeros(shape, dtype=int)
    for i in range(len(triangle_strip) - 2):
        vertices = triangle_strip[i:i+3]
        xmn, ymn = numpy.floor(vertices.min(axis=0)).astype(int)
        xmx, ymx = numpy.ceil(vertices.max(axis=0)).astype(int) + 1
        xs, ys = slice(xmn, xmx), slice(ymn, ymx)
        b_coords = barycentric_coords(vertices, grid[:, xs, ys])
        m = (b_coords >= 0).all(axis=0)
        output[xs, ys] += (b_coords >= 0).all(axis=0)
    return output
