import numpy

def cumulative_distances(points, unit=True):
    """Return cumulative distances along a polyline.

    Parameters:
    points: array of shape (n,m) consisting of n points in m dimensions
    unit: if True, return distances divided by total length of the curve,
          if False, return actual arc lengths."""
    points = numpy.asarray(points)
    distances = numpy.concatenate([[0], numpy.add.accumulate(numpy.sqrt(((points[:-1] - points[1:])**2).sum(axis=1)))])
    if unit:
        distances /= distances[-1]
    return distances

def filter_dup_points(points):
    """Return a polyline with no duplicate or near-duplicate points."""
    points_out = [points[0]]
    for point in points[1:]:
        if not numpy.allclose(point, points_out[-1]):
            points_out.append(point)
    return numpy.array(points_out)

def closest_point(point, points):
    """Find the closest position in in array 'points' to the provided 'point'."""
    points_arr = numpy.asarray(points)
    distances_squared = ((points_arr - point)**2).sum(axis=1)
    i = numpy.argmin(distances_squared)
    return i, points[i]

def closest_point_to_line_segments(point, lines_start, lines_end):
    """Given a point and a set of line segments (specified by starting
    and ending points), return the point on each line segment that is closest to the
    given point, and the parametric position along each line of that point."""
    v = lines_end - lines_start
    w = point - lines_start
    c1 = (v*w).sum(axis=1)
    c2 = (v*v).sum(axis=1)
    fractional_positions = c1 / c2
    fractional_positions = fractional_positions.clip(0, 1)
    closest_points = lines_start + fractional_positions[:,numpy.newaxis]*v
    return closest_points, fractional_positions

def closest_point_on_polyline(point, points, parameters=None):
    """Return the point along a polyline nearest the given point and the parametric
    position of that point along the polyline. If no input parameter values are
    given, then the cumulative distance along the polyline will be taken as the parameter."""
    points = numpy.asarray(points)
    closest_points, fractions = closest_point_to_line_segments(point, points[:-1], points[1:])
    distances = numpy.sqrt(((point - closest_points)**2).sum(axis=1))
    point_idx = distances.argmin()
    closest_point = closest_points[point_idx]
    if parameters is None:
        parameters = cumulative_distances(spine)
    start_u, stop_u = parameters[point_idx:point_idx+2]
    u_val = start_u + fractions[point_idx]*(stop_u - start_u)
    return closest_point, u_val

def angle_between_vectors(v_from, v_to):
    """Calculate the angle in radians between two 2d vectors."""
    return numpy.arctan2(v_from[:,0]*v_to[:,1]-v_from[:,1]*v_to[:,0], (v_from * v_to).sum(axis=1))

def find_perp(p0, p1, unit=True):
    """Return a perpendicular to line p0-p1, optionally of unit-length.

    Parameters:
    p0 and p1 can be arrays of shape (m) each representing a single point in m dimensions,
    or of (n,m) containing n points in m dimensions. The returned array is either a
    array of shape (m) containing one or of shape (n,m) contaning n perpendiculars."""
    p1 = p1.astype(float)
    diff = p1 - p0
    diff = numpy.roll(diff, 1, axis=-1)
    diff[...,0] *= -1
    if unit:
        diff /= numpy.sqrt(numpy.sum(diff**2, axis=-1))[...,numpy.newaxis]
    return diff

def find_bisector(p0, p1, p2):
    """Find offset vector from p1 that bisects the angle created by three
    2d points, p0, p1, and p2."""
    d1 = p1 - p0
    d2 = p2 - p1
    a1 = numpy.arctan2(d1[...,1], d1[...,0])
    a2 = numpy.arctan2(d2[...,1], d2[...,0])
    ad = a2 - a1
    af = a1 + (numpy.pi + ad) / 2
    return numpy.transpose([numpy.cos(af), numpy.sin(af)])

def find_polyline_perpendiculars(points):
    """Find perpendiculars to an input polyline of shape (n, 2).
    At the endpoints, the perpendiculars are at right angles to the starting
    and ending line segments. At internal points, the perpendiculars bisect
    the angle of the internal positions.
    Note: special care is taken to make sure the perpindiculars point in a
    consistent direction at the internal positions, based on the overall direction
    of the polyline."""
    perpendiculars = numpy.empty(points.shape, dtype=float)
    perpendiculars[[0,-1]] = find_perp(points[[0, -2]], points[[1, -1]])
    bisectors = find_bisector(points[:-2], points[1:-1], points[2:])
    perps = find_perp(points[:-2], points[2:])
    dots = (bisectors * perps).sum(axis=1)
    perpendiculars[1:-1] = bisectors * numpy.sign(dots)[..., numpy.newaxis]
    return perpendiculars
