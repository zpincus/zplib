import numpy
import celiagg

def draw_mask(image_shape, geometry, antialias=False):
    """Draw a mask (0-255) from a given celiagg path.

    Note: to produce a True/False mask from the output, simply do the following:
        mask = draw_mask(image_shape, geometry)
        bool_mask = mask > 0

    Parameters:
        image_shape: shape of the resulting image
        geometry: celiagg VertexSource class, such as celiagg.Path or
            celiagg.BSpline, containing geometry to draw
        antialias: if False (default), output contains only 0 and 255. If True,
            output will be antialiased (better for visualization).

    Returns: mask array of dtype numpy.uint8
    """
    image = numpy.zeros(image_shape, dtype=numpy.uint8, order='F')
    # NB celiagg uses (h, w) C-order convention for image shapes, so give it the transpose
    canvas = celiagg.CanvasG8(image.T)
    state = celiagg.GraphicsState(drawing_mode=celiagg.DrawingMode.DrawFill, anti_aliased=antialias)
    fill = celiagg.SolidPaint(1,1,1)
    transform = celiagg.Transform()
    canvas.draw_shape(geometry, transform, state, fill=fill)
    return image
