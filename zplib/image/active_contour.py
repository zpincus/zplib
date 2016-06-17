"""Geodesic Active Contours and Chan-Vese "Active Contours Without Edges"

Implementation based on morphological variant of these algorithms per:
Marquez-Neila P, Baumela L, & Alvarez L. (2014).
A morphological approach to curvature-based evolution of curves and surfaces.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(1).

This is a much-optimized version of the demo code available here:
https://github.com/pmneila/morphsnakes
In particular, a "narrow-band" approach is used whereby only pixels at the
edges of the mask are examined / updated. This speeds the code up by at least
several orders of magnitude compared to the naive approach.

Example GAC usage:
    image # image containing a structure to segment
    bounds # boolean mask with False for any image regions from which the
           # segmented structure must be excluded. (Optional.)
    initial # boolean mask with initial segmentation

    edges = ndimage.gaussian_gradient_magnitude(image, sigma)
    strong_edges = edges > threshold
    edge_gradient = numpy.gradient(edges)
    balloon_direction = -1
    # negative balloon_direction values means "deflate" the initial mask toward
    # edges except where strong_edges == True. (NB: balloon_direction can also
    # be a mask of -1, 0, and 1 values to allow spatially-varying balloon forces.)
    gac = GACMorphology(mask=initial, advection_direction=edge_gradient,
        advection_mask=strong_edges, balloon_direction=balloon_direction,
        max_region_mask=bounds)

    stopper = StoppingCondition(gac, max_iterations=100)
    while stopper.should_continue():
        gac.balloon_force() # apply balloon force
        gac.advect() # move region edges in advection_direction
        gac.smooth() # smooth region boundaries.

Obviously, different schedules of the balloon force, advection, and smoothing
can be applied. To aid in this, each of the methods above take an 'iters'
parameter to apply that many iterations of that step in a row. Also, initial
warm-up rounds with just advection or advection and smoothing may be helpful.

The smooth() method takes a 'depth' parameter that controls the spatial
smoothness of the curve. With depth=1, only very small jaggies are smoothed, but
with larger values, the curve is smoothed along larger spatial scales (rule of
thumb: min radius of curvature after smoothing is on the order of depth*3 or so).

Another useful way to calculate the advection_direction parameter, instead of
from the edge gradient (as above) is to use the edge_direction() function, which
takes a thresholded edge mask (Canny edges work great), and returns the distance
from each pixel to the nearest edge, as well as the direction from each pixel
to the nearest edge. This latter can be passed directly as the advection_direction
parameter, which will dramatically increase the capture radius for distant edges,
so a balloon_force parameter may not be needed.

Last, for Canny edge images especially, consider the EdgeClaimingAdvection
active contour class. (See the documentation for that class.)

Example ACWE usage:
    acwe = ACWEMorphology(mask=initial, image=image, max_region_mask=bounds)

    for i in range(iterations):
        acwe.acwe_step() # make inside and outside pixel means different.
        acwe.smooth() # smooth region boundaries

There is also a BinnedACWE class that uses a histogram of the image for the
inside and outside regions, rather than just means. This can be useful if
the region to segment has very distinct brightness values from the background,
but is not overall brighter or darker on average. Note that it is best to
reduce the input image to a small number of brightness values (16-64) before
using this class. The discretize_image() function is useful for this.

Last: the ActiveContour class allows for both GAC and ACWE steps, if that's
useful. BinnedActiveContour, ActiveClaimingContour, and BinnedActiveClaimingContour
similarly combine [Binned]ACWE and [EdgeClaiming]Advection functionality.
"""

import numpy
import itertools
import collections
from scipy import ndimage

from . import neighborhood

class MaskNarrowBand:
    """Track the inside and outside edge of a masked region, while allowing
    pixels from the inside edge to be moved outside and vice-versa.

    Base-class for fast morphological operations for region growing, shrinking,
    and reshaping.
    """
    S = ndimage.generate_binary_structure(2, 2)

    def __init__(self, mask, max_region_mask=None):
        mask = mask.astype(bool)
        if max_region_mask is not None:
            mask = mask & max_region_mask
        self.max_region_mask = max_region_mask
        self.mask_neighborhood = neighborhood.make_neighborhood_view(mask > 0,
            pad_mode='constant', constant_values=0) # shape = image.shape + (3, 3)
        # make self.mask identical to mask, but be a view on the center pixels of mask_neighborhood
        self.mask = self.mask_neighborhood[:,:,1,1]
        self.indices = numpy.dstack(numpy.indices(mask.shape)).astype(numpy.int32) # shape = mask.shape + (2,)
        self.index_neighborhood = neighborhood.make_neighborhood_view(self.indices,
            pad_mode='constant', constant_values=-1) # shape = mask.shape + (3, 3, 2)
        inside_border_mask = mask ^ ndimage.binary_erosion(mask, self.S) # all True pixels with a False neighbor
        outside_border_mask = mask ^ ndimage.binary_dilation(mask, self.S) # all False pixels with a True neighbor
        self.inside_border_indices = self.indices[inside_border_mask] # shape = (inside_border_mask.sum(), 2)
        self.outside_border_indices = self.indices[outside_border_mask] # shape = (outside_border_mask.sum(), 2)
        # NB: to index a numpy array with these indices, must turn shape(num_indices, 2) array
        # into tuple of two num_indices-length arrays, a la:
        # self.mask[tuple(self.outside_border_indices.T)]
        self.changed = 0

    def _assert_invariants(self):
        """Test whether the border masks and indices are in sync, and whether the
        borders are correct for the current mask."""
        inside_border_mask = self.mask ^ ndimage.binary_erosion(self.mask, self.S)
        outside_border_mask = self.mask ^ ndimage.binary_dilation(self.mask, self.S)
        assert len(self.inside_border_indices) == inside_border_mask.sum()
        assert inside_border_mask[tuple(self.inside_border_indices.T)].sum() == len(self.inside_border_indices)
        assert len(self.outside_border_indices) == outside_border_mask.sum()
        assert outside_border_mask[tuple(self.outside_border_indices.T)].sum() == len(self.outside_border_indices)

    def _move_to_outside(self, to_outside):
        """to_outside must be a boolean mask on the inside border pixels
        (in the order defined by inside_border_indices)."""
        self.inside_border_indices, self.outside_border_indices, added_idx, removed_indices = self._change_pixels(
            to_change=to_outside,
            old_border_indices=self.inside_border_indices,
            new_value=False,
            new_border_indices=self.outside_border_indices,
            some_match_old_value=numpy.any
        )
        return added_idx, removed_indices

    def _move_to_inside(self, to_inside):
        """to_inside must be a boolean mask on the outside border pixels
        (in the order defined by outside_border_indices)."""
        self.outside_border_indices, self.inside_border_indices, added_idx, removed_indices = self._change_pixels(
            to_change=to_inside,
            old_border_indices=self.outside_border_indices,
            new_value=True,
            new_border_indices=self.inside_border_indices,
            some_match_old_value=_not_all
        )
        return added_idx, removed_indices

    def _change_pixels(self, to_change, old_border_indices, new_value, new_border_indices, some_match_old_value):
        change_indices = old_border_indices[to_change]
        # prevent changes outside of the max_region_mask
        if new_value == True and self.max_region_mask is not None:
            in_region_mask = self.max_region_mask[tuple(change_indices.T)]
            change_indices = change_indices[in_region_mask]
            to_change[to_change] = in_region_mask
        change_idx = tuple(change_indices.T)
        if len(change_indices) == 0:
            return old_border_indices, new_border_indices, change_idx, change_indices
        self.changed += len(change_indices)
        # Find out which neighbors of changing pixels have the new value.
        # If we did the below after changing the mask, we would also pick up the
        # center pixels, which have the new value.
        if new_value == True:
            new_valued_neighbors = self.mask_neighborhood[change_idx]
        else:
            new_valued_neighbors = ~self.mask_neighborhood[change_idx]
        # Now, update the mask and border indices
        # (1) Update changed pixels in the mask, and remove them from old_border_indices.
        self.mask[change_idx] = new_value
        old_border_indices = old_border_indices[~to_change]

        # (2) add old-valued neighbors of newly-changed pixels to old_border_indices,
        # and then make sure the indices don't contain duplicates.
        # (Duplicates appear both because the indices might already be in the list,
        # or because the neighborhoods overlap, so old-valued neighbors might be
        # mulitply identified from several changed pixels.)
        changed_neighborhood = self.mask_neighborhood[change_idx]
        changed_neighborhood_indices = self.index_neighborhood[change_idx]
        # Find out which neighbors of changing pixels have the old value.
        # If we did the below before changing the mask, we would also pick up the
        # center pixels, which had the old value
        if new_value == True:
            old_valued_neighbors = ~changed_neighborhood
        else:
            old_valued_neighbors = changed_neighborhood
        old_valued_neighbor_indices = changed_neighborhood_indices[old_valued_neighbors]
        if new_value == True:
            # Exclude neighbors that are actually out-of-bounds "padding" pixels.
            # Only relevant for changing values to True (i.e. moving pixels inside)
            # because the out-of-bounds mask area is "False" and will thus otherwise
            # get picked up.
            # Out of bounds indices are -1 in the index_neighborhood array.
            good_indices = ~(old_valued_neighbor_indices == -1).any(axis=1)
            old_valued_neighbor_indices = old_valued_neighbor_indices[good_indices]
        # NB: many of the old_valued_neighbors are already in the old_border_indices...
        # If we kept a mask of the old_border pixels, we could look these up and
        # exclude them, which would make _unique_indices() below a bit faster. However,
        # that requires a lot of bookkeeping, so it doesn't speed things up in every case.
        old_border_indices = numpy.concatenate([old_border_indices, old_valued_neighbor_indices])
        old_border_indices = _unique_indices(old_border_indices)

        # (3) Remove all pixels from new_border_indices that no longer have any
        # old-valued neighbors left.
        # Such pixels must be a new-valued neighbor of one of the pixels
        # that changed to the new value. We know that these pixels are necessarily
        # in the new_border already because they are next to a pixel that changed.
        new_valued_neighbors_indices = changed_neighborhood_indices[new_valued_neighbors]
        # need to unique-ify indices because neighborhoods overlap and may pick up the same pixels
        new_valued_neighbors_indices = _unique_indices(new_valued_neighbors_indices)
        neighbors_of_new_valued_neighbors = self.mask_neighborhood[tuple(new_valued_neighbors_indices.T)]
        no_old_valued_neighbors = ~some_match_old_value(neighbors_of_new_valued_neighbors, axis=(1,2))
        remove_from_new_border_indices = new_valued_neighbors_indices[no_old_valued_neighbors]
        new_border_indices = _diff_indices(new_border_indices, remove_from_new_border_indices)

        # (4) Add newly-changed pixels to new_border_indices if they have an old-valued neighbor.
        changed_with_old_neighbor = some_match_old_value(changed_neighborhood, axis=(1,2))
        add_to_new_border_indices = change_indices[changed_with_old_neighbor]
        new_border_indices = numpy.concatenate([new_border_indices, add_to_new_border_indices])
        return old_border_indices, new_border_indices, change_idx, remove_from_new_border_indices

class CurvatureMorphology(MaskNarrowBand):
    """Implement basic erosion, dilation, and curvature-smoothing morphology
    steps (the latter from Marquez-Neila et al.) using a fast narrow-band approach.
    Base class for more sophisticated region-modifying steps: main function of interest
    is smooth().

    smooth_mask, if not None, is region where smoothing may be applied.
    """
    def __init__(self, mask, smooth_mask=None, max_region_mask=None):
        super().__init__(mask, max_region_mask)
        self.smooth_mask = smooth_mask
        self._reset_smoothing()

    def _reset_smoothing(self):
        self._smooth_funcs = itertools.cycle([self._SIoIS, self._ISoSI])

    def dilate(self, iters=1, border_mask=None):
        for _ in range(iters):
            if border_mask is None:
                border_mask = numpy.ones(len(self.outside_border_indices), dtype=bool)
            self._move_to_inside(border_mask)

    def erode(self, iters=1, border_mask=None):
        for _ in range(iters):
            if border_mask is None:
                border_mask = numpy.ones(len(self.inside_border_indices), dtype=bool)
            self._move_to_outside(border_mask)

    def smooth(self, iters=1, depth=1):
        """Apply 'iters' iterations of edge-curvature smoothing.
        'depth' controls the spatial scale of the smoothing. With depth=1, only
        the highest-frequency edges get smoothed out. Larger depth values smooth
        lower-frequency structures."""
        for _ in range(iters):
            smoother = next(self._smooth_funcs)
            smoother(depth)

    def _SI(self):
        idx, idx_mask = _masked_idx(self.smooth_mask, self.inside_border_indices)
        inside_border = self.mask_neighborhood[idx]
        on_a_line = ((inside_border[:,0,0] & inside_border[:,2,2]) |
                     (inside_border[:,1,0] & inside_border[:,1,2]) |
                     (inside_border[:,0,1] & inside_border[:,2,1]) |
                     (inside_border[:,2,0] & inside_border[:,0,2]))
        self._move_to_outside(_unmask_idx(idx_mask, ~on_a_line))

    def _IS(self):
        idx, idx_mask = _masked_idx(self.smooth_mask, self.outside_border_indices)
        outside_border = ~self.mask_neighborhood[idx]
        on_a_line = ((outside_border[:,0,0] & outside_border[:,2,2]) |
                     (outside_border[:,1,0] & outside_border[:,1,2]) |
                     (outside_border[:,0,1] & outside_border[:,2,1]) |
                     (outside_border[:,2,0] & outside_border[:,0,2]))
        self._move_to_inside(_unmask_idx(idx_mask, ~on_a_line))

    def _SIoIS(self, depth=1):
        for i in range(depth):
            self._IS()
        for i in range(depth):
            self._SI()

    def _ISoSI(self, depth=1):
        for i in range(depth):
            self._SI()
        for i in range(depth):
            self._IS()

class ACWE(CurvatureMorphology):
    def __init__(self, mask, image, acwe_mask=None, inside_bias=1,
        smooth_mask=None, max_region_mask=None):
        """Class for Active Contours Without Edges region-growing.

        Relevant methods for region-growing are smooth() and acwe_step().

        Parameters:
            mask: mask containing the initial state of the region to evolve
            image: ndarray of same shape as mask containing image values. The
                difference in mean image value inside and outside the region will
                be maximized by acwe_step()
            acwe_mask: region where ACWE updates will be applied. (Inside /
                outside values will still be computed outside this region.)
            inside_bias: weight for comparing means of inside vs. outside pixels.
                Generally 1 works properly. Values < 1 make it "easier" to add
                pixels to the region, and values > 1 make it "easier" to remove
                pixels from the region. (Simplification of the "lambda"
                parameters from the ACWE literature.)
            smooth_mask: region in which smoothing may be applied.
            max_region_mask: mask beyond which the region may not grow. If
                provided, this mask will also represent the pixels over which
                the ACWE inside/outside means are computed.
        """
        super().__init__(mask, smooth_mask, max_region_mask)
        # do work in _setup rather than __init__ to allow for complex multiple
        # inheritance from this class that super() alone can't handle. See
        # ActiveContour class.
        self._setup(image, acwe_mask, inside_bias)

    def _setup(self, image, acwe_mask, inside_bias):
        self.image = image
        assert self.image.shape == self.mask.shape
        self.acwe_mask = acwe_mask
        if acwe_mask is not None:
            assert self.image.shape == self.acwe_mask.shape
        self.inside_bias = inside_bias
        # note: self.mask is clipped to self.max_region_mask so the below works.
        self.inside_count = self.mask.sum()
        self.outside_count = numpy.product(self.mask[self.max_region_mask].shape) - self.inside_count
        self.inside_sum = self.image[self.mask].sum()
        self.outside_sum = self.image[self.max_region_mask].sum() - self.inside_sum

    def _assert_invariants(self):
        super()._assert_invariants()
        assert self.inside_count == self.mask.sum()
        assert self.outside_count == numpy.product(self.mask[self.max_region_mask].shape) - self.inside_count
        assert self.inside_sum == self.image[self.mask].sum()
        assert self.outside_sum == self.image[self.max_region_mask].sum() - self.inside_sum
        assert numpy.allclose(self.inside_sum/self.inside_count, self.image[self.mask].mean())
        if self.max_region_mask is None:
            assert numpy.allclose(self.outside_sum/self.outside_count, self.image[~self.mask].mean())
        else:
            assert numpy.allclose(self.outside_sum/self.outside_count, self.image[self.max_region_mask & ~self.mask].mean())

    def _image_sum_count(self, changed_idx):
        count = len(changed_idx[0])
        image_sum = self.image[changed_idx].sum()
        return count, image_sum

    def _move_to_outside(self, to_outside):
        """to_outside must be a boolean mask on the inside border pixels
        (in the order defined by inside_border_indices)."""
        added_idx, removed_indices = super()._move_to_outside(to_outside)
        count, image_sum = self._image_sum_count(added_idx)
        self.inside_count -= count
        self.outside_count += count
        self.inside_sum -= image_sum
        self.outside_sum += image_sum
        return added_idx, removed_indices

    def _move_to_inside(self, to_inside):
        """to_inside must be a boolean mask on the outside border pixels
        (in the order defined by outside_border_indices)."""
        added_idx, removed_indices = super()._move_to_inside(to_inside)
        count, image_sum = self._image_sum_count(added_idx)
        self.outside_count -= count
        self.inside_count += count
        self.outside_sum -= image_sum
        self.inside_sum += image_sum
        return added_idx, removed_indices

    def acwe_step(self, iters=1):
        """Apply 'iters' iterations of the Active Contours Without Edges step,
        wherein the region inside the mask is made to have a mean value as different
        from the region outside the mask as possible."""
        for _ in range(iters):
            if self.inside_count == 0 or self.outside_count == 0:
                return
            inside_mean = self.inside_sum / self.inside_count
            outside_mean = self.outside_sum / self.outside_count
            self._acwe_step(inside_mean, outside_mean, self.inside_bias,
                self.inside_border_indices, self._move_to_outside)
            self._acwe_step(outside_mean, inside_mean, 1/self.inside_bias,
                self.outside_border_indices, self._move_to_inside)

    def _acwe_step(self, mean_from, mean_to, from_bias, border_indices, move_operation):
        idx, idx_mask = _masked_idx(self.acwe_mask, border_indices)
        border_values = self.image[idx]
        to_move = from_bias*(border_values - mean_from)**2 > (border_values - mean_to)**2
        move_operation(_unmask_idx(idx_mask, to_move))

class BinnedACWE(CurvatureMorphology):
    def __init__(self, mask, image, acwe_mask=None, inside_bias=1,
        smooth_mask=None, max_region_mask=None):
        """Class for Active Contours Without Edges region-growing, using image
        histograms rather than simply means.

        Relevant methods for region-growing are smooth(), and acwe_step().

        Note that it is best to reduce the input image to a small number of
        brightness values (e.g. 16-64) before using this class.
        The discretize_image() function is useful for this.

        Parameters:
            mask: mask containing the initial state of the region to evolve
            image: ndarray of same shape as mask containing image values. The
                difference in image histograms inside and outside the region will
                be maximized by acwe_step()
            acwe_mask: region where ACWE updates will be applied. (Inside /
                outside values will still be computed outside this region.)
            inside_bias: weight for comparing means of inside vs. outside pixels.
                Generally 1 works properly. Values < 1 make it "easier" to add
                pixels to the region, and values > 1 make it "easier" to remove
                pixels from the region. (Simplification of the "lambda"
                parameters from the ACWE literature.)
            smooth_mask: region in which smoothing may be applied.
            max_region_mask: mask beyond which the region may not grow. If
                provided, this mask will also represent the pixels over which
                the ACWE inside/outside means are computed.
        """
        super().__init__(mask, smooth_mask, max_region_mask)
        # do work in _setup rather than __init__ to allow for complex multiple
        # inheritance from this class that super() alone can't handle. See
        # ActiveContour class.
        self._setup(image, acwe_mask, inside_bias)

    def _setup(self, image, acwe_mask, inside_bias):
        if image.dtype == bool:
            image = image.astype(numpy.uint8)
        self.image = image
        assert self.image.shape == self.mask.shape
        self.acwe_mask = acwe_mask
        if acwe_mask is not None:
            assert self.image.shape == self.acwe_mask.shape
        # note: self.mask is clipped to self.max_region_mask so the below works.
        self.inside_count = self.mask.sum()
        self.inside_bias = inside_bias
        self.outside_count = numpy.product(self.mask[self.max_region_mask].shape) - self.inside_count
        bincounts = numpy.bincount(self.image[self.max_region_mask].flat)
        self.bins = len(bincounts)
        self.inside_bincounts = self._bincount(self.mask)
        self.outside_bincounts = bincounts - self.inside_bincounts

    def _bincount(self, index_exp):
        return numpy.bincount(self.image[index_exp], minlength=self.bins)

    def _assert_invariants(self):
        super()._assert_invariants()
        assert self.inside_count == self.mask.sum()
        assert self.outside_count == numpy.product(self.mask[self.max_region_mask].shape) - self.inside_count
        assert (self.inside_bincounts == self._bincount(self.mask)).all()
        assert (self.outside_bincounts == self._bincount(self.max_region_mask) - self.inside_bincounts).all()
        if self.max_region_mask is not None:
            assert (self.outside_histogram == self._bincount(self.max_region_mask & ~self.mask)).all()

    def _image_bincount_count(self, changed_idx):
        count = len(changed_idx[0])
        bincounts = self._bincount(changed_idx)
        return count, bincounts

    def _move_to_outside(self, to_outside):
        """to_outside must be a boolean mask on the inside border pixels
        (in the order defined by inside_border_indices)."""
        added_idx, removed_indices = super()._move_to_outside(to_outside)
        count, bincounts = self._image_bincount_count(added_idx)
        self.inside_count -= count
        self.outside_count += count
        self.inside_bincounts -= bincounts
        self.outside_bincounts += bincounts
        return added_idx, removed_indices

    def _move_to_inside(self, to_inside):
        """to_inside must be a boolean mask on the outside border pixels
        (in the order defined by outside_border_indices)."""
        added_idx, removed_indices = super()._move_to_inside(to_inside)
        count, bincounts = self._image_bincount_count(added_idx)
        self.outside_count -= count
        self.inside_count += count
        self.outside_bincounts -= bincounts
        self.inside_bincounts += bincounts
        return added_idx, removed_indices

    def acwe_step(self, iters=1):
        """Apply 'iters' iterations of the Active Contours Without Edges step,
        wherein the region inside the mask is made to have a mean value as different
        from the region outside the mask as possible."""
        for _ in range(iters):
            if self.inside_count == 0 or self.outside_count == 0:
                return
            inside_rate = self.inside_bincounts / self.inside_count
            outside_rate = self.outside_bincounts / self.outside_count
            # small values of inside bias mean it is easier to move/stay inside
            self._acwe_step(self.inside_bias * outside_rate > inside_rate,
                self.inside_border_indices, self._move_to_outside)
            self._acwe_step(inside_rate > self.inside_bias * outside_rate,
                self.outside_border_indices, self._move_to_inside)

    def _acwe_step(self, move_bins, border_indices, move_operation):
        idx, idx_mask = _masked_idx(self.acwe_mask, border_indices)
        border_values = self.image[idx]
        to_move = move_bins[border_values]
        move_operation(_unmask_idx(idx_mask, to_move))

class BalloonForceMorphology(CurvatureMorphology):
    """Basic morphology operations plus spatially-varying balloon-force operation.
    Base-class to add balloon forces to more complex region-growing steps;
    rarely useful directly.
    """
    def __init__(self, mask, balloon_direction, smooth_mask=None, max_region_mask=None):
        """balloon_direction: (-1, 0, 1), or ndarray with same shape as 'mask'
        containing those values."""
        super().__init__(mask, smooth_mask, max_region_mask)
        # do work in _setup rather than __init__ to allow for complex multiple
        # inheritance from this class that super() alone can't handle. See
        # ActiveContour class.
        self._setup(balloon_direction)

    def _setup(self, balloon_direction):
        if numpy.isscalar(balloon_direction):
            if balloon_direction == 0:
                self.balloon_direction = None
            else:
                self.balloon_direction = numpy.zeros(self.mask.shape, dtype=numpy.int8)
                self.balloon_direction += balloon_direction
        else:
            self.balloon_direction = balloon_direction.copy() # may get changed internally by subclasses

    def balloon_force(self, iters=1):
        """Apply 'iters' iterations of balloon force region expansion / shrinkage."""
        if self.balloon_direction is None:
            return
        for _ in range(iters):
                to_erode = self.balloon_direction[tuple(self.inside_border_indices.T)] < 0
                self.erode(border_mask=to_erode)
                to_dilate = self.balloon_direction[tuple(self.outside_border_indices.T)] > 0
                self.dilate(border_mask=to_dilate)

class GAC(BalloonForceMorphology):
    def __init__(self, mask, advection_direction, advection_mask=None,
        balloon_direction=0, smooth_mask=None, max_region_mask=None):
        """Class for Geodesic Active Contours region-growing.

        Relevant methods for region-growing are smooth(), balloon_force(),
        and advect().

        Parameters:
            mask:  mask containing the initial state of the region to evolve
            advection_direction: list of two arrays providing the x- and y-
                coordinates of the direction that the region edge should move
                in at any given point. (Only the sign of the direction matters.)
                The gradient of an edge-magnitude image works nicely here, as
                does the result of the edge_direction() function in this module
                when applied to a thresholded edge-magnitude image.
            advection_mask: boolean mask specifying where edge advection should
                be applied (versus balloon forces -- it makes no sense to try
                to apply both in the same location). If no advection_mask is
                provided, but a balloon_direction map is given, assume that
                advection is to be applied wherever the balloon_direction is
                0. If a scalar, non-zero balloon_direction is given, and no
                advection_mask is provided, then there will be no edge
                advection.
            balloon_direction: scalar balloon force direction (-1, 0, 1) or
                image map of same values. If advection_mask is provided,
                balloon_direction will be zeroed out in regions of where
                advection is allowed.
            smooth_mask: region in which smoothing may be applied.
            max_region_mask: mask beyond which the region may not grow.
        """
        CurvatureMorphology.__init__(self, mask, smooth_mask, max_region_mask)
        # do work in _setup rather than __init__ to allow for complex multiple
        # inheritance from this class that super() alone can't handle. See
        # ActiveContour class.
        self._setup(balloon_direction, advection_direction, advection_mask)

    def _setup(self, balloon_direction, advection_direction, advection_mask):
        BalloonForceMorphology._setup(self, balloon_direction)
        self.adv_dir_x, self.adv_dir_y = advection_direction
        # None balloon direction means no balloon force was asked for.
        if advection_mask is None:
            if self.balloon_direction is not None:
                self.advection_mask = self.balloon_direction == 0
            else:
                self.advection_mask = None
        else:
            self.advection_mask = advection_mask
            if self.balloon_direction is not None:
                self.balloon_direction[advection_mask] = 0

    def advect(self, iters=1):
        """Apply 'iters' iterations of edge advection, whereby the region edges
        are moved in the direction specified by advection_direction."""
        for _ in range(iters):
            # Move pixels on the inside border to the outside if advection*gradient sum > 0 (see _advect for interpretation of sum)
            self._advect(self.inside_border_indices, numpy.greater, self._move_to_outside)
            # Move pixels on the outside border to the inside if advection*gradient sum < 0
            self._advect(self.outside_border_indices, numpy.less, self._move_to_inside)

    def _advect(self, border_indices, criterion, move_operation):
        idx, idx_mask = _masked_idx(self.advection_mask, border_indices)
        neighbors = self.mask_neighborhood[idx].astype(numpy.int8)
        dx = neighbors[:,2,1] - neighbors[:,0,1]
        dy = neighbors[:,1,2] - neighbors[:,1,0]
        adv_dx = self.adv_dir_x[idx]
        adv_dy = self.adv_dir_y[idx]
        # positive gradient => outside-to-inside in that dimension
        # positive advection direction => edge should move in the positive direction
        # So:  + gradient / + advection = move pixels outside
        # + gradient / - advection or - gradient / + advection = move pixels inside
        # Tricky case: x and y disagree = go in direction with largest abs advection direction
        # To find this, see if sum of advection and gradient in each direction is > 0
        # (move pixels outside) or > 0 (move pixels inside).
        to_move = criterion(dx * adv_dx + dy * adv_dy, 0)
        move_operation(_unmask_idx(idx_mask, to_move))

class EdgeClaimingAdvection(CurvatureMorphology):
    def __init__(self, mask, edge_mask, distance_exponent=3, force_min=0.01, smooth_mask=None, max_region_mask=None):
        """Class for growing a region toward edges, such that only the edges
        that are not already at the region border influence further region
        growth.

        Given a set of image edges (ideally one pixel wide, as produced by
        Canny filtering), a region is grown toward those edges. The direction
        of growth at a region border pixel is determined by the sum of "forces",
        where each edge pixel exerts a force in inverse proportion to its
        distance from the region border pixel. This allows the region to move
        over large distances to find an edge.

        Only region border pixels that are not atop an edge pixel are free to
        have move. Only edge pixels not "captured" by a region border are free
        to exert forces on movable region border pixels. This capturing
        mechaimsm means that gaps in the edges will not cause region borders to
        "flow in" toward other edges in the image that already have a different
        part of the border atop them. Instead, gaps in the edges will be bridged
        smoothly.

        Parameters:
            mask:  mask containing the initial state of the region to evolve.
            edge_mask: mask containing the positions of the edges to occupy.
                Thinned edges (such as from Canny edge detection) are best.
            distance_exponent: the "force" on any region border pixel exerted
                by any edge pixel is 1/d**distance_exponent, where d is the
                distance in pixels between the edge pixel and the region border
                pixel. A larger value increases the influence of nearby
                uncaptured edge pixels over more distant uncaptured edge pixels.
            force_min: forces less than this minimum value will cause region
                borders to move. Important to prevent a few distant, unclaimed
                pixels from influencing borders that are bridging gaps.
            smooth_mask: region in which smoothing may be applied.
            max_region_mask: mask beyond which the region may not grow.
        """
        super().__init__(mask, smooth_mask, max_region_mask)
        # do work in _setup rather than __init__ to allow for complex multiple
        # inheritance from this class that super() alone can't handle. See
        # ActiveContour class.
        self._setup(edge_mask, distance_exponent, force_min)

    def _setup(self, edge_mask, distance_exponent, force_min):
        # Edges are "captured" iff there is an inside-border pixel on the edge.
        # Non-captured edges can generate advection forces.
        # Inside border pixels are free to advect iff they are not atop an edge.
        # Outside border pixels are free to advect iff they are (atop an edge
        # OR not adjacent a captured edge).
        self.edge_mask = edge_mask > 0
        # make an exponent to apply to the squared distance
        self.inverse_squared_distance_exponent = -distance_exponent/2
        self.force_min = force_min
        self.inside_free = ~self.edge_mask
        self.outside_free_neighborhood = neighborhood.make_neighborhood_view(numpy.zeros_like(self.edge_mask),
            pad_mode='constant', constant_values=0) # shape = edge_mask.shape + (3, 3)
        self.outside_free = self.outside_free_neighborhood[:,:,1,1]
        self.edge_indices = self.indices[self.edge_mask]
        self.noncaptured = numpy.zeros_like(self.edge_mask)
        self.update_captured()

    def update_captured(self):
        idx_mask = self.edge_mask[tuple(self.inside_border_indices.T)]
        captured_indices = self.inside_border_indices[idx_mask]
        captured_idx = tuple(captured_indices.T)
        self.captured_idx = captured_idx
        self.outside_free[:] = True
        self.outside_free[captured_idx] = False
        neighbor_indices = self.index_neighborhood[captured_idx] # shape = (m, 3, 3, 2)
        all_neighbors = neighbor_indices.reshape((numpy.product(neighbor_indices.shape[:-1]), 2)) # shape = (m*3*3, 2)
        all_idx = tuple(all_neighbors.T)
        # now erode to get pixels non-adjacent to captured edges
        self.outside_free[all_idx] = self.outside_free_neighborhood[all_idx].all()
        # Technically, we should set outside_free[captured_idx] to True,
        # but it doesn't matter because we will never look up those locations.
        # the outside_free array is only used to look at the outside_border_indices
        # positions, which are non-overlapping with inside_border_indices.
        # Now free outside pixels atop an edge:
        self.outside_free[tuple(self.edge_indices.T)] = True
        self.noncaptured_indices = _diff_indices(self.edge_indices, captured_indices)
        self.noncaptured[:] = False
        self.noncaptured[tuple(self.noncaptured_indices.T)] = True

    def advection_forces(self, border_indices, border_idx, mask_dx, mask_dy):
        """border_indices: shape (n, 2) list of indices"""
        edge_indices = self.noncaptured_indices # shape (m, 2)
        on_edge = self.noncaptured[border_idx] # shape (n,)
        dx = numpy.subtract.outer(edge_indices[:,0], border_indices[:,0]) # shape (m, n)
        dy = numpy.subtract.outer(edge_indices[:,1], border_indices[:,1])
        square_dist = dx**2 + dy**2
        square_dist[:,on_edge] = 1
        weighting = square_dist**self.inverse_squared_distance_exponent
        fx = (dx * weighting).sum(axis=0) # shape (n,)
        fx[on_edge] = -mask_dx[on_edge]
        fx[numpy.absolute(fx) < self.force_min] = 0
        fy = (dy * weighting).sum(axis=0)
        fy[on_edge] = -mask_dy[on_edge]
        fy[numpy.absolute(fy) < self.force_min] = 0
        return fx, fy

    def advect(self, iters=1):
        """Apply 'iters' iterations of edge advection, whereby uncaptured
        region edges are moved towrad uncaptured pixels in the edge_mask."""
        for _ in range(iters):
            self.update_captured()
            # Move pixels on the inside border to the outside if advection*gradient sum > 0 (see _advect for interpretation of sum)
            # and if they are not on an edge
            self._advect(self.inside_free, self.inside_border_indices, numpy.greater, self._move_to_outside)
            # Move pixels on the outside border to the inside if advection*gradient sum < 0
            # and if they arte not adjacent to a captured edge
            self._advect(self.outside_free, self.outside_border_indices, numpy.less, self._move_to_inside)

    def _advect(self, mask, border_indices, criterion, move_operation):
        idx_mask = mask[tuple(border_indices.T)]
        border_indices = border_indices[idx_mask]
        border_idx = tuple(border_indices.T)
        neighbors = self.mask_neighborhood[border_idx].astype(numpy.int8)
        dx = neighbors[:,2,1] - neighbors[:,0,1]
        dy = neighbors[:,1,2] - neighbors[:,1,0]
        adv_dx, adv_dy = self.advection_forces(border_indices, border_idx, dx, dy)
        # positive gradient => outside-to-inside in that dimension
        # positive advection direction => edge should move in the positive direction
        # So:  + gradient / + advection = move pixels outside
        # + gradient / - advection or - gradient / + advection = move pixels inside
        # Tricky case: x and y disagree = go in direction with largest abs advection direction
        # To find this, see if sum of advection and gradient in each direction is > 0
        # (move pixels outside) or > 0 (move pixels inside).
        to_move = criterion(dx * adv_dx + dy * adv_dy, 0)
        move_operation(_unmask_idx(idx_mask, to_move))


class ActiveContour(GAC, ACWE):
    def __init__(self, mask, image, advection_direction, acwe_mask=None,
            inside_bias=1, advection_mask=None, balloon_direction=0,
            smooth_mask=None, max_region_mask=None):
        """See documentation for GAC and ACWE for parameters."""
        CurvatureMorphology.__init__(self, mask, smooth_mask, max_region_mask)
        GAC._setup(self, balloon_direction, advection_direction, advection_mask)
        ACWE._setup(self, image, acwe_mask, inside_bias)

class BinnedActiveContour(GAC, BinnedACWE):
    def __init__(self, mask, image, advection_direction, acwe_mask=None,
            inside_bias=1, advection_mask=None, balloon_direction=0,
            smooth_mask=None, max_region_mask=None):
        """See documentation for GAC and BinnedACWE for parameters."""
        CurvatureMorphology.__init__(self, mask, smooth_mask, max_region_mask)
        GAC._setup(self, balloon_direction, advection_direction, advection_mask)
        BinnedACWE._setup(self, image, acwe_mask, inside_bias)

class ActiveClaimingContour(EdgeClaimingAdvection, ACWE):
    def __init__(self, mask, image, edge_mask, force_min=0.01, acwe_mask=None,
            inside_bias=1, smooth_mask=None, max_region_mask=None):
        """See documentation for EdgeClaimingAdvection and ACWE for parameters."""
        CurvatureMorphology.__init__(self, mask, smooth_mask, max_region_mask)
        EdgeClaimingAdvection._setup(self, edge_mask, force_min)
        ACWE._setup(self, image, acwe_mask, inside_bias)

class BinnedActiveClaimingContour(EdgeClaimingAdvection, BinnedACWE):
    def __init__(self, mask, image, edge_mask, distance_exponent=3, force_min=0.01,
            acwe_mask=None, inside_bias=1, smooth_mask=None, max_region_mask=None):
        """See documentation for EdgeClaimingAdvection and BinnedACWE for parameters."""
        CurvatureMorphology.__init__(self, mask, smooth_mask, max_region_mask)
        EdgeClaimingAdvection._setup(self, edge_mask, distance_exponent, force_min)
        BinnedACWE._setup(self, image, acwe_mask, inside_bias)


class StoppingCondition:
    def __init__(self, ac, max_iterations, sample_every=5, changed_min=5, cycle_max=6):
        """Class to simplify deciding when to stop active contour iteration,
        based on total number of iterations and changes to the mask.

        Usage example:
            ac = ACWE(mask, image)
            stopper = StoppingCondition(ac, max_iterations=50)
            while stopper.should_continue():
                ac.acwe_step()
                ac.smooth()

        Parameters:
            ac: active contour object of any type
            max_iterations: total number of iterations permitted
            sample_every: period at which the mask is examined to determine if
                any pixels have changed. Doing this every iteration can be slow,
                so this parameter allows trading off between stopping sooner and
                iterating faster.
            changed_min: if fewer than this number of pixels have changed, stop
                iteration.
            cycle_max: if the number of changed pixels is constant this many
                times in a row, stop iteration.
        """
        self.ac = ac
        self.max_iterations = max_iterations
        self.sample_every = sample_every
        self.changed_min = changed_min
        self.reset()

    def reset(self):
        self.recent_changes = collections.deque(maxlen=cycle_max-1)
        self.i = 0

    def should_continue(self):
        if self.i == self.max_iterations:
            return False
        if self.i % self.sample_every == 0:
            if self.i > 0:
                changed = (self.ac.mask != self.old_mask).sum()
                if changed < self.changed_min:
                    return False
                if (len(self.recent_changes) == self.recent_changes.maxlen and
                        all(prev == changed for prev in self.recent_changes)):
                    return False
                self.recent_changes.append(changed)
            self.old_mask = self.ac.mask.copy()
        self.i += 1
        return True

def edge_direction(edge_mask):
    """Given an edge mask, return the distance from each pixel to the mask,
    and the vector from each pixel to the nearest pixel in the mask.

    Parameter:
        edge_mask: boolean array that is True where image edges are.

    Returns: distances, nearest_edge
        distances: array same shape as edge_mask, containing the distance from
            every non-edge-mask pixel to the nearest pixel in the edge mask.
        nearest_edge: arary of shape (2,)+edge_mask.shape, containing the x and
            y coordinates of the vector from each non-edge pixel to the nearest
            edge pixel.
    """
    distances, nearest_edge = ndimage.distance_transform_cdt(~edge_mask,
        return_distances=True, return_indices=True)
    nearest_edge = nearest_edge - numpy.indices(edge_mask.shape)
    nearest_edge[:, edge_mask] = 0
    return distances, nearest_edge

def discretize_image(image, n_levels):
    """Given an image, reduce the number of distinct intensity levels.

    Parameters:
        image: ndarray of any type
        n_levels: number of intensity levels in the output image. Must be < 2**16.
    """
    assert n_levels < 2**16
    max = image.max()
    min = image.min()
    left_edges = numpy.linspace(min, max, n_levels)
    discretized = numpy.digitize(image, left_edges) - 1
    if n_levels > 256:
        return discretized.astype(numpy.uint8)
    else:
        return discretized.astype(numpy.uint16)

def _diff_indices(indices, to_remove):
    """Given two arrays of shape (n,2) containing x,y indices, remove those in
    the second array from the first array.
    """
    assert indices.flags.c_contiguous
    assert to_remove.flags.c_contiguous
    assert indices.dtype == to_remove.dtype
    dtype = numpy.dtype('S'+str(indices.itemsize*2)) # treat (x,y) indices as binary data instead of pairs of ints
    remaining = numpy.setdiff1d(indices.view(dtype), to_remove.view(dtype), assume_unique=True)
    return remaining.view(indices.dtype).reshape((-1, 2))

def _unique_indices(indices):
    """Given an array of shape (n,2) containing x,y indices, return only the
    unique indices from that array.
    """
    assert indices.flags.c_contiguous
    dtype = numpy.dtype('S'+str(indices.itemsize*2)) # treat (x,y) indices as binary data instead of pairs of ints
    unique = numpy.unique(indices.view(dtype))
    return unique.view(indices.dtype).reshape((-1, 2))

def _not_all(array, axis):
    return ~numpy.all(array, axis)

def _masked_idx(mask, indices):
    """Return index expression for given indices, only if mask is also true there.

    Parameters:
        mask: True/False mask
        indices: indices into array, of shape (n, mask.ndim)
    Returns:
        idx: numpy index expression for selected indices
        idx_mask: mask of same length as indices, indicating which indices had true
            values in the mask
    """
    if mask is None:
        idx_mask = None
    else:
        idx_mask = mask[tuple(indices.T)]
        indices = indices[idx_mask]
    idx = tuple(indices.T)
    return idx, idx_mask

def _unmask_idx(idx_mask, to_change):
    """Return mask over all indices originally passed to _masked_idx, marked
    False where indices were originally masked out or where to_change parameter
    is False.

    Parameters:
        idx_mask: as returned by _masked_idx
        to_change: mask over True values in idx_mask
    """
    if idx_mask is None:
        return to_change
    else:
        idx_mask[idx_mask] = to_change
        return idx_mask
