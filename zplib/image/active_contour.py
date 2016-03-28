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

    for i in range(iterations):
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

Example ACWE usage (balloon force not generally needed with ACWE, but
can be included):
    acwe = ACWEMorphology(mask=initial, image=image, max_region_mask=bounds)

    for i in range(iterations):
        acwe.acwe_step() # make inside and outside pixel means different.
        acwe.smooth() # smooth region boundaries

Note: ActiveContour class allows for both GAC and ACWE steps, if that's useful.
"""

import numpy
import itertools
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
        indices = numpy.dstack(numpy.indices(mask.shape)).astype(numpy.int32) # shape = mask.shape + (2,)
        self.index_neighborhood = neighborhood.make_neighborhood_view(indices,
            pad_mode='constant', constant_values=-1) # shape = mask.shape + (3, 3, 2)
        inside_border_mask = mask ^ ndimage.binary_erosion(mask, self.S) # all True pixels with a False neighbor
        outside_border_mask = mask ^ ndimage.binary_dilation(mask, self.S) # all False pixels with a True neighbor
        self.inside_border_indices = indices[inside_border_mask] # shape = (inside_border_mask.sum(), 2)
        self.outside_border_indices = indices[outside_border_mask] # shape = (outside_border_mask.sum(), 2)
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

    def move_to_outside(self, to_outside):
        """to_outside must be a boolean mask on the inside border pixels
        (in the order defined by inside_border_indices)."""
        self.inside_border_indices, self.outside_border_indices, changed_idx = self._change_pixels(
            to_change=to_outside,
            old_border_indices=self.inside_border_indices,
            new_value=False,
            new_border_indices=self.outside_border_indices,
            some_match_old_value=numpy.any
        )
        return changed_idx

    def move_to_inside(self, to_inside):
        """to_inside must be a boolean mask on the outside border pixels
        (in the order defined by outside_border_indices)."""
        self.outside_border_indices, self.inside_border_indices, changed_idx = self._change_pixels(
            to_change=to_inside,
            old_border_indices=self.outside_border_indices,
            new_value=True,
            new_border_indices=self.inside_border_indices,
            some_match_old_value=not_all
        )
        return changed_idx

    def _change_pixels(self, to_change, old_border_indices, new_value, new_border_indices, some_match_old_value):
        change_indices = old_border_indices[to_change]
        # prevent changes outside of the max_region_mask
        if new_value == True and self.max_region_mask is not None:
            in_region_mask = self.max_region_mask[tuple(change_indices.T)]
            change_indices = change_indices[in_region_mask]
            to_change[to_change] = in_region_mask
        change_idx = tuple(change_indices.T)
        if len(change_indices) == 0:
            return old_border_indices, new_border_indices, change_idx
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
        # exclude them, which would make unique_indices() below a bit faster. However,
        # that requires a lot of bookkeeping, so it doesn't speed things up in every case.
        old_border_indices = numpy.concatenate([old_border_indices, old_valued_neighbor_indices])
        old_border_indices = unique_indices(old_border_indices)

        # (3) Remove all pixels from new_border_indices that no longer have any
        # old-valued neighbors left.
        # Such pixels must be a new-valued neighbor of one of the pixels
        # that changed to the new value. We know that these pixels are necessarily
        # in the new_border already because they are next to a pixel that changed.
        new_valued_neighbors_indices = changed_neighborhood_indices[new_valued_neighbors]
        # need to unique-ify indices because neighborhoods overlap and may pick up the same pixels
        new_valued_neighbors_indices = unique_indices(new_valued_neighbors_indices)
        neighbors_of_new_valued_neighbors = self.mask_neighborhood[tuple(new_valued_neighbors_indices.T)]
        no_old_valued_neighbors = ~some_match_old_value(neighbors_of_new_valued_neighbors, axis=(1,2))
        remove_from_new_border_indices = new_valued_neighbors_indices[no_old_valued_neighbors]
        new_border_indices = diff_indices(new_border_indices, remove_from_new_border_indices)

        # (4) Add newly-changed pixels to new_border_indices if they have an old-valued neighbor.
        changed_with_old_neighbor = some_match_old_value(changed_neighborhood, axis=(1,2))
        add_to_new_border_indices = change_indices[changed_with_old_neighbor]
        new_border_indices = numpy.concatenate([new_border_indices, add_to_new_border_indices])
        return old_border_indices, new_border_indices, change_idx

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
            self.move_to_inside(border_mask)

    def erode(self, iters=1, border_mask=None):
        for _ in range(iters):
            if border_mask is None:
                border_mask = numpy.ones(len(self.inside_border_indices), dtype=bool)
            self.move_to_outside(border_mask)

    def smooth(self, iters=1, depth=1):
        """Apply 'iters' iterations of edge-curvature smoothing.
        'depth' controls the spatial scale of the smoothing. With depth=1, only
        the highest-frequency edges get smoothed out. Larger depth values smooth
        lower-frequency structures."""
        for _ in range(iters):
            smoother = next(self._smooth_funcs)
            smoother(depth)

    def _SI(self):
        idx, idx_mask = masked_idx(self.smooth_mask, self.inside_border_indices)
        inside_border = self.mask_neighborhood[idx]
        on_a_line = ((inside_border[:,0,0] & inside_border[:,2,2]) |
                     (inside_border[:,1,0] & inside_border[:,1,2]) |
                     (inside_border[:,0,1] & inside_border[:,2,1]) |
                     (inside_border[:,2,0] & inside_border[:,0,2]))
        self.move_to_outside(unmask_idx(idx_mask, ~on_a_line))

    def _IS(self):
        idx, idx_mask = masked_idx(self.smooth_mask, self.outside_border_indices)
        outside_border = ~self.mask_neighborhood[idx]
        on_a_line = ((outside_border[:,0,0] & outside_border[:,2,2]) |
                     (outside_border[:,1,0] & outside_border[:,1,2]) |
                     (outside_border[:,0,1] & outside_border[:,2,1]) |
                     (outside_border[:,2,0] & outside_border[:,0,2]))
        self.move_to_inside(unmask_idx(idx_mask, ~on_a_line))

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

class BalloonForceMorphology(CurvatureMorphology):
    """Basic morphology operations plus spatially-varying balloon-force operation.
    Base-class to add balloon forces to more complex region-growing steps;
    rarely useful directly.
    """
    def __init__(self, mask, balloon_direction, smooth_mask=None, max_region_mask=None):
        """balloon_direction: (-1, 0, 1), or ndarray with same shape as 'mask'
        containing those values."""
        super().__init__(mask, smooth_mask, max_region_mask)
        if numpy.isscalar(balloon_direction):
            if balloon_direction == 0:
                self.balloon_direction = None
            else:
                self.balloon_direction = numpy.zeros(mask.shape, dtype=numpy.int8)
                self.balloon_direction += balloon_direction
        else:
            self.balloon_direction = balloon_direction.copy() # may get changed internally by subclasses

    def balloon_force(self, iters=1):
        """Apply 'iters' iterations of balloon force region expansion / shrinkage."""
        if self.balloon_direction is None:
            return
        for _ in range(iters):
                to_erode = self.balloon_direction[tuple(self.inside_border_indices.T)] <  0
                self.erode(border_mask=to_erode)
                to_dilate = self.balloon_direction[tuple(self.outside_border_indices.T)] >  0
                self.dilate(border_mask=to_dilate)

class ACWE(BalloonForceMorphology):
    def __init__(self, mask, image, acwe_mask=None, lambda_in=1, lambda_out=1,
        balloon_direction=0, smooth_mask=None, max_region_mask=None):
        """Class for Active Contours Without Edges region-growing.

        Relevant methods for region-growing are smooth(), balloon_force(),
        and acwe_step().

        Parameters:
            mask: mask containing the initial state of the region to evolve
            image: ndarray of same shape as mask containing image values. The
                difference in mean image value inside and outside the region will
                be maximized by acwe_step()
            acwe_mask: region where ACWE updates will be applied. (Inside /
                outside values will still be computed outside this region.)
            lambda_in, lambda_out: weights for comparing means of inside vs.
                outside pixels. Generally 1 works properly.
            balloon_direction: scalar balloon force direction (-1, 0, 1) or
                image map of same values. Generally no balloon forces are needed
                for ACWE.
            smooth_mask: region in which smoothing may be applied.
            max_region_mask: mask beyond which the region may not grow.
        """
        super().__init__(mask, balloon_direction, smooth_mask, max_region_mask)
        # do work in _setup rather than __init__ to allow for complex multiple
        # inheritance from this class that super() alone can't handle. See
        # ActiveContour class.
        self._setup(image, acwe_mask, lambda_in, lambda_out)

    def _setup(self, image, acwe_mask, lambda_in, lambda_out):
        self.image = image
        assert self.image.shape == self.mask.shape
        self.acwe_mask = acwe_mask
        if acwe_mask is not None:
            assert self.image.shape == self.acwe_mask.shape
        self.lambda_in = lambda_in
        self.lambda_out = lambda_out
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

    def move_to_outside(self, to_outside):
        """to_outside must be a boolean mask on the inside border pixels
        (in the order defined by inside_border_indices)."""
        changed_idx = super().move_to_outside(to_outside)
        count, image_sum = self._image_sum_count(changed_idx)
        self.inside_count -= count
        self.outside_count += count
        self.inside_sum -= image_sum
        self.outside_sum += image_sum

    def move_to_inside(self, to_inside):
        """to_inside must be a boolean mask on the outside border pixels
        (in the order defined by outside_border_indices)."""
        changed_idx = super().move_to_inside(to_inside)
        count, image_sum = self._image_sum_count(changed_idx)
        self.outside_count -= count
        self.inside_count += count
        self.outside_sum -= image_sum
        self.inside_sum += image_sum

    def acwe_step(self, iters=1):
        """Apply 'iters' iterations of the Active Contours Without Edges step,
        wherein the region inside the mask is made to have a mean value as different
        from the region outside the mask as possible."""
        for _ in range(iters):
            if self.inside_count == 0 or self.outside_count == 0:
                return
            inside_mean = self.inside_sum / self.inside_count
            outside_mean = self.outside_sum / self.outside_count
            self._acwe_step(inside_mean, outside_mean, self.lambda_in, self.lambda_out,
                self.inside_border_indices, self.move_to_outside)
            self._acwe_step(outside_mean, inside_mean, self.lambda_out, self.lambda_in,
                self.outside_border_indices, self.move_to_inside)

    def _acwe_step(self, mean_from, mean_to, lambda_from, lambda_to, border_indices, move_operation):
        idx, idx_mask = masked_idx(self.acwe_mask, border_indices)
        border_values = self.image[idx]
        to_move = (lambda_from*(border_values - mean_from)**2 >
            lambda_to*(border_values - mean_to)**2)
        move_operation(unmask_idx(idx_mask, to_move))

class BinnedACWE(BalloonForceMorphology):
    def __init__(self, mask, image, acwe_mask=None, lambda_in=1, lambda_out=1,
        balloon_direction=0, smooth_mask=None, max_region_mask=None):
        """Class for Active Contours Without Edges region-growing, using image
        histograms rather than simply means.

        Relevant methods for region-growing are smooth(), balloon_force(),
        and acwe_step().

        Parameters:
            mask: mask containing the initial state of the region to evolve
            image: ndarray of same shape as mask containing image values. The
                difference in image histograms inside and outside the region will
                be maximized by acwe_step()
            acwe_mask: region where ACWE updates will be applied. (Inside /
                outside values will still be computed outside this region.)
            balloon_direction: scalar balloon force direction (-1, 0, 1) or
                image map of same values. Generally no balloon forces are needed
                for ACWE.
            smooth_mask: region in which smoothing may be applied.
            max_region_mask: mask beyond which the region may not grow.
        """
        super().__init__(mask, balloon_direction, smooth_mask, max_region_mask)
        # do work in _setup rather than __init__ to allow for complex multiple
        # inheritance from this class that super() alone can't handle. See
        # ActiveContour class.
        self._setup(image, acwe_mask)

    def _setup(self, image, acwe_mask):
        if image.dtype == bool:
            image = image.astype(numpy.uint8)
        self.image = image
        assert self.image.shape == self.mask.shape
        self.acwe_mask = acwe_mask
        if acwe_mask is not None:
            assert self.image.shape == self.acwe_mask.shape
        # note: self.mask is clipped to self.max_region_mask so the below works.
        self.inside_count = self.mask.sum()
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

    def move_to_outside(self, to_outside):
        """to_outside must be a boolean mask on the inside border pixels
        (in the order defined by inside_border_indices)."""
        changed_idx = super().move_to_outside(to_outside)
        count, bincounts = self._image_bincount_count(changed_idx)
        self.inside_count -= count
        self.outside_count += count
        self.inside_bincounts -= bincounts
        self.outside_bincounts += bincounts

    def move_to_inside(self, to_inside):
        """to_inside must be a boolean mask on the outside border pixels
        (in the order defined by outside_border_indices)."""
        changed_idx = super().move_to_inside(to_inside)
        count, bincounts = self._image_bincount_count(changed_idx)
        self.outside_count -= count
        self.inside_count += count
        self.outside_bincounts -= bincounts
        self.inside_bincounts += bincounts

    def acwe_step(self, iters=1):
        """Apply 'iters' iterations of the Active Contours Without Edges step,
        wherein the region inside the mask is made to have a mean value as different
        from the region outside the mask as possible."""
        for _ in range(iters):
            if self.inside_count == 0 or self.outside_count == 0:
                return
            inside_rate = self.inside_bincounts / self.inside_count
            outside_rate = self.outside_bincounts / self.outside_count
            self._acwe_step(outside_rate > inside_rate, self.inside_border_indices,
                self.move_to_outside)
            self._acwe_step(inside_rate > outside_rate, self.outside_border_indices,
                self.move_to_inside)

    def _acwe_step(self, move_bins, border_indices, move_operation):
        idx, idx_mask = masked_idx(self.acwe_mask, border_indices)
        border_values = self.image[idx]
        to_move = move_bins[border_values]
        move_operation(unmask_idx(idx_mask, to_move))


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
                in at any given point.
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
        super().__init__(mask, balloon_direction, smooth_mask, max_region_mask)
        # do work in _setup rather than __init__ to allow for complex multiple
        # inheritance from this class that super() alone can't handle. See
        # ActiveContour class.
        self._setup(advection_direction, advection_mask)

    def _setup(self, advection_direction, advection_mask):
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
            self._advect(self.inside_border_indices, numpy.greater, self.move_to_outside)
            # Move pixels on the outside border to the inside if advection*gradient sum < 0
            self._advect(self.outside_border_indices, numpy.less, self.move_to_inside)

    def _advect(self, border_indices, criterion, move_operation):
        idx, idx_mask = masked_idx(self.advection_mask, border_indices)
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
        move_operation(unmask_idx(idx_mask, to_move))

class ActiveContour(GAC, ACWE):
    def __init__(self, mask, image, advection_direction, acwe_mask=None,
            advection_mask=None,lambda_in=1, lambda_out=1, balloon_direction=0,
            smooth_mask=None, max_region_mask=None):
        """See documentation for GACMorphology and ACWEMorphology for parameters."""
        BalloonForceMorphology.__init__(self, mask, balloon_direction, smooth_mask, max_region_mask)
        GAC._setup(self, advection_direction, advection_mask)
        ACWE._setup(self, image, acwe_mask, lambda_in, lambda_out)

class BinnedActiveContour(GAC, BinnedACWE):
    def __init__(self, mask, image, advection_direction, acwe_mask=None,
            advection_mask=None, balloon_direction=0,
            smooth_mask=None, max_region_mask=None):
        """See documentation for GACMorphology and ACWEMorphology for parameters."""
        BalloonForceMorphology.__init__(self, mask, balloon_direction, smooth_mask, max_region_mask)
        GAC._setup(self, advection_direction, advection_mask)
        BinnedACWE._setup(self, image, acwe_mask)

def discretize_image(image, n_bins):
    max = image.max()
    min = image.min()
    left_edges = numpy.linspace(min, max, n_bins)
    return numpy.digitize(image, left_edges) - 1

def diff_indices(indices, to_remove):
    assert indices.flags.c_contiguous
    assert to_remove.flags.c_contiguous
    assert indices.dtype == to_remove.dtype
    dtype = numpy.dtype('S'+str(indices.itemsize*2)) # treat (x,y) indices as binary data instead of pairs of ints
    remaining = numpy.setdiff1d(indices.view(dtype), to_remove.view(dtype), assume_unique=True)
    return remaining.view(indices.dtype).reshape((-1, 2))

def unique_indices(indices):
    assert indices.flags.c_contiguous
    dtype = numpy.dtype('S'+str(indices.itemsize*2)) # treat (x,y) indices as binary data instead of pairs of ints
    unique = numpy.unique(indices.view(dtype))
    return unique.view(indices.dtype).reshape((-1, 2))

def not_all(array, axis):
    return ~numpy.all(array, axis)

def masked_idx(mask, indices):
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

def unmask_idx(idx_mask, to_change):
    """Return mask over all indices originally passed to masked_idx, marked
    False where indices were originally masked out or where to_change parameter
    is False.

    Parameters:
        idx_mask: as returned by masked_idx
        to_change: mask over True values in idx_mask
    """
    if idx_mask is None:
        return to_change
    else:
        idx_mask[idx_mask] = to_change
        return idx_mask
