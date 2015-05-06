import numpy
import pyfftw
import pickle

from . import fft

PRECISION = {
    32: numpy.float32,
    64: numpy.float64
}

PRECISION_FFT = {
    32: numpy.complex64,
    64: numpy.complex128
}

def store_plan_hints(filename):
    """Store data about the best FFT plans for this computer.

    FFT planning can take quite a while. After planning, the knowledge about
    the best plan for a given computer and given transform parameters can be
    written to disk so that the next time, planning can make use of that
    knowledge.
    """
    with open(filename, 'wb') as f:
        pickle.dump(pyfftw.export_wisdom(), f)

def load_plan_hints(filename):
    """Load data about the best FFT plans for this computer.

    FFT planning can take quite a while. After planning, the knowledge about
    the best plan for a given computer and given transform parameters can be
    written to disk so that the next time, planning can make use of that
    knowledge.
    """
    with open(filename, 'rb') as f:
        pyfftw.import_wisdom(pickle.load(f))

class FilterBase:
    def __init__(self, shape, precision=32, threads=4, better_plan=False):
        """Base class for FFT filtering.

        Constructor parameters:
            shape: shape of the images to be filtered (must be 2d)
            precision: 32 or 64. Refers to the floating-point precision with
                which the FFT used for filtering is to be calculated.
            threads: number of threads to use for FFT computation.
            better_plan: if True, spend time (possibly minutes) identifying the
                best FFT plan. This can dramatically speed future FFTs. The
                functions 'store_plan_hints()' and 'load_plan_hints()' can be
                used to store the planning data so that this time-cost need only
                be paid once.
        """
        assert precision in PRECISION.keys()
        n = pyfftw.simd_alignment
        self.image_arr = pyfftw.n_byte_align_empty(shape, n, dtype=PRECISION[precision], order='F')
        fft_shape = list(shape)
        fft_shape[1] = fft_shape[1] // 2 + 1
        self.fft_arr = pyfftw.n_byte_align_empty(fft_shape, n, dtype=PRECISION_FFT[precision], order='F')
        effort = 'FFTW_PATIENT' if better_plan else 'FFTW_MEASURE'
        flags = (effort, 'FFTW_DESTROY_INPUT')
        self.fft = pyfftw.FFTW(self.image_arr, self.fft_arr, axes=(0,1), direction='FFTW_FORWARD', flags=flags, threads=threads)
        self.ifft = pyfftw.FFTW(self.fft_arr, self.image_arr, axes=(0,1), direction='FFTW_BACKWARD', flags=flags, threads=threads)

class SpatialFilter(FilterBase):
    def __init__(self, shape, period_range, spacing=1.0, order=2, keep_dc=False, precision=32, threads=4, better_plan=False):
        """Class to apply the same filter over a multiple images.

        Constructor parameters:
            shape: shape of the images to be filtered (must be 2d)
            period_range: (min_size, max_size) tuple representing the minimum
                and maximum spatial size of objects in the filter's pass-band.
                Objects smaller than min_size and larger than max_size will be
                filtered out of the image. Either value can be None, indicating
                that there should be no size limit at that end of the scale.
            spacing: allows the physical spacing between array elements to be
                specified, so that the size values (above) can be given in
                physical units. This parameter can either be a scalar or a list
                of spacings in each dimension.
            order: controls the order of the butterworth filter. Higher order
                filters remove more of the structure in the stop-bands
                (specifically, those near the transition between pass-band and
                stop-band), but may introduce ringing artifacts.
            keep_dc: if True, then the DC component of the array will not be
                removed; otherwise it will. (If the DC component is removed,
                then the average array value will be zero; otherwise the
                average value will remain unchanged.)
            precision: 32 or 64. Refers to the floating-point precision with
                which the FFT used for filtering is to be calculated.
            threads: number of threads to use for FFT computation.
            better_plan: if True, spend time (possibly minutes) identifying the
                best FFT plan. This can dramatically speed future FFTs. The
                functions 'store_plan_hints()' and 'load_plan_hints()' can be
                used to store the planning data so that this time-cost need only
                be paid once.
        """
        super().__init__(shape, precision, threads, better_plan)
        filter_coeffs = fft.make_spatial_filter(shape, period_range, spacing, order, keep_dc)
        self.filter_coeffs = filter_coeffs.astype(PRECISION[precision], order='F')

    def filter(self, image):
        """Filter a given image as specified in the SpatialFilter constructor.

        Returns: filtered image, of dtype numpy.float32 or numpy.float64,
            depending on the 'precision' parameter to the constructor.
            NB: the returned image WILL be overwritten by future calls to
            filter(). If this image needs to be kept past subsequent filter()
            calls, a copy must be made.
        """
        assert image.shape == self.image_arr.shape
        self.image_arr[:] = image
        self.fft()
        self.fft_arr.real *= self.filter_coeffs
        self.ifft()
        return self.image_arr

class MultiFilter(FilterBase):
    def __init__(self, shape, filters, precision=32, threads=4, better_plan=False):
        """Class to efficiently apply multiple FFT filters to each of many images.

        Constructor parameters:
            shape: shape of the images to be filtered (must be 2d)
            filters: list of filter coefficients generated with, e.g.
                fft.make_spatial_filter()
            precision: 32 or 64. Refers to the floating-point precision with
                which the FFT used for filtering is to be calculated.
            threads: number of threads to use for FFT computation.
            better_plan: if True, spend time (possibly minutes) identifying the
                best FFT plan. This can dramatically speed future FFTs. The
                functions 'store_plan_hints()' and 'load_plan_hints()' can be
                used to store the planning data so that this time-cost need only
                be paid once.
        """
        super().__init__(shape, precision, threads, better_plan)
        self.filters = [fc.astype(PRECISION[precision], order='F') for fc in filters]

    def filter(self, image):
        """Filter a given image with each of the FFT filters, and return a list
        of the filtered images.
        """
        assert image.shape == self.image_arr.shape
        self.image_arr[:] = image
        self.fft()
        fft_vals = self.fft_arr.real.copy()
        images_out = []
        for filter_coeffs in self.filters:
            self.fft_arr.real = fft_vals * filter_coeffs
            self.ifft()
            images_out.append(self.image_arr.copy())
        return images_out

