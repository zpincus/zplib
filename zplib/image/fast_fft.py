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


class SpatialFilter:
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
        assert precision in PRECISION.keys()
        n = pyfftw.simd_alignment
        self.filter_coeffs = fft.make_spatial_filter(shape, period_range, spacing, order, keep_dc)
        self.image_arr = pyfftw.n_byte_align_empty(shape, n, dtype=PRECISION[precision], order='F')
        self.fft_arr = pyfftw.n_byte_align_empty(self.filter_coeffs.shape, n, dtype=PRECISION_FFT[precision], order='F')
        effort = 'FFTW_PATIENT' if better_plan else 'FFTW_MEASURE'
        flags = (effort, 'FFTW_DESTROY_INPUT')
        self.fft = pyfftw.FFTW(self.image_arr, self.fft_arr, axes=(0,1), direction='FFTW_FORWARD', flags=flags, threads=threads)
        self.ifft = pyfftw.FFTW(self.fft_arr, self.image_arr, axes=(0,1), direction='FFTW_BACKWARD', flags=flags, threads=threads)

    def filter(self, image):
        """Filter a given image as specified in the SpatialFilter constructor.

        Returns: filtered image, of dtype numpy.float32 or numpy.float64,
            depending on the 'precision' parameter to the constructor.
            NB: the returned image WILL be overwritten by future calls to
            filter(). If this image needs to be kept past subsequent filter()
            calls, a copy must be made.
        """
        assert image.shape == self.image_arr.shape
        self.image_arr[:] =  image
        self.fft()
        self.fft_arr *= self.filter_coeffs
        self.ifft()
        return self.image_arr