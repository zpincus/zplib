import numpy

def spatial_filter(array, period_range, spacing=1.0, order=2, keep_dc=False):
    """Filter the given array with a butterworth filter of the specified order.

        Parameters:
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

    To filter multiple images of the same shape with the same filter parameters,
    it will be faster to construct the filter coefficients once with
    make_spatial_filter() and then apply them to each image with filter_nd().
    """
    filter_coeffs = make_spatial_filter(array.shape, period_range, spacing, order, keep_dc)
    return filter_nd(array, filter_coeffs)

def make_spatial_filter(shape, period_range, spacing=1.0, order=2, keep_dc=False):
    """Return an array of butterworth filter coefficients to filter one or more
    images with. See documentation for 'spatial_filter()' for parameter definitions."""
    low_cutoff_period, high_cutoff_period = period_range
    nyquist = 2*spacing
    if high_cutoff_period is not None and low_cutoff_period is not None and high_cutoff_period <= low_cutoff_period:
        raise ValueError('the high cutoff period for a bandpass filter must be larger than the low cutoff period')
    if (high_cutoff_period is not None and high_cutoff_period < nyquist) or (low_cutoff_period is not None and low_cutoff_period < nyquist):
        raise ValueError(f'Period cutoffs must be either None, or values greater than {nyquist} (the Nyquist period).')
    if low_cutoff_period is None and high_cutoff_period is None:
        raise ValueError('At least one cutoff frequency must be sepecified.')
    elif low_cutoff_period is None:
        filter_coeffs = highpass_butterworth_nd(1.0 / high_cutoff_period, shape, spacing, order)
    elif high_cutoff_period is None:
        filter_coeffs = lowpass_butterworth_nd(1.0 / low_cutoff_period, shape, spacing, order)
    else:
        filter_coeffs = bandpass_butterworth_nd(1.0 / high_cutoff_period, 1.0 / low_cutoff_period, shape, spacing, order)
    if keep_dc:
        filter_coeffs.flat[0] = 1
    else:
        filter_coeffs.flat[0] = 0
    return filter_coeffs

def lowpass_butterworth_nd(cutoff, shape, d=1.0, order=2):
    """Create a low-pass butterworth filter with the given pass-band and
    n-dimensional shape. The 'd' parameter is a scalar or list giving the sample
    spacing in all/each dimension, and the 'order' parameter controls the order
    of the butterworth filter."""
    cutoff = float(cutoff)
    nyquist = 1/(2*d)
    if cutoff > nyquist:
        raise ValueError(f'Filter cutoff frequency must be <= {nyquist} (the nyquist frequency for d={d}).')
    return 1.0 / (1.0 + (rfftfreq_nd(shape, d) / cutoff)**(2*order))

def highpass_butterworth_nd(cutoff, shape, d=1.0, order=2):
    """Create a high-pass butterworth filter with the given pass-band and
    n-dimensional shape. The 'd' parameter is a scalar or list giving the sample
    spacing in all/each dimension, and the 'order' parameter controls the order
    of the butterworth filter."""
    return 1 - lowpass_butterworth_nd(cutoff, shape, d, order)

def bandpass_butterworth_nd(low_cutoff, high_cutoff, shape, d=1.0, order=2):
    """Create a band-pass butterworth filter with the given pass-band and
    n-dimensional shape. The 'd' parameter is a scalar or list giving the sample
    spacing in all/each dimension, and the 'order' parameter controls the order
    of the butterworth filter."""
    if low_cutoff >= high_cutoff:
        raise ValueError('Low frequency cutoff must be < high frequency cutoff')
    return lowpass_butterworth_nd(high_cutoff, shape, d, order) * highpass_butterworth_nd(low_cutoff, shape, d, order)

def filter_nd(array, filter_coeffs):
    """Filter an array's fft with a the given filter coefficients."""
    array, filter_coeffs = numpy.asarray(array), numpy.asarray(filter_coeffs)
    fft = numpy.fft.rfftn(array)
    filtered = fft * filter_coeffs
    return numpy.fft.irfftn(filtered)

def rfftfreq_nd(shape, d=1.0):
    """Return an array containing the frequency bins of an N-dimensional real FFT.
    Parameter 'd' specifies the sample spacing."""
    freqs2 = [numpy.fft.fftfreq(n, d)**2 for n in shape[:-1]]
    rfreq2 = numpy.fft.rfftfreq(shape[-1], d)**2
    freqs2 += [rfreq2]
    # freqs2 is a list of the squared real FFT frequencies along each dimension.
    # We now want to add these up in outer-product style to make an nd array
    # containing the sum of sqared frequencies at each point.
    # Note that A length-n array 'a' and a length-m array 'b' can be added to
    # make a shape (n,m) array easily: a[:,numpy.newaxis] + b[numpy.newaxis,:]
    # The below is just an n-dimensional generalization of the above.
    all_slice = slice(None) # equivalent to ':' in a slice-expression
    slices = []
    nd = len(shape)
    for i in range(nd):
        axis_slice = [numpy.newaxis]*nd
        axis_slice[i] = all_slice
        slices.append(axis_slice)
    sum_squared_freqs = sum(f[sl] for f, sl in zip(freqs2, slices))
    return numpy.sqrt(sum_squared_freqs)