import subprocess
import platform
import numpy
import sys
import json

if platform.system() == 'Windows':
    FFMPEG_BIN = 'ffmpeg.exe'
    FFPROBE_BIN = ''
else:
    FFMPEG_BIN = 'ffmpeg'
    FFPROBE_BIN = 'ffprobe'

BYTEORDERS = {'<':'le', '>':'be', '=':'le' if sys.byteorder == 'little' else 'be'}

def read_video(input, force_grayscale=False):
    """Return iterator over frames from an input video via ffmpeg.

    Parameters:
        input: filename to open
        force_grayscale: if True, return uint8 grayscale frames, otherwise
            returns rgb frames.
    """
    ffprobe_command = FFPROBE_BIN + ' -loglevel fatal -select_streams V:0 -show_entries stream=pix_fmt,width,height -print_format json ' + input
    probe = subprocess.run(ffprobe_command.split(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if probe.returncode != 0:
        raise RuntimeError('Could not read video metadata:\n'+probe.stderr.decode())
    metadata = json.loads(probe.stdout.decode())['streams'][0]

    pixel_format_in = metadata['pix_fmt']
    shape = metadata['width'], metadata['height']
    if pixel_format_in.startswith('gray16'):
        pixel_format_out = 'gray16'+BYTEORDERS['='] # output system-native endian
        dtype = numpy.uint16
    elif pixel_format_in == 'gray' or force_grayscale:
        pixel_format_out = 'gray'
        dtype = numpy.uint8
    else:
        pixel_format_out = 'rgb24'
        dtype = numpy.uint8
        shape += (3,)

    command = [FFMPEG_BIN,
        '-loglevel', 'fatal',
        '-nostdin', # do not expect interaction, do not fuss with tty settings
        '-i', input,
        '-map', '0:V:0', # grab video channel 0, just like ffprobe above
        '-f', 'rawvideo', # write raw image data
        '-pix_fmt', pixel_format_out,
        '-' # pipe output to stdout
    ]

    ffmpeg = subprocess.Popen(command, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    dtype = numpy.dtype(dtype)
    size = numpy.product(shape) * dtype.itemsize
    while True:
        frame = ffmpeg.stdout.read(size)
        if len(frame) == 0:
            break
        yield _get_arr(frame, dtype, shape)
    ffmpeg.stdout.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError('Could not read video data:\n'+ffmpeg.stderr.read().decode())

def write_video(frame_iterator, framerate, output, preset=None, lossless=False, verbose=True, **h264_opts):
    """Write uint8 RGB or grayscale image frames to a h264-encoded video file
    using ffmpeg.

    In general, the mp4 container is best here, so the output filename should end
    with '.mp4'. Other suffixes will produce different container files.

    The yuv420p pixel format is used, which is widely compatible with video decoders.
    This pixel format allows (nearly) lossless encoding of grayscale images (with
    +/-1 unit quantization errors for 10-20% of pixels), but not for color images.
    For maximum compatibility, even image dimensions are required as well.

    Parameters:
        frame_iterator: yields numpy arrays. Each frame must be the same size and
            dtype. Currently, only uint8 grayscale and RGB images are supported.
            Images must have even dimensions to ensure maximum compatibility.
        framerate: the frames per second for the output movie.
        output: the file name to write. (Should end with '.mp4'.)
        preset: the ffmpeg preset to use. Defaults to 'medium', or 'ultrafast'
            if lossless compression is specified. The other useful preset for
            lossless is 'veryslow', to get as good compression as possible.
            For lossy compression, 'faster', 'fast', 'medium', 'slow', and 'slower'
            is a reasonable range.
        lossless: if True, use lossless compression. Only valid with grayscale
            images.
        verbose: if True, print status message while compressing.
        **h264_opts: options to be passed to the h264 encoder, from:
           https://www.ffmpeg.org/ffmpeg-codecs.html#Options-23
           'threads', 'tune', and 'profile' may be especially useful.
    """
    frame_iterator = iter(frame_iterator)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        raise ValueError('No image files provided.')

    assert first_frame.dtype == numpy.uint8
    assert first_frame.ndim in (2,3)
    if first_frame.ndim == 3:
        assert first_frame.shape[2] == 3
        pixel_format_in = 'rgb24'
    else:
        pixel_format_in = 'gray'
    assert first_frame.shape[0] % 2 == 0 and first_frame.shape[1] % 2 == 0
    if preset is None:
        if lossless:
            preset = 'ultrafast'
        else:
            preset = 'medium'
    if lossless:
        h264_opts['qp'] = '0'

    _write_video(first_frame, frame_iterator, framerate, output, codec='libx264',
        pixel_format_in=pixel_format_in, pixel_format_out='yuv420p', verbose=verbose,
        preset=preset, **h264_opts)

def write_lossless_video(frame_iterator, framerate, output, threads=None, verbose=True):
    """Write uint8 (color/gray) or uint16 (gray) image frames to a lossless
    FFV1-encoded video file using ffmpeg.

    In general, the mkv container is best here, so the output filename should end
    with '.mkv'. Other suffixes will produce different container files.

    RGB images can only be uint8, but grayscale images can be uint8 or uint16.

    Parameters:
        frame_iterator: yields numpy arrays. Each frame must be the same size and
            dtype.
        framerate: the frames per second for the output movie.
        output: the file name to write. (Should end with '.mkv'.)
        threads: if not None, the number of threads to use and slices to divide
           the image into for compression.
        verbose: if True, print status message while compressing.
    """
    frame_iterator = iter(frame_iterator)
    try:
        first_frame = next(frame_iterator)
    except StopIteration:
        raise ValueError('No image files provided.')

    assert first_frame.dtype in (numpy.uint8, numpy.uint16)
    assert first_frame.ndim in (2,3)
    if first_frame.ndim == 3:
        assert first_frame.shape[2] == 3 and first_frame.dtype == numpy.uint8
        pixel_format_in = 'rgb24'
    elif first_frame.dtype == numpy.uint8:
        pixel_format_in = 'gray'
    else:
        pixel_format_in = 'gray16'+BYTEORDERS[first_frame.dtype.byteorder]

    ffv_args = dict(level='3', g='1', context='1', slicecrc='1')
    if threads is not None:
        threads = str(threads)
        ffv_args['threads'] = threads
        ffv_args['slices'] = threads

    _write_video(first_frame, frame_iterator, framerate, output, codec='ffv1',
        pixel_format_in=pixel_format_in, pixel_format_out=pixel_format_in, verbose=verbose, **ffv_args)


def _write_video(first_frame, frame_iterator, framerate, output, codec, pixel_format_in, pixel_format_out, verbose=False, **codec_opts):
    command = [FFMPEG_BIN,
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-video_size', '{}x{}'.format(*first_frame.shape[:2]), # size of one frame
        '-pix_fmt', pixel_format_in,
        '-framerate', str(framerate), # frames per second
        '-i', '-', # The input comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', codec,
        '-pix_fmt', pixel_format_out,
    ]
    for k, v in codec_opts.items():
        command += ['-'+k, str(v)]
    command.append(output)

    stderr = None if verbose else subprocess.DEVNULL
    ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr)
    ffmpeg.stdin.write(_get_bytes(first_frame))
    try:
        for i, frame in enumerate(frame_iterator):
            if frame.shape != first_frame.shape or frame.dtype != first_frame.dtype:
                raise ValueError('Frame {} has unexpected shape/dtype'.format(i+1))
            ffmpeg.stdin.write(_get_bytes(frame))
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError('ffmpeg encoding failed')


def _get_bytes(image_arr):
    if image_arr.ndim == 2:
        return image_arr.tobytes(order='F')
    else:
        return image_arr.transpose((2,0,1)).tobytes(order='F')

def _get_arr(image_bytes, dtype, shape):
    if len(shape) == 2:
        # could be uint8 or uint16 -- need to account for itemsize
        strides = (dtype.itemsize, shape[0]*dtype.itemsize)
    else:
        # assume uint8 here
        strides = (3, shape[0]*3, 1)
    return numpy.ndarray(shape=shape, buffer=image_bytes, dtype=dtype, strides=strides)
