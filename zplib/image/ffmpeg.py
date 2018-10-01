import subprocess
import platform
import numpy
import sys
import json

if platform.system() == 'Windows':
    FFMPEG_BIN = 'ffmpeg.exe'
    FFPROBE_BIN = 'ffprobe.exe'
else:
    FFMPEG_BIN = 'ffmpeg'
    FFPROBE_BIN = 'ffprobe'

BYTEORDERS = {'<': 'le', '>': 'be', '=': 'le' if sys.byteorder == 'little' else 'be'}

def read_video(input, force_grayscale=False):
    """Return iterator over frames from an input video via ffmpeg.

    Parameters:
        input: filename to open
        force_grayscale: if True, return uint8 grayscale frames, otherwise
            returns rgb frames.
    """
    ffprobe_command = [FFPROBE_BIN,
        '-loglevel', 'fatal',
        '-select_streams', 'V:0',
        '-show_entries', 'stream=pix_fmt,width,height',
        '-print_format', 'json',
        input]
    probe = subprocess.run(ffprobe_command, stdin=subprocess.DEVNULL,
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

    if len(shape) == 2:
        # could be uint8 or uint16 -- need to account for itemsize
        strides = (dtype.itemsize, shape[0]*dtype.itemsize)
    else:
        # assume uint8 here
        strides = (3, shape[0]*3, 1)

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
        frame_bytes = ffmpeg.stdout.read(size)
        if len(frame_bytes) == 0:
            break
        yield numpy.ndarray(shape=shape, buffer=frame_bytes, dtype=dtype, strides=strides)
    ffmpeg.stdout.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError('Could not read video data:\n'+ffmpeg.stderr.read().decode())

def write_video(frames, framerate, output, preset=None, lossless=False, verbose=True, **h264_opts):
    """Write uint8 RGB or grayscale image frames to a h264-encoded video file
    using ffmpeg.

    In general, the mp4 container is best here, so the output filename should end
    with '.mp4'. Other suffixes will produce different container files.

    The yuv420p pixel format is used, which is widely compatible with video decoders.
    This pixel format allows (nearly) lossless encoding of grayscale images (with
    +/-1 unit quantization errors for 10-20% of pixels), but not for color images.
    For maximum compatibility, even image dimensions are required as well.

    Parameters:
        frames: yields numpy arrays. Each frame must be the same size and
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
    with VideoWriter(framerate, output, preset, lossless, verbose, **h264_opts) as writer:
        for frame in frames:
            writer.encode_frame(frame)

def write_lossless_video(frames, framerate, output, threads=None, verbose=True):
    """Write uint8 (color/gray) or uint16 (gray) image frames to a lossless
    FFV1-encoded video file using ffmpeg.

    In general, the mkv container is best here, so the output filename should end
    with '.mkv'. Other suffixes will produce different container files.

    RGB images can only be uint8, but grayscale images can be uint8 or uint16.

    Parameters:
        frames: yields numpy arrays. Each frame must be the same size and
            dtype.
        framerate: the frames per second for the output movie.
        output: the file name to write. (Should end with '.mkv'.)
        threads: if not None, the number of threads to use and slices to divide
           the image into for compression.
        verbose: if True, print status message while compressing.
    """
    with LosslessVideoWriter(framerate, output, threads, verbose) as writer:
        for frame in frames:
            writer.encode_frame(frame)

class _VideoWriter:
    def __init__(self, framerate, output, verbose, codec_options, codec, pixel_format_out):
        self.ffmpeg = None
        self.framerate = framerate
        self.output = output
        self.verbose = verbose
        self.codec_options = codec_options
        self.codec = codec
        self.pixel_format_out = pixel_format_out

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def encode_frame(self, frame):
        if self.ffmpeg is None:
            self._init_ffmpeg(frame)
        elif frame.shape != self.shape or frame.dtype != self.dtype:
            raise ValueError(f'Expected shape {self.shape}, dtype {self.dtype}, got {frame.shape} and {frame.dtype}')
        if frame.ndim == 2:
            frame_bytes = frame.tobytes(order='F')
        else:
            frame_bytes = frame.transpose((2,0,1)).tobytes(order='F')
        self.ffmpeg.stdin.write(frame_bytes)

    def _init_ffmpeg(self, frame):
        assert frame.ndim in (2, 3)
        assert frame.dtype in (numpy.uint8, numpy.uint16)
        if frame.dtype == numpy.uint8:
            if frame.ndim == 3:
                assert frame.shape[2] == 3
                pixel_format_in = 'rgb24'
            else:
                pixel_format_in = 'gray'
        elif frame.dtype == numpy.uint16:
            if frame.ndim == 3:
                raise ValueError('Cannot encode RGB uint16 movies.')
            pixel_format_in = 'gray16' + BYTEORDERS[frame.dtype.byteorder]
        if self.pixel_format_out is None:
            self.pixel_format_out = pixel_format_in
        command = [FFMPEG_BIN,
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-video_size', f'{frame.shape[0]}x{frame.shape[1]}', # size of one frame
            '-pix_fmt', pixel_format_in,
            '-framerate', str(self.framerate), # frames per second
            '-i', '-', # The input comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-vcodec', self.codec,
            '-pix_fmt', self.pixel_format_out]
        for option, value in self.codec_options.items():
            command += ['-'+option, str(value)]
        command.append(self.output)
        stderr = None if self.verbose else subprocess.DEVNULL
        self.shape = frame.shape
        self.dtype = frame.dtype
        self.ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr)

    def close(self):
        if self.ffmpeg is not None:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            if self.ffmpeg.returncode != 0:
                raise RuntimeError('ffmpeg encoding failed')

class VideoWriter(_VideoWriter):
    def __init__(self, framerate, output, preset=None, lossless=False, verbose=True, **h264_options):
        """Stream uint8 RGB or grayscale image frames to a h264-encoded video file
        using ffmpeg.

        This function can be used to send new frames to ffmpeg as they are produced,
        unlike write_video, which requires an iterator of all the existing frames.

        To use, create a and then use the encode_frame() method to provide new frames.
        When finished, call the close() method. Alternately, use as a context manager
        and close will be called automatically.

        writer = VideoWriter(...)
        writer.encode_frame(frame1)
        writer.encode_frame(frame2)
        ...
        writer.close()

        or, better, as a context manager:
        with VideoWriter(...) as writer:
            writer.encode_frame(frame1)
            writer.encode_frame(frame2)
            ...

        Each frame must be the same size and dtype. Currently, only uint8 grayscale
        and RGB images are supported. Images must have even dimensions to ensure
        maximum compatibility.

        In general, the mp4 container is best here, so the output filename should end
        with '.mp4'. Other suffixes will produce different container files.

        The yuv420p pixel format is used, which is widely compatible with video decoders.
        This pixel format allows (nearly) lossless encoding of grayscale images (with
        +/-1 unit quantization errors for 10-20% of pixels), but not for color images.
        For maximum compatibility, even image dimensions are required as well.

        Parameters:
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
        if preset is None:
            if lossless:
                preset = 'ultrafast'
            else:
                preset = 'medium'
        h264_options['preset'] = preset
        if lossless:
            h264_options['qp'] = '0'
        super().__init__(framerate, output, verbose, h264_options,
            codec='libx264', pixel_format_out='yuv420p')

    def _init_ffmpeg(self, frame):
        assert frame.dtype == numpy.uint8
        assert frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0
        super()._init_ffmpeg(frame)

class LosslessVideoWriter(_VideoWriter):
    def __init__(self, framerate, output, threads=None, verbose=True):
        """Write uint8 (color/gray) or uint16 (gray) image frames to a lossless
        FFV1-encoded video file using ffmpeg.

        To use, create a and then use the encode_frame() method to provide new frames.
        When finished, call the close() method. Alternately, use as a context manager
        and close will be called automatically.

        writer = LosslessVideoWriter(...)
        writer.encode_frame(frame1)
        writer.encode_frame(frame2)
        ...
        writer.close()

        or, better, as a context manager:
        with LosslessVideoWriter(...) as writer:
            writer.encode_frame(frame1)
            writer.encode_frame(frame2)
            ...

        In general, the mkv container is best here, so the output filename should end
        with '.mkv'. Other suffixes will produce different container files.

        RGB images can only be uint8, but grayscale images can be uint8 or uint16.

        Parameters:
            framerate: the frames per second for the output movie.
            output: the file name to write. (Should end with '.mkv'.)
            threads: if not None, the number of threads to use and slices to divide
               the image into for compression.
            verbose: if True, print status message while compressing.
        """
        ffv_options = dict(level='3', g='1', context='1', slicecrc='1')
        if threads is not None:
            threads = str(threads)
            ffv_options['threads'] = threads
            ffv_options['slices'] = threads
        super().__init__(framerate, output, verbose, ffv_options,
            codec='ffv1', pixel_format_out=None)