import subprocess
import platform
import numpy

if platform.system() == 'Windows':
    FFMPEG_BIN = "ffmpeg.exe"
else:
    FFMPEG_BIN = "ffmpeg"

def write_video(frame_iterator, framerate, output, preset=None, lossless=False, verbose=True, **h264_opts):
    """Write image frames from an iterator to a h264-encoded video file using ffmpeg.

    Parameters:
        frame_iterator: yields numpy arrays. Each frame must be the same size and
            dtype. Currently, only uint8 greyscale and RGB images are supported.
            Images must have even dimensions to ensure maximum compatibility.
        framerate: the frames per second for the output movie.
        output: the file name to write. (Should end with '.mp4'.)
        preset: the ffmpeg preset to use. Defaults to 'medium', or 'ultrafast'
            if lossless compression is specified. The other useful preset for
            lossless is 'veryslow', to get as good compression as possible.
            For lossy compression, 'faster', 'fast', 'medium', 'slow', and 'slower'
            is a reasonable range.
        lossless: if True, use lossless compression.
        verbose: if True, print status message while compressing.
        **h264_opts: options to be passed to the h264 encoder, from:
           https://www.ffmpeg.org/ffmpeg-codecs.html#Options-23
           'threads', 'tune', and 'profile' may be especially useful.
    """
    frame_iterator = iter(frame_iterator)
    first_frame = next(frame_iterator)
    assert first_frame.dtype == numpy.uint8
    assert first_frame.ndim in (2,3)
    if first_frame.ndim == 3:
        assert first_frame.shape[2] == 3
        pixel_format = 'rgb24'
    else:
        pixel_format = 'gray'
    assert first_frame.shape[0] % 2 == 0 and first_frame.shape[1] % 2 == 0
    if preset is None:
        if lossless:
            preset = 'ultrafast'
        else:
            preset = 'medium'
    command = [FFMPEG,
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-video_size', '{}x{}'.format(*first_frame.shape[:2]), # size of one frame
        '-pixel_format', pixel_format,
        '-framerate', str(framerate), # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', preset
    ]
    if lossless:
        command += ['-qp', '0']
    for k, v in h264_opts.items():
        command += ['-'+k, str(v)]
    command.append(output)

    stderr = None if verbose else subprocess.DEVNULL
    ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr)
    ffmpeg.stdin.write(_get_bytes(first_frame))
    for i, frame in enumerate(frame_iterator):
        if frame.shape != first_frame.shape or frame.dtype != first_frame.dtype:
            raise ValueError('Frame {} has unexpected shape/dtype'.format(i+1))
        ffmpeg.stdin.write(_get_bytes(frame))

    ffmpeg.stdin.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError('ffmpeg encoding failed')

def _get_bytes(image_arr):
    if image_arr.ndim == 2:
        return image_arr.tobytes(order='F')
    else:
        return image_arr.transpose((2,0,1)).tobytes(order='F')
