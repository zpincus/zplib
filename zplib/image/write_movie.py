import numpy

import freeimage

from . import ffmpeg
from . import colorize

def write_movie(image_generator, output_file, framerate=15):
    """Write a movie from images.

    Parameters:
        image_generator: a generator that will produce images for movie frames.
        output_file: filename to write movie file. Must end with '.mp4'
        framerate: playback speed of the movie in frames per second

    Examples:
        1) Load all PNGs from a directory and write to a movie:
            image_dir = pathlib.Path('/path/to/images')
            image_files = sorted(image_dir.glob('*.png'))
            image_generator = generate_images_from_files(image_files)
            write_movie(image_generator, 'movie.mp4', framerate=10)

        2) Load, crop, and scale brightfield and GFP images, and tile side by side:
            image_dir = pathlib.Path('/path/to/images')
            upper_left = [10, 20]
            lower_right = [600, 750]
            bf_files = sorted(image_dir.glob('bf*.png'))
            bf_generator = generate_images_from_files(bf_files, upper_left, lower_right, min=0, max=12560)
            gfp_files = sorted(image_dir.glob('gfp*.png'))
            gfp_generator = generate_images_from_files(gfp_files, upper_left, lower_right, min=350, max=1020, gamma=0.7)
            image_generator = tile_images(bf_generator, gfp_generator, axis='x')
            write_movie(image_generator, 'movie.mp4', framerate=10)

        3) Load, crop, scale, and colorize brightfield and GFP images, and combine on top of each other with "screen" blending:
            image_dir = pathlib.Path('/path/to/images')
            upper_left = [10, 20]
            lower_right = [600, 750]
            bf_files = sorted(image_dir.glob('bf*.png'))
            bf_generator = generate_images_from_files(bf_files, upper_left, lower_right, min=0, max=12560)
            gfp_files = sorted(image_dir.glob('gfp*.png'))
            gfp_generator = generate_images_from_files(gfp_files, upper_left, lower_right, min=350, max=1020, gamma=0.7, color=(0,255,0))
            image_generator = screen_images(bf_generator, gfp_generator)
            write_movie(image_generator, 'movie.mp4', framerate=10)
    """
    output_file = str(output_file)
    if not output_file.endswith('.mp4'):
        raise ValueError('Output filename should end with ".mp4".')
    ffmpeg.write_video(image_generator, framerate, output_file)

def generate_images_from_files(image_files, upper_left=(0,0), lower_right=(None,None), min=None, max=None, gamma=1, color=(255,255,255)):
    """Load images from a file, optionally cropping, scaling, and colorizing them
    before converting to 8 bit. Each image is yielded as it is loaded and transformed.

    Parameters:
        image_files: list of image files to read
        upper_left: (x, y) coordinates of upper left corner of region to include
        lower_right: (x, y) coordinates of lower right corner of region to include
            (use None to use full image extent).
        min, max: image intensity values to map to black and white, respectively.
            If None, use each image's min and max value.
        gamma: gamma value for intensity transformation.
        color: 8-bit (R,G,B) color to map "white" to. By default this is white:
            (255, 255, 255). To map the brightest possible color to green instead
            use (0, 255, 0), for example.
    """
    x1, y1 = upper_left
    x2, y2 = lower_right
    for image_file in image_files:
        image = freeimage.read(image_file)
        cropped_image = image[x1:x2, y1:y2]
        scaled_image = colorize.scale(cropped_image, min, max, gamma, output_max=1)
        colorized_8bit_image = colorize.color_tint(scaled_image, color).astype(numpy.uint8)
        yield colorized_8bit_image

def tile_images(*image_generators, axis='x'):
    """Tile a set of images produced by two or more generators along one direction
    and yield the tiled image.

    Parameters:
        axis: either 'x' or 'y' to indicate the direction the images should be tiled in.
    """
    axes = {'x':0, 'y':1}
    if axis not in axes:
         raise ValueError('Axis for laying out images side by side must be "x" or "y".')
    for images in zip(*image_generators):
        yield numpy.concatenate(images, axis=axes[axis])

def screen_images(*image_generators):
    """Combine images produced by two or more generators by using 'screen'
    blending to stack them on top of eachother. This is useful with one brightfield
    image as the first (bottom) layer, and then colorized fluorescence images atop."""
    for images in zip(*image_generators):
        yield colorize.multi_screen(images).astype(numpy.uint8)

