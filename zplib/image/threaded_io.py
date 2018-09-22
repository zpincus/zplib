# This code is licensed under the MIT License (see LICENSE file for details)

from concurrent import futures
import tempfile
import pathlib
import os

import freeimage

class COMPRESSION:
    DEFAULT = 0 # save TIFFs using FreeImage default LZW compression, and PNGs with ZLib level 6 compression

    PNG_NONE = freeimage.IO_FLAGS.PNG_Z_NO_COMPRESSION # save without compression
    PNG_FAST = freeimage.IO_FLAGS.PNG_Z_BEST_SPEED # save using ZLib level 1 compression flag
    PNG_BEST = freeimage.IO_FLAGS.PNG_Z_BEST_COMPRESSION # save using ZLib level 9 compression flag

    TIFF_NONE = freeimage.IO_FLAGS.TIFF_NONE # save without compression


class _ThreadpoolBase:
    def __init__(self, num_threads):
        self.threadpool = futures.ThreadPoolExecutor(num_threads)

    @staticmethod
    def wait_all(futures_out):
        """Wait until all the provided futures have completed; raise an error if
        one or more error out."""
        # wait until all have completed or errored out
        futures.wait(futures_out)
        # now get the result() from each future, which will raise any errors encountered
        # during the execution.
        # The futures.wait() call above makes sure that everything that doesn't
        # error out has a chance to finish before we barf an exception.
        [f.result() for f in futures_out]

    @staticmethod
    def wait_first_error(futures_out):
        """Wait until all the provided futures have completed or the first error
        arises. If an error is raised (or control-c is pressed), cancel the rest
        of the futures and return."""
        try:
            for future in futures.as_completed(futures_out):
                future.result() # if exception occured in the future, this will raise an error
        except:
            # errors were raised above
            for future in futures_out:
                future.cancel()
            raise

class ThreadedIO(_ThreadpoolBase):
    def write(self, images, paths, flags=0):
        """Write out a list of images to the given paths.
        Returns a list of futures representing the jobs, which the user can wait on when desired.
        """
        return [self.threadpool.submit(freeimage.write, image, str(path), flags) for image, path in zip(images, paths)]

    def read(self, paths):
        """Return an iterator over image arrays read from the given paths."""
        paths = map(str, paths)
        return self.threadpool.map(freeimage.read, paths)

class PNG_Compressor(_ThreadpoolBase):
    def __init__(self, level, num_threads):
        super().__init__(num_threads)
        self.level = level

    def compress(self, image_paths):
        """Compress the provided PNG files to the level specified in the constructor.
        Returns a list of futures representing the jobs, which the user can wait on when desired.
        """
        return [self.threadpool.submit(self._compress, image_path) for image_path in image_paths]

    def _compress(self, image_path):
        image_path = pathlib.Path(image_path)
        assert image_path.suffix == '.png'
        image = freeimage.read(image_path)
        temp = tempfile.NamedTemporaryFile(dir=image_path.parent,
            prefix=image_path.stem + 'compressing_', suffix='.png', delete=False)
        try:
            freeimage.write(image, temp.name, flags=self.level)
            os.replace(temp.name, image_path)
        except:
            if os.path.exists(temp.name):
                os.unlink(temp.name)
            raise
        finally:
            temp.close()
