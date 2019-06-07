# This code is licensed under the MIT License (see LICENSE file for details)

from concurrent import futures
import tempfile
import pathlib
import os
import itertools

import freeimage

class COMPRESSION:
    DEFAULT = 0 # save TIFFs using FreeImage default LZW compression, and PNGs with ZLib level 6 compression

    PNG_NONE = freeimage.IO_FLAGS.PNG_Z_NO_COMPRESSION # save without compression
    PNG_FAST = freeimage.IO_FLAGS.PNG_Z_BEST_SPEED # save using ZLib level 1 compression flag
    PNG_BEST = freeimage.IO_FLAGS.PNG_Z_BEST_COMPRESSION # save using ZLib level 9 compression flag

    TIFF_NONE = freeimage.IO_FLAGS.TIFF_NONE # save without compression


class _ThreadpoolBase:
    def __init__(self, num_threads, max_queued_jobs=None):
        self.threadpool = futures.ThreadPoolExecutor(num_threads)
        self.futures = set()
        self.done = set()
        self.max_queued_jobs = max_queued_jobs

    def submit(self, fn, *args, **kws):
        if self.max_queued_jobs is not None and len(self.futures) >= self.max_queued_jobs:
            self._wait(1)
        future = self.threadpool.submit(fn, *args, **kws)
        self.futures.add(future)
        return future

    def _wait(self, n):
        iterator = futures.as_completed(self.futures)
        for i in range(n): # iterate through the first n futures to finish
            next(iterator)
        done, self.futures = futures.wait(self.futures, timeout=0)
        self.done.update(done)

    def wait(self, n=None):
        """Wait until the futures have completed, raising an error if one or
        more error out.

        Parameters:
            n: number of futures to wait for, or None to wait for all.
        """
        if n is None:
            n = len(self.futures)
        if n == 0:
            return
        assert n > 0
        self._wait(n)
        # now get the result() from each completed future, which will raise any
        # errors encountered during the execution.
        try:
            for future in self.done:
                future.result()
        finally:
            self.done.clear()

    def wait_first_error(self):
        """Wait until all the provided futures have completed or the first error
        arises. If an error is raised (or control-c is pressed), cancel the rest
        of the futures and return."""
        all_futures = self.futures.union(self.done)
        try:
            for future in futures.as_completed(all_futures):
                future.result() # if exception occured in the future, this will raise an error
        except:
            # errors were raised above
            for future in self.futures:
                future.cancel()
            raise
        finally:
            self.futures.clear()
            self.done.clear()

class ThreadedIO(_ThreadpoolBase):
    def write(self, images, paths, flags=0):
        """Write out a list of images to the given paths.
        Returns a list of futures representing the jobs, which the user can wait on when desired.
        """
        return [self.submit(freeimage.write, image, str(path), flags) for image, path in zip(images, paths)]

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
        return [self.submit(self._compress, image_path) for image_path in image_paths]

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
