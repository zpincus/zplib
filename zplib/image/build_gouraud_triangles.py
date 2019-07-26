# This code is licensed under the MIT License (see LICENSE file for details)
import cffi
import pathlib

ffibuilder = cffi.FFI()
directory = pathlib.Path(__file__).parent
header = 'gouraud_triangles.h'
ffibuilder.cdef((directory / header).read_text())
ffibuilder.set_source('zplib.image._gouraud_triangles', f'#include "{header}"',
    sources=[str(directory/'gouraud_triangles.c')],
    include_dirs=[str(directory)],
    libraries=['m'])

if __name__ == "__main__":
    ffibuilder.compile()
