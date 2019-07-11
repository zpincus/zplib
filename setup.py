import setuptools
import numpy

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

extensions = [setuptools.Extension('zplib.image._sample_texture',
                sources = ['zplib/image/_sample_texture.' + ('pyx' if HAS_CYTHON else 'c')],
                include_dirs = [numpy.get_include()])
]

if HAS_CYTHON:
    extensions = cythonize(extensions)

setuptools.setup(
    name = 'zplib',
    version = '1.5',
    description = 'basic zplab tools',
    packages = setuptools.find_packages(),
    package_data = {'zplab.image': ['_label_colors.npy']},
    ext_modules = extensions
)
