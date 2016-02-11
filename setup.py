import distutils.core
import pathlib
import numpy

try:
    from Cython.Build import cythonize
    ext_processor = cythonize
except ImportError:
    def uncythonize(extensions, **_ignore):
        for extension in extensions:
            sources = []
            for src in map(pathlib.Path, extension.sources):
                if src.suffix == '.pyx':
                    if extension.language == 'c++':
                        ext = '.cpp'
                    else:
                        ext = '.c'
                    src = src.with_suffix(ext)
                sources.append(str(src))
            extension.sources[:] = sources
        return extensions
    ext_processor = uncythonize

sample_texture = distutils.core.Extension('zplib.image._sample_texture',
    sources = ['zplib/image/_sample_texture.pyx'],
    include_dirs = [numpy.get_include()]
)

distutils.core.setup(name='zplib',
    version='1.0',
    description='zplib package',
    ext_modules=ext_processor([sample_texture]),
    packages=['zplib', 'zplib.image', 'zplib.curve', 'zplib.scalar_stats'])
