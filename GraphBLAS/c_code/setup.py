from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name='container',
    ext_modules=[
        Extension('container',
            sources=['container.pyx'],
            include_dirs=['.'],
            include_path=['./graphblas/'],
            extra_compile_args=[
                '-std=c++11',
                '-O3',
            ],
            language='c++')
    ],
    cmdclass={'build_ext': build_ext}
)
