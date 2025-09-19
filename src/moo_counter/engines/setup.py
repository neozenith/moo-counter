from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "cython_engine",
        ["cython_engine.pyx"],
    )
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True
        }
    ),
    zip_safe=False
)