from setuptools import setup, Extension
from Cython.Build import cythonize

# Define both C and Cython extensions
cython_ext = Extension(
    "cython_engine",
    ["cython_engine.pyx"],
)

c_ext = Extension(
    "c_engine",
    ["c_engine.c"],
)

setup(
    ext_modules=cythonize(
        [cython_ext],
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True
        }
    ) + [c_ext],  # Add C extension separately (doesn't need cythonize)
    zip_safe=False
)