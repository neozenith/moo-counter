# Third Party
from Cython.Build import cythonize
from setuptools import Extension, setup

# Define both C and Cython extensions
cython_ext = Extension(
    "cython_engine",
    ["cython_engine.pyx"],
)

c_ext = Extension(
    "c_engine",
    ["c_engine.c"],
)

c_full_ext = Extension(
    "c_engine_full",
    ["c_engine_full.c"],
)

setup(
    ext_modules=cythonize(
        [cython_ext],
        compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False, "cdivision": True},
    )
    + [c_ext, c_full_ext],  # Add C extensions separately (don't need cythonize)
    zip_safe=False,
)
