"""
Build script for the one Cython extension in this package.

Everything else (metadata, dependencies) lives in pyproject.toml; this file
exists only because the `_cdnmf_fast_cy` coordinate-descent kernel must be
compiled to a native extension. It is cythonized from the .pyx source and
linked against NumPy's headers at *build* time, so the installed package needs
neither Cython nor a C compiler to run.
"""

import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "icqf._cdnmf_fast_cy",
        sources=["icqf/_cdnmf_fast_cy.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
