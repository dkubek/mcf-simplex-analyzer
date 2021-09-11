from pathlib import Path

import numpy as np
import gmpy2

from setuptools import Distribution, Extension

from Cython.Build import cythonize
from Cython.Distutils.build_ext import new_build_ext as cython_build_ext

GMPY2_INCLUDE_DIR = str(Path(gmpy2.__file__).parent)

compile_args = [
    "-march=native",
    "-O3",
]
link_args = []
include_dirs = [np.get_include(), GMPY2_INCLUDE_DIR]
libraries = ["gmp", "mpfr", "mpc"]


def build(setup_kwargs):
    extensions = [
        Extension(
            "*",
            ["src/**/*.pyx"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            include_dirs=include_dirs,
            libraries=libraries,
        )
    ]
    ext_modules = cythonize(
        extensions,
        include_path=include_dirs,
        compiler_directives={"binding": True, "language_level": 3},
    )

    distribution = Distribution(
        {
            "name": "cython-test",
            "src_root": "src/",
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": cython_build_ext},
        }
    )

    # Grab the build_ext command and copy all files back to source dir.
    # Done so Poetry grabs the files during the next step in its build.
    distribution.run_command("build_ext")
    build_ext_cmd = distribution.get_command_obj("build_ext")
    build_ext_cmd.copy_extensions_to_source()

    return setup_kwargs


if __name__ == "__main__":
    build({})
