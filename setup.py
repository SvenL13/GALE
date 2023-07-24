import os
import sys

from setuptools import setup

DESCRIPTION = "A Python Library for Global Adaptive Learning"
DISTNAME = "gale"
AUTHOR = "Sven Laemmle"
AUTHOR_EMAIL = "laemmle.sven@googlemail.com"
LICENSE = ""
DOWNLOAD_URL = ""
VERSION = "0.1"
PYTHON_REQUIRES = ">=3.9"
PLATFORMS = ["any"]

CLASSIFIERS = ["Intended Audience :: Science/Research",
               "Intended Audience :: Developers",
               "Programming Language :: Python",
               "Topic :: Software Development",
               "Topic :: Scientific/Engineering",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: POSIX",
               "Operating System :: Unix",
               "Operating System :: MacOS",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: 3.9",
               "License :: OSI Approved :: BSD License",
               ]

base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "requirements.txt"), "r") as f:
    INSTALL_REQUIRES = f.readlines()

with open(os.path.join(base_dir, "README.md"), "r") as f:
    LONG_DESCRIPTION = f.read()

with open(os.path.join(base_dir, "LICENSE"), "r") as f:
    LICENSE_FILE = f.read()

LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
PACKAGES = ["gale",
            "gale.doe",
            "gale.doe.acquisition",
            "gale.experiments",
            "gale.experiments.config",
            "gale.models",
            "gale.models.dmbrl",
            ]

if __name__ == "__main__":

    if sys.version_info[:2] < (3, 9):
        raise RuntimeError("Installation requires python >= 3.9")

    setup(
        name=DISTNAME,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        license=LICENSE,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        classifiers=CLASSIFIERS,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        platforms=PLATFORMS,
        packages=PACKAGES,
    )
