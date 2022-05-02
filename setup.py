from setuptools import setup
from setuptools import find_packages

PACKAGES = find_packages(where=".")

setup(
    name="clif",
    version="0.2.2",
    description="CLImate Fingerprinting methods",
    long_description="",
    url="",
    author=["Kenny Chowdhary", "Jake Nichol"],
    author_email="kchowdh@sandia.gov",
    license="BSD3",
    packages=PACKAGES,
    package_data={"clif": ["folder/*.txt"]},
    test_suite="nose.collector",
    tests_required=["nose"],
    install_requires=[
        "numpy",
        "sklearn",
        "tqdm",
        "xarray",
        "netCDF4",
        "statsmodels",
        "cftime",
        "matplotlib",
        "dask",
    ],
    include_package_data=True,
    zip_safe=False,
)
