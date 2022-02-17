from setuptools import setup


setup(
    name="clif",
    version="0.1",
    description="CLImate Fingerprinting methods",
    long_description="",
    url="",
    author=["Kenny Chowdhary", "Jake Nichol"],
    author_email="kchowdh@sandia.gov",
    license="BSD3",
    packages=["clif"],
    test_suite="nose.collector",
    tests_required=["nose"],
    install_requires=["numpy", "sklearn", "tqdm", "xarray", "netCDF4", "statsmodels"],
    include_package_data=True,
    zip_safe=False,
)
