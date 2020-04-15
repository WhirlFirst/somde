from setuptools import setup, find_packages

__version__ = "0.1.0"

setup(
    name="somde",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'somoclu',
        'SpatialDE'
        ],
    scripts=[
        "som.py",
        "util.py"
        ],
    author="Minsheng Hao",
    author_email="hmsh653@gmail.com",
    keywords=["spatial transcriptomics", "SpatialDE", "bioinformatics", "self organizing map(SOM)"],
    description="Algorithm for finding gene spatial pattern based on Gaussian process accelerated by SOM",
    license="MIT",
    py_modules=["som"],
)