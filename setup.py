from setuptools import setup, find_packages

__version__ = "0.1.6"

with open("somde/README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="somde",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.17.4',
        'pandas>=0.25.3',
        'matplotlib>=3.1.1',
        'somoclu>=1.7.5',
        'patsy'
        ],
    author="Minsheng Hao",
    author_email="hmsh653@gmail.com",
    keywords=["spatial transcriptomics", "SpatialDE", "bioinformatics", "self organizing map(SOM)"],
    description="Algorithm for finding gene spatial pattern based on Gaussian process accelerated by SOM",
    license="MIT",
    url='https://github.com/WhirlFirst/somde',
    long_description_content_type='text/markdown',
    long_description=long_description
)