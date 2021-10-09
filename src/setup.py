from setuptools import find_packages, setup

with open("../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='outlier_hub',
    version='0.0.10',
    author='Max Luebbering, Rajkumar Ramamurthy, Michael Gebauer',
    description="Outlierhub, a collection of machine learning datasets.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=["nltk", "datastack", "pandas", "h5py", "torch", "tqdm", "flair", "torchtext"],
    python_requires=">=3.7"
)
