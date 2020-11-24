from setuptools import find_packages, setup


setup(
    name='outlier_hub',
    version='0.0.1',
    packages=find_packages(),
    install_requires=["nltk", "datastack", "pandas", "h5py", "torch", "tqdm", "flair"],
    python_requires=">=3.7"
)
