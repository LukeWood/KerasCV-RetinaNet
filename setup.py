from setuptools import find_packages
from setuptools import setup

VERSION = "0.0.0"

setup(
    name="retina_net",
    packages=find_packages(exclude=("*_test.py",)),
    version=VERSION,
    description="KerasCV RetinaNet model",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/lukewood/KerasCV-retinanet",
    author="Luke Wood",
    author_email="lukewoodcs@gmail.com",
    install_requires=[
        "black",
        "isort",
        "flake8",
        "tensorflow>=2.9.0",
        "absl-py",
        "tensorflow_datasets",
        "ml_collections",
        "opencv-python",
        "pandas",
        "pyyaml",
        "tqdm",
        "contextlib2",
        "matplotlib",
        "click",
        "wandb",
        "tensorflow-metadata",
        "dill",
    ],
)
