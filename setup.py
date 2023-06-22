from setuptools import find_packages, setup
import os

# get Pytorch version
PT_VERSION = os.environ.get('PT_VERSION', None)
if PT_VERSION is None:
    raise ValueError('Set the environment variable `PT_VERSION` to the version of PyTorch you are using: export PT_VERSION="pt2" or "pt13"')
else:
    # load corresponding requirements
    path_requirements = f"requirements_{PT_VERSION}.txt"
    with open(path_requirements) as f:
        install_requires = f.read().splitlines()

setup(
    name="ldm",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.9",
    py_modules=["ldm"],
    description="Stability Generative Models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Stability-AI/generative-models/tree/main",
    license="",
    install_requires=install_requires,
    dependency_links=[
        "https://download.pytorch.org/whl/cu117"
    ]
)