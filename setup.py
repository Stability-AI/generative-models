from setuptools import find_packages, setup

setup(
    name="sgm",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.8",
    py_modules=["sgm"],
    description="Stability Generative Models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Stability-AI/generative-models",
)
