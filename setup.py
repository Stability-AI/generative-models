from setuptools import find_packages, setup

setup(
    name="sgm",
    version="0.0.1",
    author="Your Name",
    author_email="your_email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "dependency1",
        "dependency2",
    ],
    entry_points={
        "console_scripts": [
            "sgm-cli=sgm.cli:main",
        ],
    },
    package_data={
        "sgm": ["data/*.txt"],
    },
    description="Stability Generative Models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Stability-AI/generative-models",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
