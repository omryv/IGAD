from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = ROOT / "README.md"

INSTALL_REQUIRES = [
    "numpy>=1.24,<3",
    "scipy>=1.10,<2",
    "scikit-learn>=1.2,<2",
    "matplotlib>=3.7,<4",
]

DEV_REQUIRES = [
    "pytest>=9.0.3,<10",
    "pytest-html>=4.1,<5",
    "pytest-json-report>=1.5,<2",
    "pip-audit>=2.10,<3",
]

setup(
    name="igad",
    version="1.0.0",
    description=(
        "Information-Geometric Anomaly Detection "
        "via Fisher–Rao Scalar Curvature"
    ),
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Omry Damari",
    author_email="omryv@pm.me",
    url="https://github.com/Visigence/IGAD",
    packages=find_packages(exclude=("tests", "tests.*",
                                    "experiments", "experiments.*")),
    python_requires=">=3.10,<3.13",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Source": "https://github.com/Visigence/IGAD",
    },
)
