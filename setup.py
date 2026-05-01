from setuptools import find_packages, setup

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
]

setup(
    name="igad",
    version="0.1.0",
    description="Information-Geometric Anomaly Detection via Fisher-Rao Scalar Curvature",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Omry Damari",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Source": "https://github.com/Visigence/IGAD",
        "Security": "https://github.com/Visigence/IGAD/security/policy",
    },
)