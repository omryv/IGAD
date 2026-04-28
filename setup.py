from setuptools import setup, find_packages

setup(
    name="igad",
    version="0.1.0",
    description="Information-Geometric Anomaly Detection via Fisher-Rao Scalar Curvature",
    author="Omry Damari",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7",
            "pytest-html",
            "pytest-json-report",
        ],
    },
)
