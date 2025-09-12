from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="quantitative-derivatives-engine",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced derivatives pricing engine with multiple models and automatic differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantitative-derivatives-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=3.5.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    keywords="finance, derivatives, options, quantitative, black-scholes, heston, monte-carlo, risk-management",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/quantitative-derivatives-engine/issues",
        "Source": "https://github.com/yourusername/quantitative-derivatives-engine",
        "Documentation": "https://quantitative-derivatives-engine.readthedocs.io/",
    },
)