"""
Setup configuration for NeuroPredict package.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").splitlines()
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]
else:
    requirements = []

setup(
    name="neuropredict",
    version="0.1.0",
    author="Seu Nome",
    author_email="seu.email@exemplo.com",
    description="Sistema de Medicina de Precisão para Epilepsia Refratária",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/neuropredict",
    project_urls={
        "Bug Tracker": "https://github.com/seu-usuario/neuropredict/issues",
        "Documentation": "https://neuropredict.readthedocs.io",
        "Source Code": "https://github.com/seu-usuario/neuropredict",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuropredict=neuropredict.cli:main",
        ],
    },
)