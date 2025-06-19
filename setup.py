# setup.py
from setuptools import find_packages, setup

setup(
    name="hitters-ml-project",
    version="0.1.0",
    description="Baseball salary prediction with MLOps best practices",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=0.11.0",
        "wandb>=0.15.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "ISLP>=0.3.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A PyTorch project for predicting baseball salaries",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
