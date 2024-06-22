import os

from setuptools import find_packages, setup

requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
with open(requirements_path, encoding='utf-8') as f:
    requirements = f.read().splitlines()

# Delete the actual version of the package
requirements = [req.split('==')[0] for req in requirements]

setup(
    name="donoratlas",
    version="1.0",
    packages=find_packages(),
    description="Helpers for DonorAtlas",
    author="DonorAtlas",
    install_requires=requirements,
)
