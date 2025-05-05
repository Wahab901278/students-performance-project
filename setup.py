from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(filepath):
    requirements=[]
    with open(filepath) as fileobj:
        requirements=fileobj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='students-performance-project',
    version='0.0.1',
    author='Abdul Wahab',
    author_email='abdulwahab901278@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)