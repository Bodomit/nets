import distutils
from distutils.core import setup
import setuptools
import subprocess

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nets',
    version='0.1',
    packages=['nets'],
    license='GNU Affero General Public License v3.0',
    long_description=open('README.md').read(),
    include_package_data=True,
    install_requires=requirements
)
