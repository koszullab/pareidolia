import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())

with open("pareidolia/__init__.py", "r") as init:
    init_conts = init.read()
    vers_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    try:
        VERSION = re.search(vers_regex, init_conts, re.MULTILINE)[1]
    except TypeError:
        raise RuntimeError(f"Cannot find version string in __init__.py")

setup(
    name="pareidolia",
    version=VERSION,
    url="https://github.com/cmdoret/pareidolia",
    license='MIT',
    author="Cyril Matthey-Doret",
    author_email="cyril.matthey-doret@pasteur.fr",
    description="Multi-sample change detection in Hi-C patterns",
    long_description=read("README.rst"),
    packages=find_packages(exclude=('tests',)),
    install_requires=read('requirements.txt').splitlines(),
    entry_points={"console_scripts": ["pareidolia=pareidolia.cli:pareidolia_cmd"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
    ],
)
