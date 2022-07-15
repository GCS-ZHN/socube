# MIT License
#
# Copyright (c) 2022 Zhang.H.N
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup, find_packages

install_requires = [
    "anndata>=0.7.8",
    "matplotlib>=3.5.0",
    "numpy>=1.20.1",
    "pandas>=1.3.4",
    "python-highcharts>=0.4.2",
    "scanpy>=1.8.2",
    "scikit-learn>=1.0.1",
    "scipy>=1.7.3",
    "tables<=3.6.1",
    "lapjv>=1.3.1",
    "torch>=1.8.1",
    "torchvision>=0.9.1",
    "tqdm>=4.62.3",
    "umap-learn>=0.5.2"
]

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    platforms="any",
    install_requires=install_requires,
    entry_points=dict(console_scripts=["socube = socube:main"]),
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires = ">=3.7"
)
