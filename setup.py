# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

install_requires = [
    'jax',
    'jax-md',
    'jax-sgmc',
    'optax',
    'dm-haiku',
    'sympy',
    'tree_math',
    'cloudpickle',
    'chex',
    'blackjax==0.3.0',
    'jaxopt',
]

extras_requires = {
    'all': ['mdtraj<=1.9.6',
            'matplotlib'
            ],
    }

with open('README.md', 'rt') as f:
    long_description = f.read()

setup(
    name='chemtrain',
    version='0.0.1',
    license='Apache 2.0',
    description='Training molecular dynamics potentials.',
    author='Stephan Thaler',
    author_email='stephan.thaler@tum.de',
    packages=find_packages(exclude='examples'),
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require=extras_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tummfm/chemtrain',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
    zip_safe=False,
)
