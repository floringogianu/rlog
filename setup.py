from setuptools import setup, find_packages


# Following semver semantics:
# MAJOR version when you make incompatible API changes,
# MINOR version when you add functionality in a backwards compatible manner, and
# PATCH version when you make backwards compatible bug fixes.
# Labels for pre-releases such as `1.0.0-alpha`.

VERSION = '1.1.0-alpha'  # single source of truth versioning? :)

print('-- Installing rlog ' + VERSION)
with open("./rlog/version.py", 'w') as f:
    f.write("__version__ = '{}'\n".format(VERSION))

# package setup
setup(
    name='rlog',
    version=VERSION,
    description='A simple logger for reinforcement learning.',
    url='https://github.com/floringogianu/rlog',
    author='Florin Gogianu',
    author_email='florin.gogianu@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    # install_requires=[],
)
