from setuptools import setup, find_packages


VERSION = '0.2.0'  # single source of truth versioning? :)

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
