from setuptools import setup, find_packages


PACKAGENAME = "umachine_pyio"
VERSION = "0.1"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Python tools for I/O of umachine outputs",
    long_description="Python tools for I/O of umachine outputs",
    install_requires=["numpy"],
    packages=find_packages(),
    url="https://github.com/aphearin/umachine_pyio",
)
