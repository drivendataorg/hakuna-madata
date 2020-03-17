from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="thunder_hammer",
    version="0.0.1",
    url="https://github.com/n01z3/thunder-hammer",
    author="n01z3",
    author_email="kuzin.artur@gmail.com",
    description="",
    packages=find_packages(),
    install_requires=required,
)
