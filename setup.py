from setuptools import setup, find_packages

setup(
    name="avm-core",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pyyaml",
        "tqdm",
        "beautifulsoup4",
        "requests"
    ]
)