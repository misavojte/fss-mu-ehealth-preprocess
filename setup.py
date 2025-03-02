from setuptools import setup, find_packages

setup(
    name="fss-mu-ehealth-preprocess",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
    ],
    author="FSS MU eHealth Team",
    description="Data preprocessing tools for eHealth eye tracking research",
    python_requires=">=3.6",
) 