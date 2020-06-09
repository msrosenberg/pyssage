import setuptools


with open("README.md", "r") as infile:
    long_description = infile.read()


setuptools.setup(
    name="pyssage",
    version="0.0.1",
    author="Michael S. Rosenberg",
    author_email="msrosenberg@vcu.edu",
    description="python version of some of the analyses from PASSaGE 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/....",
    packages=setuptools.find_packages(),
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.6',
)
