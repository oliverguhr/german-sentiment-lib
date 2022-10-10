import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="germansentiment",
    version="1.1.0",
    author="Oliver Guhr",
    author_email="oliver.guhr@htw-dresden.de",
    description="A python package for german language sentiment classification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oliverguhr/german-sentiment-lib",
    packages=setuptools.find_packages(),    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
       "transformers",
       "torch>=1.8.1",
    ],
    python_requires='>=3.6',
)
