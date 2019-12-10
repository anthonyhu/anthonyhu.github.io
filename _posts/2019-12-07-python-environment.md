---
layout: post
title: Python environment
permalink: python-environment
comments: true
---

Setting a python environment is great to: 1) share code with others and 2) work on multiple machines. It ensures 
that every machine, and every collaborator runs the same code, which is a good way to stay sane.

Use conda to create a custom environment for the project. 

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), which is a lightweight installer of conda.
2. At the root of our project repository, create an `environment.yml` file that contains all the dependencies of our
project (example below).
3. In this repository, run `conda env create` to create a conda environment as specified in the yaml file.
4. Every time we update the `environment.yml` with additional packages, simply run the command `conda env update`.


Here is an example of an `environment.yml` file.
```yaml
# Name of the environment
name: ml-research
# Packages installed with conda
dependencies:
- python=3.6.9
- pytorch=1.3.1
- torchvision=0.4.2
- cudatoolkit=10.0
- numpy=1.17.4
- pip=19.3.1
# Packages installed with pip
- pip:
    - flake8==3.7.9
    - tensorboardx==1.9
```

That's it. Whenever we want to share our code with others, we simply point them to the github repository and 
they'll be able to generate the same python environment to run our model.

Also, [autoenv](https://github.com/inishchith/autoenv) is a handy tool to automatically activate the conda environment
whenever we navigate to the project folder.

-----
