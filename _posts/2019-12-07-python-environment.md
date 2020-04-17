---
layout: post
title: Python environment
permalink: python-environment
comments: true
github: https://github.com/anthonyhu/ml-research
---

Setting a python environment is great to: 1) share code with others and 2) work on multiple machines. It ensures 
that every machine, and every collaborator runs the same code, which is a good way to stay sane.

Use conda to create a custom environment for your project. 

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), which is a lightweight installer of conda.
2. In your project repository, create an `environment.yml` file that contains all the dependencies of your
project (example below).
3. From this repository, run `conda env create` to create a conda environment as specified in the `environment.yml` file.
4. Every time we update our environment (add/delete packages), run the command `conda env update` to update
it.


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
they'll be able to generate the same python environment to run the model with `conda env create`.
-----
