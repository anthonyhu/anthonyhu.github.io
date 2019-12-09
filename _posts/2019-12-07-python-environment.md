---
layout: post
title: Python environment
permalink: python-environment
comments: true
---

Setting a python environment is great when: 1. sharing code with others and 2. working on multiple machines. We ensure 
that every machine, and every collaborator runs the same code, which is a good way to stay sane.

Use conda to create a custom environment for our project. 

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), which is a lightweight installer of conda.
2. At the root of our project repository, create an `environment.yml` file that contains all the dependencies of our
project. 
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
- pillow=6.2.1
- pip=19.3.1
- pyyaml=5.1.2
# Packages installed with pip
- pip:
    - flake8==3.7.9
    - tensorboardx==1.9
```

That's it. Whenever we want to share our code with others, we simply point them to your github repository and 
they'll be able to generate the same python environment to run our model. 