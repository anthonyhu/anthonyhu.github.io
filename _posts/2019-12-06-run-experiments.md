---
layout: post
title: How to run and organise experiments
permalink: run-experiments
comments: true
github: https://github.com/anthonyhu/vision
---

Each experiment we run must be fully specified in a config file. See the following example for CIFAR10:

```yaml
output_path: '/tmp/'
tag: 'baseline'

batch_size: 128
n_iterations: 10000
# Print frequency
print_iterations: 100
# Visualisation frequency
vis_iterations: 500
# Validation frequency
val_iterations: 2000
# Number of workers for data loaders
n_workers: 4
gpu: True

# Optimiser
learning_rate: 0.001
weight_decay: 0.0001
```

In our codebase, we create a folder named `experiments` where we store all the config files (see the accompanying
[repository](https://github.com/anthonyhu/vision)). Also, it is useful to 
create a debug config file, that runs a full training session in order to catch any bug. For example, here is the
content of `experiments/debug_cifar.yml`:

```yaml
output_path: '/tmp/debug/'
tag: 'debug'

batch_size: 32
# Only 100 training steps to quickly run the full training session
n_iterations: 100
print_iterations: 25
vis_iterations: 50
val_iterations: 50
n_workers: 2
gpu: True

learning_rate: 0.001
weight_decay: 0.0001
```

We often find ourselves strugging to rerun an experiment because the code has changed in-between. A simple solution is 
to save the git hash of the commit associated to our experiment in the training session folder, i.e. 
in `session_{machine}_{time}_{tag}/git_hash`. When we want to run a past
experiment, or restore the weights of a trained session, we simply need to go back to that particular git hash using:

```bash
git checkout <git_hash>
``` 

And run the experiment. If from this commit we'd like to create a new branch, that's possible with `git checkout -b <branch_name>`. 
Otherwise, we can go back to master with `git checkout master`.

-----
