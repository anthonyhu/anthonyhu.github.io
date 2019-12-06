---
layout: post
title: How to run and organise experiments
permalink: run-experiments
comments: true
---

Each experiment we run must be fully specified in a config file. See the following example for CIFAR:

```yaml
output_path: '/tmp/'
tag: 'baseline'

batch_size: 128
n_iterations: 10000
# Output print frequency
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

In our codebase, we create a folder named `experiments` where we store all the config files. Also, it is useful to 
create a debug config file, that runs a full training session in order to catch any bug. For example, here is the
content of `experiments/debug_cifar.yml`:

```yaml
output_path: '/tmp/debug/'
tag: 'debug'

batch_size: 32
n_iterations: 100
print_iterations: 25
vis_iterations: 50
val_iterations: 50
n_workers: 2
gpu: True

learning_rate: 0.001
weight_decay: 0.0001
```

We often find ourselves strugging to rerun an experiment because the code has changed between the moment the given
experiment was run and present time. To solve this issue, we save the git hash of the particular commit associated
to our experiment in the session folder, i.e. in `session_{machine}_{time}_{tag}/git_hash`. When we want to run a past
experiment, or restore the weights of a trained session, we simply need to go back to that particular git hash using:

```bash
git checkout <git_hash>
``` 

And run the experiment. If from this commit we'd like to create a new branch, that's possible with `git checkout -b <branch_name>`. Otherwise,
we can go back to master with `git checkout master`.


