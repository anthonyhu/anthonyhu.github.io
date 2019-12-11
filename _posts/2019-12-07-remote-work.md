---
layout: post
title: Remote work
permalink: remote-work
comments: true
github: https://github.com/anthonyhu/ml-research
---

Most of the time, we code on our personal machine (that might not have a GPU), and run experiments on remote servers. 
First, we must ensure that all machines use the same [python environment]({% post_url 2019-12-07-python-environment%}). 


### SSH configuration
Then we setup the ssh configuration to be able to connect to a remote server with a simple command like `ssh direwolf`.
Here is my `~/.ssh/config` file:

```
# Log into the Cambridge network
Host cued
  HostName         gate.eng.cam.ac.uk
  User             ah2029

# Then to my machine. Note the ProxyCommand that first ssh into the 
# Cambridge network
Host direwolf
  HostName         direwolf
  User             anthonyhu
  ProxyCommand     ssh -W %h:%p cued
  LocalForward     6006 127.0.0.1:6006
  LocalForward     8888 127.0.0.1:8888
```

The two extra `LocalForward` lines enables to forward the port `6006` and `8888` to our personal machine. For example,
the port `6006` can be used to run tensorboard on the remote machine, and the port `8888` to run jupyter notebook.
We can then access tensorboard and jupyter on our personal machine on `http://localhost:6006` and `http://localhost:8888`.


### PyCharm remote interpreter
Now, let us setup PyCharm to use a remote interpreter, i.e. so we can code locally on our personal machine, while 
running/debugging scripts on a remote machine, which is very handy if we want to test GPU code or multi-GPU code for 
example. We need PyCharm Professional to do that, which is available freely [for students](https://www.jetbrains.com/student/). 

1. Setup the remote interpreter. PyCharm → Preferences → Project Interpreter → Click on cogwheel → Add → SSH Interpreter.
2. Select 'New server configuration' and type: Host: `direwolf` Username: `anthonyhu` (replace with your hostname and username)
and type your password.
3. Specify the path of the interpreter. (To get the path, go to your remote machine, 
activate your conda environment and type `which python`.
The output will look like `/home/anthonyhu/miniconda3/envs/vision/bin/python`.)
4. Specify the matching repository on the remote machine, i.e. the path of the project repository on the remote machine.

All good. now any change applied locally will also take effect on the remote machine. We can now debug code/use 
jupyter notebook on the remote machine, from the comfort of our home.


Last point: as the modification are automatically transferred to the remote machine, we cannot use the command `git pull`
on the remote machine directly (even though all the files are the same, the git diff will be different as .git is not synchronised). 
First do `git reset --hard` (to clean branch) and `git clean -fd` (to remove all untracked files). Only then can we
safely `git pull` on the remote machine.

-----
