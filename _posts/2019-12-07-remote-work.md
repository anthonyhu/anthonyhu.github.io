---
layout: post
title: Remote work
permalink: remote-work
comments: true
github: https://github.com/anthonyhu/ml-research
---

Most of the time, we code on our personal computer and run experiments on remote servers. As using git and pushing/pulling
all the time is tedious, I'll show you a better way to:
1. Automatically sync local changes with your remote machine.
2. Debug code using the Python interpreter of your remote machine on an IDE like _PyCharm_.


### SSH configuration
First, let's setup the ssh configuration on your personal computer so that you can connect to a remote server with a simple 
command like `ssh <machine_name>`. Modify your `~/.ssh/config` file as such:

```
# First log into your internal network (optional)
Host <network_name>
  HostName         <network_hostname>
  User             <username>

# Then to your remote machine.
Host <machine_name>
  HostName         <machine_hostname>
  User             <username>
  ProxyCommand     ssh -W %h:%p <network_name>   # (optional, only if you've got an internal network)
  LocalForward     6006 127.0.0.1:6006
  LocalForward     8888 127.0.0.1:8888
```

The two `LocalForward` lines forward the port `6006` and `8888` of the remote server to your personal machine. For example,
the port `6006` can be used to run tensorboard on the remote machine, and the port `8888` to run jupyter notebook.
You can then access tensorboard and jupyter on your personal machine on `http://localhost:6006` and `http://localhost:8888`,
as if you were on the remote server.


### Remote interpreter
Now, let's setup PyCharm to use a remote interpreter, i.e. so we can code locally on our personal machine, while 
running/debugging scripts on a remote machine, which is very handy if we want to test GPU or multi-GPU code for 
example. We need PyCharm Professional to do that, which is available freely [for students](https://www.jetbrains.com/student/). 

1. Open your project repository on PyCharm.
2. Setup the remote interpreter by clicking on: PyCharm (top left) → Preferences → Project Interpreter → Click on the
cogwheel (top right) → Add → SSH 
Interpreter.
3. Select 'New server configuration' and type: Host: \<machine_name>, Username: \<username>
followed by your password.
4. Specify the path of the remote interpreter. (To get the path, ssh into your remote machine, 
activate your conda environment and type `which python`)
5. Specify the matching github repository on the remote machine, i.e. the path of the project repository on the remote machine.

All good! Now any change applied locally will also happen on the remote machine. We can now debug code/develop 
on the remote machine, from the comfort of our home.


Last point: as the modifications are automatically transferred to the remote machine, we can't directly use the command `git pull`
on the remote machine as it will create git conflicts. Even though all the files are identical on both the local and remote machines, 
the `.git` folder is not synchronised. To pull on the remote machine, first do `git reset --hard` (to clean branch) 
and `git clean -fd` (to remove all untracked files). Only then can we
safely use `git pull`.

-----
