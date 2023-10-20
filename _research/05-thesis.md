---
layout: research
title: "Neural World Models for Computer Vision"
authors: Anthony Hu
venue: PhD Thesis (2022)
thumbnail: thumbnails/cambridge.png
permalink: phd-thesis
---
<center>
<h1 class="page-title">Neural World Models for Computer Vision
</h1>

Anthony Hu

<br/>
<b><a href="https://arxiv.org/pdf/2306.09179.pdf">Thesis</a></b>
</center>


<br/>
Humans navigate in their environment by learning a mental model of the world through
passive observation and active interaction. Their world model allows them to anticipate
what might happen next and act accordingly with respect to an underlying objective.
Such world models hold strong promises for planning in complex environments like
in autonomous driving. A human driver, or a self-driving system, perceives their
surroundings with their eyes or their cameras. They infer an internal representation
of the world which should: (i) have spatial memory (e.g. occlusions), (ii) fill partially
observable or noisy inputs (e.g. when blinded by sunlight), and (iii) be able to reason
about unobservable events probabilistically (e.g. predict different possible futures). They
are embodied intelligent agents that can predict, plan, and act in the physical world
through their world model. In this thesis we present a general framework to train a world
model and a policy, parameterised by deep neural networks, from camera observations
and expert demonstrations. We leverage important computer vision concepts such as
geometry, semantics, and motion to scale world models to complex urban driving scenes.

In our framework, we derive the probabilistic model of this active inference setting
where the goal is to infer the latent dynamics that explain the observations and actions
of the active agent. We optimise the lower bound of the log evidence by ensuring the
model predicts accurate reconstructions as well as plausible actions and transitions.

First, we propose a model that predicts important quantities in computer vision:
depth, semantic segmentation, and optical flow. We then use 3D geometry as an inductive
bias to operate in the bird’s-eye view space. We present for the first time a model that
can predict probabilistic future trajectories of dynamic agents in bird’s-eye view from
360° surround monocular cameras only. Finally, we demonstrate the benefits of learning
a world model in closed-loop driving. Our model can jointly predict static scene, dynamic
scene, and ego-behaviour in an urban driving environment.

-----
