---
layout: research
title: "FIERY: Future Instance Prediction in Bird's-Eye View </br>from Surround Monocular Cameras"
---
<center>
Anthony Hu &emsp; Zachary Murez &emsp; Nikhil Mohan &emsp; Sofía Dudas &emsp;
<p>Jeff Hawke &emsp; ‪Vijay Badrinarayanan‬ &emsp; Roberto Cipolla &emsp; Alex Kendall</p>
University of Cambridge. Wayve, UK.
</center>

<br/>

Prediction of future states is a key challenge in many autonomous decision making systems. This is particularly true 
for motion planning in highly dynamic environments such as autonomous driving where the motion of other road 
users and pedestrians has a substantial influence on the success of motion planning.
Estimating the motion and future poses of these road users enables motion planning algorithms to better resolve 
multimodal outcomes where the optimal action may be ambiguous knowing only the current state of the world.

Autonomous driving is inherently a geometric problem, where the goal is to navigate a vehicle safely and correctly 
through 3D space. As such, an orthographic bird's-eye view (BEV) perspective is commonly used for motion planning and 
prediction based on LiDAR sensing. Recent advances in camera-based perception have 
rivalled LiDAR-based perception, and we anticipate that this will also be possible for wider 
monocular vision tasks, including prediction. Building a perception and prediction system based on cameras would enable 
a leaner, cheaper and higher resolution visual recognition system over LiDAR sensing.


We present __FIERY__: a future instance prediction model in bird's-eye view from monocular cameras. Our model 
predicts future instance segmentation and motion of dynamic agents which can be transformed into non-parametric 
future trajectories. We combine perception, sensor fusion and prediction by estimating 
bird's-eye-view prediction directly from surround RGB monocular camera inputs, rather than a multi-stage 
discrete pipeline of tasks prone to cascading errors and high-latency.

FIERY can reason about the inherent stochastic nature of the future, and predicts multimodal and
 visually plausible future trajectories as shown in the video below. 
 
<figure>
    <img src='/research/fiery_media/fiery_intro.gif' alt='FIERY future prediction.' />
    <figcaption align='center'> The 2 top rows correspond to the input of the model: monocular camera video covering 
    a 360° field of view around the ego-car. We visualise the predicted future trajectories and segmentations projected
     on the RGB images.<br/>
    On the bottom row, we show the mean future, and the distribution of futures in bird's-eye view.</figcaption>
</figure>


# Model Architecture

<figure>
    <img src='/research/fiery_media/model_diagram.jpg' alt='Model architecture.' />
    <figcaption align='center'>The 6 building blocks of FIERY. </figcaption>
</figure>

1. At each past timestep $\{1,...,t\}$, we lift camera features $(O_1, ..., O_t)$ to 3D by predicting a depth 
probability distribution and using known camera intrinsics and extrinsics.
2. These features are projected to bird's-eye view $(x_1, ..., x_t)$. Using past ego-motion $(a_1, ..., a_{t-1})$, we 
transform the bird's-eye view features into the present reference frame (time t) with a Spatial Transformer module $S$.
3. A 3D convolutional temporal model learns a spatio-temporal state $s_t$.
4. We parametrise two probability distributions: the present and the future distribution. The present distribution is 
conditioned on the current state $s_t$, and the future distribution is conditioned on both the current state $s_t$ and future labels $(y_{t+1}, ..., y_{t+H})$.
5. We sample a latent code $\eta_t$ from the future distribution during training, and from the present distribution 
during inference. The current state $s_t$ and the latent code $\eta_t$ are the inputs to the future prediction model 
that recursively predicts future states ($\hat{s}_{t+1}, ..., \hat{s}_{t+H})$. $\hat{a}$
6. The states are decoded into future instance segmentation and future flow in bird's-eye view $(\hat{y}_t, ..., 
\hat{y}_{t+H})$.

# What made it work?

<figure>
    <img src='/research/fiery_media/ablations.png' alt='Various ablations of our model.' />
    <figcaption align='center'>Performance of various ablations of our model. The metric reported is future video 
    panoptic quality for 2.0s future prediction. </figcaption>
</figure>

### Learning a spatio-temporal state.
Predicting the future requires understanding the past. However, learning correspondences and motion from past camera 
inputs can be tricky as the ego-vehicle is also moving.

As shown in the figure above, the two largest performance gains 
come from (i) having a temporal model to incorporate past context, and (ii) lifting past features to the common 
reference frame of the present. When __past features are in a common reference frame__ (and therefore ego-motion is 
factored out) the task of learning correspondences and motion of dynamic agents becomes much simpler.

### Predicting future states.
When predicting the future, it is important to model its __sequential nature__, i.e. the prediction at time $t+1$ 
should 
be conditioned on the prediction at time $t$.

The "no unrolling" variant which directly predicts all future instance segmentations and motions from the current 
state $s_t$, results in a large performance drop. This is because the sequential constraint is no longer enforced, 
contrarily to our approach that predicts future states in a recursive way.
 
### Probabilistic modelling
The future in inherently uncertain, and different outcomes are probable given a unique and deterministic past. By 
blindly penalising the model with the ground-truth future labels, which corresponds to one of the possible futures, 
our model is not encouraged to learn the different modes of the futures.

We introduce two probabilistic distributions:
1. The __future distribution__ is used during training, and is conditioned on the current state $s_t$ as well as future 
labels $(y_{t+1}, ..., y_{t+H})$. We sample a latent code from the future distribution to guide the future prediction
 module to the observed future in the training sequence.
2. The __present distribution__ is only conditioned on the current state $s_t$. It is encouraged to capture the 
different
 modes of the future with a mode-covering Kullback-Leibler divergence loss of the future distribution with respect to
  the present distribution. During 
 inference, different latent codes samples from the present distribution will correspond to different plausible 
 futures.
 
 
# Future Work

Autonomous driving requires decision making in multimodal scenarios, where the present state of the world is not 
always sufficient to reason correctly alone. Predictive models estimating the future state of the world -- 
particularly other dynamic agents -- are therefore a key component to robust driving.

In future work, we would like to jointly train a driving policy to condition the future prediction model on future 
actions. Such a framework would enable effective motion planning in a model-based reinforcement learning setting.

<br/>
<br/>

For more information check our paper here: [link] and video [embed video].

The code is also publicly available. [link]

-----
