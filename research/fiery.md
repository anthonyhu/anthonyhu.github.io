---
layout: research
title: "FIERY: Future Instance Prediction in Bird's-Eye View </br>from Surround Monocular Cameras"
---
<center>
Anthony Hu &emsp; Zachary Murez &emsp; Nikhil Mohan &emsp; Sofia Dudas &emsp;
<p>Jeff Hawke &emsp; ‪Vijay Badrinarayanan‬ &emsp; Roberto Cipolla &emsp; Alex Kendall</p>
University of Cambridge. Wayve, UK.
</center>
<br/>
<center>
<b><a href="https://arxiv.org/pdf/2003.06409.pdf">Paper</a> &emsp; &emsp; &emsp;<a href="https://wayve
.ai/blog/predicting-the-future">Blog</a> &emsp; &emsp; &emsp;<a href="https://youtu.be/ibRd_HucdWg">Video</a></b>
</center>

<br/>
We present __FIERY__: a future instance prediction model in bird's-eye view from monocular cameras. Our model generates 
diverse multimodal future trajectories of dynamic agents. We show below example of predictions. We visualise the 
predicted trajectories overlayed on the RGB images, and the bird's-eye view predictions. 
<p align='center'><img src='/research/fiery_media/fiery_intro.gif' alt='FIERY future prediction'/></p>


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
    <figcaption align='center'>Performance of various ablations of our model. </figcaption>
</figure>

### Learning a spatio-temporal state.
As shown in the figure above, the two largest performance gains come from (i) having a temporal model to process past 
context, and (ii) lifting past features to the common reference frame of the present. This is because when past 
features are in a common reference frame (and ego-motion is factored out) the task of 
learning correspondences and motion becomes much simpler.

### Predicting future states.
The "no unrolling" variant directly predicts all future instance segmentations and motions from the current state $s_t$,
and results in a large performance drop. This approach does not take into account the sequential nature of the future.
The next prediction is conditioned on the prediction that comes before that, and this constraint is naturally present
 in our approach that predicts next states in a recursive way.
 
### Probabilistic modelling
The future in inherently uncertain, and different outcomes are probable given a unique and deterministic past. By 
blindly penalising the model with the ground-truth future labels, which corresponds to one of the possible futures, 
our model is not encouraged to learn about the different modes of the futures.

We introduce two probabilistic distributions.
1. The __future distribution__ is used during training, and is conditioned on the current state $s_t$ as well as future 
labels $(y_{t+1}, ..., y_{t+H})$. We sample a latent code from the future distribution to guide the future prediction
 module to the correct future of that training sequence.
2. The __present distribution__ is only conditioned on the current state $s_t$. It is encouraged to capture the 
different
 modes of the future with a mode-covering Kullback-Leibler divergence loss with the future distribution. During 
 inference, different latent codes samples from the present distribution will correspond to different plausible 
 futures.
 
 
# Future Work

We want to condition the future prediction model on the policy of the ego-vehicle. This world-model would then be 
used in a model-based reinforcement learning setting to learn a robust driving policy. 


-----
