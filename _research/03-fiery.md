---
layout: research
title: "FIERY: Future Instance Prediction in Bird's-Eye View from Surround Monocular Cameras"
authors: Anthony Hu, Zak Murez, Nikhil Mohan, Sofía Dudas, Jeffrey Hawke, ‪Vijay Badrinarayanan, Roberto Cipolla, Alex Kendall
venue: International Conference on Computer Vision (Oral - ICCV 2021)
thumbnail: thumbnails/fiery_thumbnail.png
permalink: fiery
---
<figure>
    <img src='/research/banners/fiery_banner.png'/>
</figure>
<center>
<h1 class="page-title">FIERY: Future Instance Prediction in Bird's-Eye View <br>from Surround Monocular Cameras
</h1>

Anthony Hu &emsp; Zak Murez &emsp; Nikhil Mohan &emsp; Sofía Dudas
<p>Jeffrey Hawke &emsp; ‪Vijay Badrinarayanan‬ &emsp; Roberto Cipolla &emsp; Alex Kendall</p>

<p>Wayve, University of Cambridge</p>
<b><a href="https://arxiv.org/pdf/2104.10490.pdf">Paper</a> &emsp; &emsp; &emsp;<a href="https://wayve.ai/blog/fiery-future-instance-prediction-birds-eye-view/">
Blog</a> &emsp; &emsp; &emsp;<a href="https://github.com/wayveai/fiery">Code</a></b>
</center>


<br/>
Autonomous driving is inherently a geometric problem, where the goal is to navigate a vehicle safely and correctly 
through 3D space. As such, an orthographic bird’s-eye view (BEV) perspective is commonly used for motion planning and 
prediction based on LiDAR sensing.

Over recent years, we’ve seen advances in camera-based perception rival LiDAR-based perception, and we anticipate that 
this will also be possible for wider monocular vision tasks, including prediction. Building a perception and prediction 
system based on cameras would enable a leaner, cheaper and higher resolution visual recognition system over LiDAR 
sensing.

We present <b>FIERY</b>: a future instance prediction model in bird’s-eye view from monocular cameras only. Our model 
predicts future instance segmentation and motion of dynamic agents that can be transformed into non-parametric 
future trajectories.

Our approach combines the perception, sensor fusion and prediction components of a traditional autonomous driving 
stack end-to-end, by estimating bird’s-eye-view prediction directly from surround RGB monocular camera inputs. 
We favour an end-to-end approach as it allows us to directly optimise our representation, rather than decoupling 
those modules in a multi-stage discrete pipeline of tasks which is prone to cascading errors and high-latency.

Further, classical autonomous driving stacks tackle future prediction by extrapolating the current behaviour of 
dynamic agents, without taking into account possible interactions. They rely on HD maps and use road connectivity 
to generate a set of future trajectories. In contrast, FIERY learns to predict future motion of dynamic agents directly 
from camera driving data in an end-to-end manner, without relying on HD maps or LiDAR sensing. It can reason about the 
inherent stochastic nature of the future, and predicts multimodal future trajectories as shown in the video below.
 
<figure>
    <img src='/research/fiery_media/fiery_intro.gif' alt='FIERY future prediction.' />
    <figcaption align='center'><em> Multimodal future predictions by our bird’s-eye view network.
<br/><b>Top two rows:</b> RGB camera inputs. The predicted future trajectories and segmentations are projected to the 
ground 
plane in the images.
<br/><b>Bottom row:</b>  future instance prediction in bird’s-eye view in a 100m×100m capture size around the ego-vehicle, which 
is indicated by a black rectangle in the center.</em></figcaption>
</figure>
-----
