---
layout: research
title: "Model-Based Imitation Learning for Urban Driving"
authors: Anthony Hu, Gianluca Corrado, Nicolas Griffiths, Zak Murez, Corina Gurau, Hudson Yeo, Alex Kendall, Roberto Cipolla, Jamie Shotton
venue: Neural Information Processing Systems (NeurIPS 2022)
thumbnail: thumbnails/mile_thumbnail.png
permalink: mile
---
<center>
<h1 class="page-title">Model-Based Imitation Learning for Urban Driving
</h1>

Anthony Hu &emsp; Gianluca Corrado &emsp; Nicolas Griffiths &emsp; Zak Murez &emsp; Corina Gurau 
<p>Hudson Yeo &emsp; Alex Kendall &emsp; Roberto Cipolla &emsp; Jamie Shotton</p>

<p>Wayve, University of Cambridge</p>
<b><a href="https://arxiv.org/pdf/2210.07729.pdf">Paper</a> &emsp; &emsp; &emsp;<a href="https://wayve.ai/thinking/learning-a-world-model-and-a-driving-policy/">
Blog</a> &emsp; &emsp; &emsp;<a href="https://github.com/wayveai/mile">Code</a></b>
</center>


<br/>
An accurate model of the environment and the dynamic agents acting in it offers great potential for improving motion planning. We present MILE: a Model-based Imitation LEarning approach to jointly learn a model of the world and a policy for autonomous driving. Our method leverages 3D geometry as an inductive bias and learns a highly compact latent space directly from high-resolution videos of expert demonstrations. Our model is trained on an offline corpus of urban driving data, without any online interaction with the environment. MILE improves upon prior state-of-the-art by 31% in driving score on the CARLA simulator when deployed in a completely new town and new weather conditions. Our model can predict diverse and plausible states and actions, that can be interpretably decoded to bird's-eye view semantic segmentation. Further, we demonstrate that it can execute complex driving manoeuvres from plans entirely predicted in imagination. Our approach is the first camera-only method that models static scene, dynamic scene, and ego-behaviour in an urban driving environment.
 
<figure>
    <img src='/research/mile_media/mile_driving_in_imagination.gif' alt='MILE driving in its imagination.' />
    <figcaption align='center'><em> Multimodal future predictions by our bird’s-eye view network.
Our model can drive in the simulator with a driving plan predicted entirely from imagination.
<br/>From left to right we visualise: RGB input, ground truth bird's-eye view semantic segmentation, predicted bird's-eye view segmentation.
<br/>When the RGB input becomes sepia-coloured, the model is driving in imagination.</em></figcaption>
</figure>
-----
