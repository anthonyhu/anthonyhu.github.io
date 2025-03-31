---
layout: research
title: "GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving"
authors: Lloyd Russell*, Anthony Hu*, Lorenzo Bertoni*, George Fedoseev*, Jamie Shotton, Elahe Arani, Gianluca Corrado*
venue: Technical Report (2025)
thumbnail: thumbnails/gaia2_thumbnail.jpg
permalink: gaia2
---
<center>
<h1 class="page-title">GAIA-2: <br>A Controllable Multi-View Generative World Model for Autonomous Driving
</h1>

Lloyd Russell* &emsp; Anthony Hu* &emsp; Lorenzo Bertoni* &emsp; George Fedoseev*
<p>Jamie Shotton &emsp; Elahe Arani &emsp; Gianluca Corrado*</p>

<p>Wayve</p>
<b><a href="https://arxiv.org/pdf/2503.20523">Paper</a> &emsp; &emsp; &emsp;<a href="https://wayve.ai/thinking/gaia-2/">
Blog</a> &emsp; &emsp; &emsp;</b>
</center>


<br/>
GAIA-2 is a latent world model trained with flow matching in the continuous latent space induced by a video tokenizer. It allows scalable simulation of diverse driving scenarios, reducing the reliance on expensive real-world data collection and facilitating robust evaluation in safe and repeatable environments. In particular, our world model excels at generation of edge-case/safety-critical data, resimulation of existing scenarios, and extreme generalisation.

Our model can generate novel driving scenarios with remarkable multi-view and temporal consistency. 
Compared to its predecessor, controllability has been significantly improved through fine-grained control on ego-vehicle 
actions, dynamic agents behaviour, scene geometry, and environmental factors. 
 
<figure>
    <img src='/research/gaia2_media/gaia2.gif' alt='Generation from GAIA-2.' />
    <figcaption align='center'><em> Generations from GAIA-2.</em></figcaption>
</figure>
-----
