---
layout: research
title: "GAIA-1: A Generative World Model for Autonomous Driving"
authors: Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado
venue: Technical Report (2023)
thumbnail: thumbnails/gaia_thumbnail.jpeg
permalink: gaia
---
<center>
<h1 class="page-title">GAIA-1: <br>A Generative World Model for Autonomous Driving
</h1>

Anthony Hu &emsp; Lloyd Russell &emsp; Hudson Yeo &emsp; Zak Murez &emsp; George Fedoseev
<p>Alex Kendall &emsp; Jamie Shotton &emsp; Gianluca Corrado</p>

<p>Wayve</p>
<b><a href="https://arxiv.org/pdf/2309.17080.pdf">Paper</a> &emsp; &emsp; &emsp;<a href="https://wayve.ai/thinking/scaling-gaia-1/">
Blog</a> &emsp; &emsp; &emsp;</b>
</center>


<br/>
Autonomous driving promises transformative improvements to transportation, but building systems capable of safely navigating the unstructured complexity of real-world scenarios remains challenging. A critical problem lies in effectively predicting the various potential outcomes that may emerge in response to the vehicle’s actions as the world evolves. 

To address this challenge, we introduce GAIA-1 (‘Generative AI for Autonomy’), a generative world model that leverages video, text, and action inputs to generate
realistic driving scenarios while offering fine-grained control over ego-vehicle
behavior and scene features. A world model is a predictive model of the future, allowing it to understand the consequences of its actions. We cast world modelling as a self-supervised sequence modelling problem, where the goal is to predict the next discrete token in the sequence. Similarly to language models, we show that the performance of world models scales gracefully with more parameters (~10B) and compute. Emerging properties of GAIA-1 include: generalisation to out-of-distribution states, contextual awareness, and understanding of 3D geometry.

 
<figure>
    <img src='/research/gaia_media/gaia.gif' alt='Generation from GAIA-1.' />
    <figcaption align='center'><em> Generation from GAIA-1 with different text promps, and action conditioning.</em></figcaption>
</figure>
-----
