---
layout: research
title: Learning a Spatio-Temporal Embedding for Video Instance Segmentation
authors: Anthony Hu, Alex Kendall and Roberto Cipolla
venue: preprint (2019)
thumbnail: thumbnails/video_embedding_thumbnail.png
permalink: video-instance-embedding
---
<figure>
    <img src='/research/thumbnails/video_embedding_thumbnail.png'/>
</figure>
<center>
<h1 class="page-title">Learning a Spatio-Temporal Embedding for <br>Video Instance Segmentation
</h1>

Anthony Hu &emsp; Alex Kendall &emsp; Roberto Cipolla
<p>University of Cambridge</p>
</center>
<center>
<b><a href="https://arxiv.org/pdf/1912.08969.pdf">Paper</a> &emsp; &emsp; &emsp;<a href="https://youtu.be/dc-3meFF6z0">Video</a></b>
</center>

<br/>
We present a novel embedding approach for video instance segmentation. Our
method learns a spatio-temporal embedding integrating cues from appearance,
motion, and geometry; a 3D causal convolutional network models motion, and a
monocular self-supervised depth loss models geometry. In this embedding space,
video-pixels of the same instance are clustered together while being separated
from other instances, to naturally track instances over time without any complex post-processing. Our network runs in real-time as our architecture is entirely
causal – we do not incorporate information from future frames, contrary to previous methods. We show that our model can accurately track and segment instances,
even with occlusions and missed detections, advancing the state-of-the-art on the
KITTI Multi-Object and Tracking Dataset.



-----
