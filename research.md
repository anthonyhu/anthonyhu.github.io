---
layout: research_index
title: Research
---
{% for post in site.research reversed %}

<div>
  <a href="{{ post.url }}"><img src="{{post.thumbnail}}"></a>
  
  <span>
  <p><a href="{{ post.url }}">{{ post.title }}</a></p> 
  <p>{{ post.authors }}</p>
  <p>{{ post.venue }}</p>
  </span>
</div>
  <p><br/></p>
  <p><br/></p>

{% endfor %}
