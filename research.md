---
layout: research_index
title: Research
---
{% for post in site.research reversed %}

<div>
  <span>
  <p><a href="{{ post.url }}">{{ post.title }}</a></p> 
  <p>{{ post.authors }}</p>
  <p>{{ post.venue }}</p>
  </span>
  <a href="{{ post.url }}"><img src="{{post.thumbnail}}"></a>
</div>
  <p><br/></p>

{% endfor %}
