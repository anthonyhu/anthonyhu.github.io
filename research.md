---
layout: research_index
title: Research
---
{% for post in site.research reversed %}

<div>
  <span>
  <p><b><a href="{{ post.url }}">{{ post.title }}</a></b></p> 
  <p>{{ post.authors }}</p>
  <p><b>{{ post.venue }}</b></p>
  </span>
  <a href="{{ post.url }}"><img src="{{post.thumbnail}}"></a>
</div>
  <p><br/></p>

{% endfor %}
