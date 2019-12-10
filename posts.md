---
layout: page
title: Blog posts
---
{% for post in site.posts %}
  * [ {{ post.title }} ]({{ post.url }}) ({{ post.date | date: "%B, %Y" }})
{% endfor %}