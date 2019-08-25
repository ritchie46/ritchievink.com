+++
date = "2019-08-25"
description = "Optimize black box functions with Bayesian Optimization in Python."
tags = ["algorithm breakdown", "machine learning", "python", "gaussian processes", "bayesian", "optimization"]
draft = false
keywords = ["algorithm breakdown", "machine learning", "python", "gaussian processes", "bayesian", "optimization"]
author = "Ritchie Vink"
title = "Algorithm Breakdown: Bayesian Optimization"
og_image = "/img/post-26/explore-forest.png"
+++

{{< figure src="/img/post-26/explore-forest.png" >}}

Not that long ago I wrote an introduction post on [Gaussian Processes]({{< ref "gaussian-process.md" >}}) (GP's), a regression technique where we condition a Gaussian prior distribution over functions on observed data. GP's can model any function that is possible within a given prior distribution. And we don't get a function $f$, we get a whole posterior distritbution of functions $P(f|X)$.

This of course sounds very cool and all, but there is no free lunch. GP's have a complexity $\mathcal{O}(N^3)$, where $N$ is the number of data points. GP's work well up to approximately 1000 data points (I should note that there are approximating solutions that scale better). This led me to believe that I wouldn't be using them too much in real world problems, but luckily I was very wrong! 




<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
  });
  </script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<head>

<style>

.formula-wrap {
overflow-x: scroll;
}

</style>

</head>
