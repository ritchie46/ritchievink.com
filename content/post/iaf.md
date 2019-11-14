+++
date = "2019-11-12"
description = "Improve variation posterior with Inverse Autoregressive Flows."
tags = ["machine learning", "python", "deep-learning", "bayesian", "pytorch"] 
draft = false
author = "Ritchie Vink"
title = "Inverse Autoregressive Flows"
keywords = []
og_image = "/img/post-30-iaf/og_image.png"
+++

{{< figure src="/img/post-30-iaf/og_image.png" >}}

<br>
This post we will explore normalizing flow called **Inverse Autoregressive Flow**. A composition (flow) of transformations, while preserving the constraints of a probability distribution (normalizing), can help us obtain highly correlated and variational distributions. 

*Don't repeat yourself*<br>
If what was mentioned in the previous lines didn't ring a bell, do first read these posts: [variational inference]({{< ref "vi_from_scratch.md" >}}) and [normalizing flows]({{< ref "normalizing_flows.md" >}}). This post could really be seen as an extension of the latter.

## Planar/ Radial flows
Rezende & Mohammed<sup>[1]</sup> discussed two normalizing flows, the planar fow and the radial flow. If we take a look at the definition of the planar flow transformation:

\begin{equation}
f(z) = z + u h(w^Tz + b)
\end{equation}

Where the latent variable $z \in \mathbb{R}^D$, and the transformations learnable parameters; $u \in \mathbb{R}^D, w \in \mathbb{R}^D, b \in \mathbb{R}$, and $h(\cdot)$ is a smooth activation function. <br>
As $w^Tz \mapsto \mathbb{R}^1$, the transformation is comparable to a neural network layer with **one** activation node. Evidently, this is quite restrictive and we need a lot of transformation to be able to transform high dimensional space. This same restriction is true of the radial flow variant.

## Improving normalizing flows
As discussed in the post [normalizing flows]({{< ref "normalizing_flows.md" >}}), a normalizing flow is defined as:

\begin{eqnarray}
Q(z') &=& Q(z) \left| \det \frac{\partial{f}}{\partial{z}} \right|^{-1}
\end{eqnarray}

Where $f(\cdot)$ is an invertible transformation $f: \mathbb{R}^m \mapsto \mathbb{R}^m$. This leads to two constraints of $f(\cdot)$:

* $f(\cdot)$ should be invertible.
* The determinant Jacobian $\left| \det \frac{\partial{f}}{\partial{z}} \right|$ should be cheaply computable.


## References
&nbsp; [1] Rezende & Mohammed (2016, Jun 14) *Variational Inference with Normalizing Flows*. Retrieved from https://arxiv.org/pdf/1505.05770.pdf <br>
&nbsp; [2] Adam Kosiorek (2018, Apr 3) *Normalizing Flows*. Retrieved from http://akosiorek.github.io/ml/2018/04/03/norm_flows.html <br>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]},
    TeX: { equationNumbers: { autoNumber: "AMS" } }
  });
  </script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.6/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<head>

<style>

.formula-wrap {
overflow-x: scroll;
}

</style>

</head>
