+++
date = "2019-05-24"
description = "Algorithm Breakdown: Expectation Maximization"
tags = ["machine learning", "python", "clustering", "bayesian"]
draft = false
author = "Ritchie Vink"
title = "Algorithm Breakdown: Expectation Maximization"
keywords = ["algorithm breakdown"]
og_image = "/img/post-24-em/grand_canyon.jpg"
+++

{{< figure src="/img/post-24-em/grand_canyon.jpg" >}}

SOME INTRO

## Gaussian Mixture Model
The schoolbook example of Expectation Maximization starts with a Gaussian Mixture model. Below we will go through the definition of a GMM in 1D, but note that this will generalize to ND. Gaussian Mixtures help with the following cluster problem. Assume the following generative process. Let $X$ be an observed random variable.

$$ z_i \sim Multionomial(\phi) $$
$$ x_i|z_i \sim N(\mu_k, \sigma_k^2) $$

$x_i$ is sampled from two different Gaussians. $z_i$ is an unobserved variable that determines from which Gaussian is sampled. Note that in this 1D case, the Multinomial distribution is the Bernoulli distribution. The parameter $\phi_j$ gives $p(z_i = j)$. We could generate this data in Python as follows:

``` python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(654)
m = np.array([stats.norm(2, 1), stats.norm(5, 1.8)])
z_i = stats.bernoulli(0.75).rvs(100)
x_i = np.array([g.rvs() for g in m[z_i]])

# plot the generated data and the latent distributions.
x = np.linspace(-5, 12, 150)
plt.figure(figsize=(16, 6))
plt.plot(x, m[0].pdf(x))
plt.plot(x, m[1].pdf(x))
plt.plot(x, m[0].pdf(x) + m[1].pdf(x), lw=1, ls='-.', color='black')
plt.fill_betweenx(m[0].pdf(x), x, alpha=0.1)
plt.fill_betweenx(m[1].pdf(x), x, alpha=0.1)
plt.vlines(x_i, 0, 0.01, color=np.array(['C0', 'C1'])[z_i])
```

{{< figure src="/img/post-24-em/gmm-example.png" title="Generative process Gaussian Mixture Models." >}}

The picture above clearly describes the generative process of a Gaussian Mixture model. The dashed line shows $p(x_i|\theta)$. The blue Gaussian shows $p(x_i|z_i=1, \theta)$, and the orange Gaussian shows $p(x_i|z_i=2, \theta)$. Note that $\theta$ are the parameters of the Gaussians, $\mu$ and $\sigma$.

If $Z$ was an observed variable we would know that all data points would come from two Gaussians and for each single data point we could tell from which Gaussian it has originated.

{{< figure src="/img/post-24-em/gmm_z_observed.png" title="$X$ and $Z$ are observed." >}}

In such a case we could model the joint distribution $p(x_i, z_i) = p(x_i| z_i) p(z_i)$ (i.e. fit the Gaussians) by applying maximum likelihood estimation. However, in an unsupervised case $Z$ is latent and all we observe is $X$.

{{< figure src="/img/post-24-em/gmm_z_unobserved.png" title="$X$ is observed" >}}

With observing only $X$, it has become a lot harder to determine $p(x_i, z_i)$, as we are not sure by which Gaussian $x_i$ is produced.

# Expectation Maximization in GMM
The log likelihood of the model is defined below, but as $Z$ is unobserved, the log likelihood function has no closed form for the maximum likelihood. 

$$\ell(\phi, \theta) = \sum_{i=1}^n \text{log} p(x_i| \theta, \phi)$$

$$\ell(\phi, \theta) = \sum\_{i=1}^n \text{log} \sum\_{z\_i=1}^k p(x\_i| \theta, z\_i) p(z\_i| \phi)$$

Because of this we will use an optimization algorithm called Expectation Maximization, where we guess $Z$ and iteratively try to maximize the log likelihood.

## E-step
Given a set of initialized chosen (or updated) parameters we determine $w_{ij}$ for each data point. 

$$ w_{ij} := p(z_i = j|x_i; \theta, \phi)$$

Given the current set of parameters, what is the likelihood of data point $i$, being assigned to Gaussian $j$.

## M-step
Now we are going the parameters by applying an algorithmic step, quite similar to K-means, but now we are weighing them with the likelihoods $wi\_{ij}$. In K-means, we have a hard cut off, and determine the new means, from the data points assigned to the cluster from the previous iteration. Now we will use all the data points for determining the new mean, but we scale them.


$$ \phi\_j := \frac{1}{n}\sum\_{i=1}^n w\_{ij} $$ 
$$ \mu\_j := \frac{\sum\_{i=1}^nw\_{ij} x\_i} {\sum\_{i=1}^nw\_{ij}} $$ 
$$ \sigma_j :=  \sqrt{\frac{\sum\_{i=1}^nw\_{ij}(x_i - \mu_j)^2} {\sum\_{i=1}^nw\_{ij}}}  $$

If we iterate the E-M steps, we hope to converge to maximum likelihood. (The log likelihood function, is multi-modal so we could get stuck in a local optimum)

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
