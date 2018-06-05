+++
date = "2018-06-05"
description = ""
tags = ["python", "machine learning", "pymc3", "edward", "bayesian", "clustering"]
draft = false
author = "Ritchie Vink"
title = "Clustering with Gaussian Mixture models and Dirichlet Process Mixtures."
+++

{{< figure src="/img/post-15-mixture_models/gmount1.jpg">}}

Last post I've described the [Affinity Propagation]({{< ref "post/affinity_propagation.md" >}}) algorithm. The reason why I wrote about this algorithm was because I was interested in clustering data points without specifying **k**, i.e. the number of clusters present in the data. 

This post continues with the same fascination, however now we take a generative approach. In other words, we are going to examine which models could have generated the observed data. Through bayesian inference we hope to find the hidden (latent) distributions that most likely generated the data points. When there is more than one latent distribution we can speak of a Mixture of distributions. If we assume that the latent distribrutions are Gaussians than we call this model a Gaussian Mixture model.

First we are going to define a bayesian model in [Edward](http://edwardlib.org) to determine a multivariate gaussian mixture model where we predifine **k**. Just as in k-means clustering we will have a hyperparameter **k** that will dictate the amount of clusters. In the second part of the post we will reduce the dimensionality of our problem to one dimension and look at a model that is completely nonparametric and will determine **k** for us.

## Dirichlet Distribution

Before we start with the generative model, we take a look at the Dirichlet distribution. This is a distribution of distributions and can be a little bit hard to get your head around. If we sample from a Dirichlet we'll retrieve a vector of probabilities that sum to 1. These discrete probabilites can be seen as seperate events. A Dirichlet distribution can be compared to a bag of badly produced dice, where each dice has a totally different probability of throwing 6. Each time you sample a dice from the bag you sample another probabilty of throwing 6. However you still need to sample from the dice. Actually throwing the dice will lead to sampling the event.

The Dirichlet is defined by:

<div class="formula-wrap">
$$\theta \sim Dir(\alpha)$$
</div>

<div class="formula-wrap">
$$P(x) = \frac{1}{B(\alpha)}\prod_{i=1}^{k}{x_i^{\alpha_i-1}}$$
</div>

where 

<div class="formula-wrap">
$$B(\alpha) = \frac{\prod_{i=1}^{k}{\Gamma(\alpha_i)}}{\Gamma(\sum_{i=1}^{n}{\alpha_i})}$$
</div>

<br>
This distribution has one parameter $\alpha$ that influences the probability vector that is sampled. Let's take a look at the influence of $alpha$ on the samples. We can best investigate the Dirichlet distribution in three dimensions; $\theta = [\theta_1, \theta_2, \theta_3]$. We can plot every probability sample $\theta$ as a point in three dimensions. By sampling a lot of distribution points $\theta$ we will get an idea of the Dirichlet distribution $Dir(\alpha)$. 

If we want to create a Dirichlet distribution in three dimensions we need to initialize it with $\alpha = [\alpha_1, \alpha_2, \alpha_3]$. The expected value of $\theta$ becomes:

<div class="formula-wrap">
$$\mathbb{E} \theta_i = \frac{\alpha_i}{\sum_{j=1}^{k}{\alpha_j}}$$
</div>

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

for alpha in [[1, 3, 4], [1, 1, 1], [10, 0.2, 0.2], [0.1, 0.1, 0.1]]:
    theta = stats.dirichlet(alpha).rvs(500)

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.gca(projection='3d')
    plt.title(r'$\alpha$ = {}'.format(alpha))
    ax.scatter(theta[:, 0], theta[:, 1], theta[:, 2])
    ax.view_init(azim=30)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')
    plt.show()
```

{{< figure src="/img/post-15-mixture_models/134.png" width="500px" >}}
{{< figure src="/img/post-15-mixture_models/111.png" width="500px" >}}
{{< figure src="/img/post-15-mixture_models/100202.png" width="500px" >}}
{{< figure src="/img/post-15-mixture_models/010101.png" width="500px" >}}

Above we can see the underlying distribution of $Dir(\alpha)$ by sampling from it. Note that $\alpha = [10, 0.2, 0.2]$ leads to high probability of <span>$P(\alpha\_1)$ close to 1 and that $\alpha = [1, 1, 1]$ can be seen as uniform Dirichlet distribution, i.e. that there is an equal probability for all distributions that suffices $\sum_{i=1}^{k}{\theta_i} = 1$.


## Clustering with generative models
Now, we have had a nice intermezzo of Dirichlet distributions, we're going to apply this distribution in a Gaussian Mixture model. We will try to cluster the Iris dataset. This is a dataset containing 4 columns of data gathered from 3 different types of flowers.

```python
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
from scipy import stats
import tensorflow as tf
import edward as ed

df = pd.DataFrame(load_iris()['data'])
y = df.values
# Standardize the data
y = (y - y.mean(axis=0)) / y.std(axis=0)

# A 2D pairplot between variables
df['target'] = load_iris()['target']
sns.pairplot(df, hue='target', vars=[0, 1, 2, 3])
```

{{< figure src="/img/post-15-mixture_models/iris.png" width="700px" >}}

### Gaussian Mixture

A Gaussian Mixture model is the Mother Of All Gaussians. For column **0** in our dataframe it is the cumulative of the histograms of the data labels.

```python
plt.figure(figsize=(12, 6))
b = 8
a = 4

plt.title('MOAS')
plt.xlabel(r'$X$')
plt.ylabel('bin count')
plt.annotate('Cummulative distribution', (6.8, 17), (7, 20), arrowprops=dict(facecolor='black', shrink=0.05))
plt.hist(df[0], color=sns.color_palette()[3], rwidth=a)
plt.hist(df[0][df['target'] == 2], color=sns.color_palette()[2], rwidth=a)
plt.hist(df[0][df['target'] == 1], color=sns.color_palette()[1], rwidth=a)
plt.hist(df[0][df['target'] == 0], color=sns.color_palette()[0], rwidth=a)
```

{{< figure src="/img/post-15-mixture_models/MOAS.png" width="700px" >}}

We could model the red histogram as single Gaussian. However we know that this often will be a flawed model as the red histogram may be pretty skewed. A better model would be comprised of a mixture of Gaussian. However the integral of a Gaussian distribution is equal to 1, as a probability function should be. If we combine various Gaussian we need to weight them so the integral will meet the conditon of being equal to 1.

<div class="formula-wrap">
$$\sum_{i=1}^{k} { \int_{-\infty}^{\infty}{ f_i }} = 1$$
</div>

where 

<div class="formula-wrap">
$$ f_i = Normal(\mu_i, \sigma_i)$$
</div>

We could of course scale the mixture of Gaussian by weights summing to 1. Here comes the Dirichlet distribution in place. Every sample from a Dirichlet sums to one and could be used as weights to scale down the mixture of Gaussian distributions.

The final generative model can thus be defined by:

<div class="formula-wrap">
$$ P(x_j | \pi, \mu, \sigma) = \sum_{i=1}^{k} \pi_i Normal(x_j | \mu_i, \sigma_i)$$
</div>

where $\pi$ are the weights drawn from $Dir(\alpha)$. One such mixture model for the histogram above for instance could look like:

```python
np.random.seed(13)
x = np.linspace(-2, 6, 500)
pi = stats.dirichlet([1, 1, 1]).rvs()[0]
mu = stats.norm(1, 1).rvs(3)

y = stats.norm(mu, np.ones(3)).pdf(x[:, None])
plt.figure(figsize=(12, 6))
plt.title('Gaussian Mixture model')
plt.ylabel(r'$P(x)$')
plt.xlabel(r'$X$')

plt.plot(x, y[:, 1] * pi[1], ls='--')
plt.plot(x, y[:, 2] * pi[2], ls='--')
plt.plot(x, y[:, 0] * pi[0], ls='--')

for i in range(3):
    xi = x[np.argmax(y[:, i])]
    yi = (y[:, i] * pi[i]).max()
    plt.vlines(xi, 0, yi)
    plt.text(xi + 0.05, yi + 0.01, r'$\pi_{}$'.format(i + 1))
plt.plot(x, (y * pi).sum(1))

```

{{< figure src="/img/post-15-mixture_models/gmm.png" width="700px" >}}

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
  });
  </script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<head>

<style>

.formula-wrap {
overflow-x: scroll;
}

</style>

</head>
