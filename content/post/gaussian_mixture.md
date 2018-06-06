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

The Dirichlet distribution is defined by:

<div class="formula-wrap">
$$\theta \sim Dir(\alpha) \tag{1.0}$$
</div>

<div class="formula-wrap">
$$P(x) = \frac{1}{B(\alpha)}\prod_{i=1}^{k}{x_i^{\alpha_i-1}} \tag{1.1}$$
</div>

where 

<div class="formula-wrap">
$$B(\alpha) = \frac{\prod_{i=1}^{k}{\Gamma(\alpha_i)}}{\Gamma(\sum_{i=1}^{n}{\alpha_i})} \tag{1.2}$$
</div>

<br>
This distribution has one parameter $\alpha$ that influences the probability vector that is sampled. Let's take a look at the influence of $alpha$ on the samples. We can best investigate the Dirichlet distribution in three dimensions; $\theta = [\theta_1, \theta_2, \theta_3]$. We can plot every probability sample $\theta$ as a point in three dimensions. By sampling a lot of distribution points $\theta$ we will get an idea of the Dirichlet distribution $Dir(\alpha)$. 

If we want to create a Dirichlet distribution in three dimensions we need to initialize it with $\alpha = [\alpha_1, \alpha_2, \alpha_3]$. The expected value of $\theta$ becomes:

<div class="formula-wrap">
$$\mathbb{E} \theta_i = \frac{\alpha_i}{\sum_{j=1}^{k}{\alpha_j}} \tag{1.3}$$
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

Of we obtain a dataset without labels. What we would observe from the world is the single red distribution. However we know now that the data actually is produced by the 3 latent distributions, namely three different kind of flowers. The red observed distribution is a mixture of the hidden distributions. We could model this by choosing a mixture of 3 Gaussian distributions
 
However the integral of a Gaussian distribution is equal to 1, as a probability function should be. If we combine various Gaussians we need to weigh them so the integral will meet the conditon of being equal to 1.

<div class="formula-wrap">
$$\sum_{i=1}^{k} { \int_{-\infty}^{\infty}{ Normal(\mu_i, \sigma_i) }} = k \tag{2.0}$$
</div>

We could of course scale the mixture of Gaussian by weights summing to 1. Here comes the Dirichlet distribution in place. Every sample from a Dirichlet sums to one and could be used as weights to scale down the mixture of Gaussian distributions.

The final generative model can thus be defined by:

<div class="formula-wrap">
$$ P(x_j | \pi, \mu, \sigma) = \sum_{i=1}^{k} \pi_i Normal(x_j | \mu_i, \sigma_i) \tag{2.1}$$
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

### Generative model in Edward

The model defined in **eq. (2.1)** is conditional on $\pi$, $\mu$ and $\sigma$. We don't know the values of these variables, but as we are going to use bayesian inference we can gice a prior 
probability on them. The priors we choose:

<div class="formula-wrap">
$$ P(\pi) = Dir(\vec{1}) \tag{3.0} $$
</div>

<div class="formula-wrap">
$$ P(\mu) = Normal(0, 1) \tag{3.1} $$
</div>

<div class="formula-wrap">
$$ P(\sigma^2) = InverseGamma(1, 1) \tag{3.2} $$
</div>

We can define this generative model in Edward. First we define the priors and finally we combine them in a mixture of Multivariate Normal distributions.


```python
k = 3  # number of clusters

d = df.shape[1]
n = df.shape[0]

pi = ed.models.Dirichlet(tf.ones(k))
mu = ed.models.Normal(tf.zeros(d), tf.ones(d), sample_shape=k)  # shape (3, 4) 3 gaussians, 4 variates
sigmasq = ed.models.InverseGamma(tf.ones(d), tf.ones(d), sample_shape=k)
x = ed.models.ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                 ed.models.MultivariateNormalDiag,
                 sample_shape=n)
z = x.cat
```

Now this model is defined we can start inferring from the observed data. To be able to do Gibbs sampling in Edward we need to define Empricial distributions for our priors. 

```
t = 500  # number of samples

qpi = ed.models.Empirical(tf.get_variable('qpi', shape=[t, k], initializer=tf.constant_initializer(1 / k)))
qmu = ed.models.Empirical(tf.get_variable('qmu', shape=[t, k, d], initializer=tf.zeros_initializer()))
qsigmasq = ed.models.Empirical(tf.get_variable('qsigmasq', shape=[t, k, d], initializer=tf.ones_initializer()))
qz = ed.models.Empirical(tf.get_variable('qz', shape=[t, n], initializer=tf.zeros_initializer(), dtype=tf.int32))     
```

And hit the magix infer button!

```python
inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz}, 
                    data={x: y})
inference.run()

```

**Out:**
```python
500/500 [100%] ██████████████████████████████ Elapsed: 2s | Acceptance Rate: 1.000
```

Conditioned on our priors, Edward has now inferred the most likely posterior distribution given the data. The cool thing is that we can now sample from this posterior distribution and analize:

* The cluster assignment of the data points.
* The uncertainty of our inferred variables.
* Generate new data similar to our original dataset.

### Uncertainty

Let's sample from the posterior.

```python
mu_s = qmu.sample(500).eval()
sigmasq_s = qsigmasq.sample(500).eval()
pi_s = qpi.sample(500).eval()
```

We can for instance get information about the uncertainty of our result by looking the distributions of our variables. By plotting this, we see that a value for $\mu$ is probably close to -1.3 (blue line) and that the model is pretty confident about this result as the width of the distribution is very small.

```python
plt.figure(figsize=(12, 6))
plt.title(r'Distribution of the likelihood of $\mu | X$ in all dimensions')
plt.ylabel(r'$P(\mu|X)$')
plt.xlabel(r'$\mu$')
for i in range(3):
    for j in range(4):
        sns.distplot(mu_s[:, i, j], hist=False)
```

{{< figure src="/img/post-15-mixture_models/pmu.png" width="700px" >}}

### Cluster assignment
We can see the cluster of each datapoint by using our sampled variables from the posterior to create the model from **eq (2.1)** and assign each datapoint to the Gaussian with the highest probability.

```python
np.vstack([pi_s.mean(0)[i] * \
           stats.multivariate_normal(mu_s.mean(0)[i], np.sqrt(sigmasq_s.mean(0))[i]).pdf(y) \
           for i in range(3)]).argmax(axis=0)
```

**Out:**
```python
array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
```

By reversing the order we can look at the accuracy of our unsupervised learning model.

```python
np.vstack([pi_s.mean(0)[i] * \
           stats.multivariate_normal(mu_s.mean(0)[i], np.sqrt(sigmasq_s.mean(0))[i]).pdf(y) \
          for i in range(3)])[::-1].argmax(axis=0).sum() / df.shape[0]
```

**Out:**
```python
0.94
```

We have a accuracy of 94%. Which isn't bad considered the fact that we do unsupervised learning here. 

There is one thing however that still bothers me. We've put prior knowledge in this model, namely the number of clusters! We've instanciated our prior Dirichlet distribution with $\alpha = \vec{1}$. Hereby dictating the number of clusters we can weigh to sum op to 1. Wouldn't it be neat, if we could define a model that is able to choose its number of clusters as it would see fit? 


## Dirichlet Process
The Dirichlet Process is just as the Dirichlet distribution also a distribution of discrete distributions. And this one could help us with a model that is able to define **k** for us.

If we sample from a Dirichlet Process, we'll get a distribution of infinite discrete probabilities $\theta$.

<div class="formula-wrap">
$$H \sim DP(\alpha H_0) \tag{4.0}$$
</div>

The function of the Dirichlet Process is described by:

<div class="formula-wrap">
$$f(\theta) = \sum_{i=1}^{\infty}{\pi_i \cdot \delta(\theta_i)} \tag{4.1}$$
</div>

where

* $H_0$ is an original distribution by our choosing.
* $\pi_i$ are weights that sum to 1.
* $\delta$ is the Dirac delta function. 
* $\theta$ are samples drawn from $H_0$.

$\pi$ are weights summing to one and can be defined by a metaphor called the stick breaking process, where we first sample infinite values $\pi'$ from a Beta distribution parameterized by $1$ and $\alpha$. We start with a whole stick and we will infinitly break of $1 - \pi'_i$ from the stick. The first iteration from the whole stick, the other iterations from the remaining part.



<div class="formula-wrap">
$$\pi_i = \pi' \cdot \prod_{j=1}^{k-1}(1 - \pi'_j) \tag{4.2}$$
</div>

where

<div class="formula-wrap">
$$\pi' \sim Beta(1, \alpha) \tag{4.3}$$
</div>

Ok, that was the formal definition. Let's look at what it actually means to sample from a Dirichlet Process. Below we define a function where we simulate a Dirichlet Process. We simulate it because in real life it is impossible to sample an infinite amount of values. We truncate the amount of samples based on some heuristics, assuming that the sum of $\pi'$ will be approximating with a negligible difference. 

In the plots below we see the influence of $\alpha$ on the sampled distributions. As $H_0$ we choose the familiar Gaussian distribution. In every row we plot three distribution samples. Note that for higher values of $\alpha$, the sampled distribution $H$ will look closer to the original distribution $H_0$.

```python
def dirichlet_process(h_0, alpha):
    """
    Truncated dirichlet process.
    :param h_0: (scipy distribution)
    :param alpha: (flt)
    :param n: (int) Truncate value.
    """
    n = max(int(5 * alpha + 2), 500)  # truncate the values. 
    pi = stats.beta(1, alpha).rvs(size=n)
    pi[1:] = pi[1:] * (1 - pi[:-1]).cumprod()  # stick breaking process
    theta = h_0(size=n)  # samples from original distribution
    return pi, theta
    
    
def plot_normal_dp_approximation(alpha):
    pi, theta = dirichlet_process(stats.norm.rvs, alpha)
    x = np.linspace(-4, 4, 100)
    
    plt.figure(figsize=(14, 4))
    plt.suptitle(r'$\alpha$ = {}'.format(alpha))
    plt.ylabel(r'$\pi$')
    plt.xlabel(r'$\theta$')
    pi = pi * (stats.norm.pdf(0) / pi.max())
    plt.vlines(theta, 0, pi)
    plt.ylim(0, 1)
    plt.plot(x, stats.norm.pdf(x))

for alpha in [.1, 1, 10, 100, 1000]:
    plot_normal_dp_approximation(alpha)
```

{{< figure src="/img/post-15-mixture_models/alpha1.png" width="700px" >}}
{{< figure src="/img/post-15-mixture_models/alpha10.png" width="700px" >}}
{{< figure src="/img/post-15-mixture_models/alpha100.png" width="700px" >}}
{{< figure src="/img/post-15-mixture_models/alpha1000.png" width="700px" >}}
{{< figure src="/img/post-15-mixture_models/alpha10000.png" width="700px" >}}


## Nonparametric model

How can we use the Dirichlet Process in order to nonparametricly (thus without specifying **k**) define the number of clusters in the data? Well we can actually use the same model as defined in **eq (2.1)**. However instead of sampling from a Dirichlet distribution were we define the number of clusters with the parameter $\alpha$, we now sample from a Dirichlet Process and give a prior on the parameter $\alpha$. Parameter $\alpha$ influences how fast $\sum_{i=1}^{\infty}{\pi_i}$ approaches 1. If we accept only the clusters under a certain percentage the model can define by itself what value for $\alpha$ is most likely given the data. Below we can see how $\alpha$ influences the number of clusters when we accept a total probability of 98%.


```python
np.random.seed(95)

acceptance_p = 0.98
plt.figure(figsize=(12, 6))
plt.title(f'Number samples need to have a total probability equal to {acceptance_p}%')
plt.ylabel('$\sum_{i=1}^{\infty}{\pi_i}$')
plt.xlabel('Number of clusters $k$')
def plot_summation(alpha):
    pi, _ = dirichlet_process(stats.norm.rvs, alpha)
    p_total = np.cumsum(pi)
    i = np.argmin(np.abs(p_total - acceptance_p))
    plt.plot(np.arange(pi[:i + 1].shape[0]), p_total[:i + 1])
    return i

k = 0
for alpha in [1, 2, 5, 7, 10]:
    k_i = plot_summation(alpha)
    k = max(k, k_i)
    plt.vlines(k_i, 0, acceptance_p, linestyles='--')
    plt.text(k_i + 0.2, 0.2, f'k = {k_i}', rotation=45)
    
plt.ylim(0, 1)
plt.xlim(0, k + 2)
plt.hlines(acceptance_p, 0, k, linestyles='--')
```

{{< figure src="/img/post-15-mixture_models/asymp.png" width="700px" >}}

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
