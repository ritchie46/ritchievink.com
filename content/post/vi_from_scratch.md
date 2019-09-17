+++
date = "2019-09-16"
description = "Variational inference from scratch. Ok, we will use autograd."
tags = ["algorithm breakdown", "machine learning", "python", "bayesian", "optimization"]
draft = false
keywords = ["algorithm breakdown", "machine learning", "python", "bayesian", "optimization"]
author = "Ritchie Vink"
title = "Variational inference from scratch."
og_image = "/img/post-27-vi-from-scratch/soundboard.png"
+++

{{< figure src="/img/post-27-vi-from-scratch/soundboard.png" >}}

In the posts [Expectation Maximization]({{< ref "expectation_maximization.md" >}}) and [Bayesian inference; How we are able to chase the Posterior]({{< ref "variational_inference.md" >}}), we laid the mathematical foundation of variational inference. This post we will continue on that foundation and implement variational inference in Pytorch. If you are not familiar with the basis, I'd recommend reading these posts to get you up to speed.

This post we'll model a probablistic layer as output layer of a neural network. This will give us insight in the aleatoric uncertainty (the noise in the data). We will evaluate the results on a fake dataset.

```python
import numpy as np
import torch
from torch import nn
from sklearn import datasets
import matplotlib.pyplot as plt

w0 = 0.125
b0 = 5.
x_range = [-20, 60]

def load_dataset(n=150, n_tst=150):
    np.random.seed(43)

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g**2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    y = (y - y.mean()) / y.std()
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    return y[:, None], x[:, None]

y, x = load_dataset()
```

{{< figure src="/img/post-27-vi-from-scratch/syndata.png" title="Generated data with noise dependent on $X$." >}}

## Maximum likelihood estimate
First we'll model a neural network $g\_{\theta}(x)$ with maximum likelihood estimation. Where we assume a Gaussian likelihood.

$$ y \sim \mathcal{N}(g\_{\theta}(x), \sigma^2) $$

<div>$$ \hat{\theta}_{\text{MLE}} = \text{argmax}_\theta \prod_i^nP(y_i|\theta) $$</div>

``` python
# Go to pytorch world
X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

class MaximumLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    def forward(self, x):
        return self.out(x)
    
epochs = 200
m = MaximumLikelihood()
optim = torch.optim.Adam(m.parameters(), lr=0.01)

for epoch in range(epochs):
    optim.zero_grad()
    y_pred = m(X)
    loss = (0.5 * (y_pred - Y)**2).mean()
    loss.backward()
    optim.step()       
```

If we train this model, we might observe a regression line like below. We are able to predict the expectation of y, but we are not able to make a statement about the noise of our predictions. The outputs are point estimates.

{{< figure src="/img/post-27-vi-from-scratch/fit_mle.png" title="Fitted model with MLE." >}}


## Variational regression

Now let's consider a model where we want to  obtain the distribution $P(y|x) \propto P(x|y) P(y)$. In variational inference, we accept that we cannot obtain the true posterior $P(y|x)$, but we try to approximate this distribution with another distribution $Q(\theta)$, where $\theta$ are the variational parameters. This distribution we call a variational distribution.

If we choose a factorized (diagonal) Gaussian variational distribution, $Q(\theta)$ becomes $Q(\mu, \text{diag}(\sigma^2))$. We want this distribution to be conditioned to $x$, therefore we define a function $g\_{\theta}: x \mapsto \mu, \sigma$. The function $g\_{\theta}$ will be a neural network that predicts the variational parameters. The total model can thus be described as:

$$ P(y) = \mathcal{N}(0, 1) $$

$$ Q(y|x) = \mathcal{N}(g\_{\theta}(x)\_{\mu}, \text{diag}(g\_{\theta}(x)\_{\sigma^2})))$$

Where we set a unit Gaussian prior $P(y)$. 

## Optimization problem
Variational inference is done by maximizing the ELBO (**E**vidence **L**ower **BO**und). Which is often written in a more intuitive form:

$$ \text{argmax}\_{\theta} = E\_{\theta \sim Q}[\underbrace{\log P(D|\theta)}\_{\text{likelihood}}] + D\_{KL}(Q(\theta)||\underbrace{P(\theta)}\_{\text{prior}})$$ 

Where we have a likelihood term (in Variational Autoencoders often called reconstruction loss) and the KL-divergence between the prior and the variational distribution. We are going to rewrite this ELBO definition so that it is more clear how we can use it to optimize the model, we've just defined. 

Let's first rewrite the KL-divergence term in integral form;

$$ E\_{\theta \sim Q}[\log P(D|\theta)] + \int Q(\theta) \frac{P(\theta)}{Q(\theta)}d\theta $$

Now we observe that we can rewrite the integral form as an expectation $\theta$;

$$ E\_{\theta \sim Q}[\log P(D|\theta)] + E\_{\theta \sim Q}[ \frac{P(\theta)}{Q(\theta)}]d\theta $$

And by applying the log rule $\log\frac{A}{B}=\log A - \log B$, we get;

$$ E\_{\theta \sim Q}[\log P(D|\theta)] + E\_{\theta \sim Q}[\log P(\theta) - \log Q(\theta) ] $$

## Monte Carlo ELBO
Deriving those expectations can be some tedious mathematics, or maybe not even possible. Luckily we can get estimates of the mean by taking samples from $Q(\theta)$ and average over those results.

### Reparameterization trick
If we start taking samples from a $Q(\theta)$ we leave the deterministic world, and the gradient can not flow through the model anymore. We avoid this problem by reparameterizing the samples from the distribution.

Instead of sampling directly from the variational distribution;
$$ y \sim Q(\mu, \sigma^2) $$  

We sample from a unit gaussian and recreate samples from the variational distribution. Now the stochasticity of $\epsilon$ is external and will not prevent the flow of gradients.
$$ y = \mu + \sigma \epsilon $$ 

Where

$$ \epsilon \sim \mathcal{N}(0, 1) $$ 

## Implementation
This is all we need for implementing and optimizing this model. Below we'll define the model in Pytorch. By calling the **forward** method we retrieve samples from the variational distribution.


```python
class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var
```

The prior, the likelihood and the varitational distribution are all Gaussian, hence we only need to derive the log likelihood function for the Gaussian distribution.

$$ \mathcal{L}(\mu, \sigma, x)= -\frac{n}{2}(2\pi \sigma^2) - \frac{1}{2\sigma^2}\sum\_{i=1}^n(x\_i - \mu)^2$$

```python
def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (y-mu)**2
```

In the `elbo` function below, it all comes together. We compute the needed probabilities, and last we get an estimate of the expectation (see ELBO definition) by taking the means over a complete batch. In the `det_loss` function, we only reverse the sign, as all the optimizers in Pytorch are minimizers, not maximizers. And that is all we need, the result is an optimization problem with gradients. Hardly a problem at all in times of autograd.


```python
def elbo(y_pred, y, mu, log_var):
    # likelihood of observing y given Variational mu and sigma
    likelihood = ll_gaussian(y, mu, log_var)
    
    # prior probability of y_pred
    log_prior = ll_gaussian(y_pred, 0, torch.log(torch.tensor(1.)))
    
    # variational probability of y_pred
    log_p_q = ll_gaussian(y_pred, mu, log_var)
    
    # by taking the mean we approximate the expectation
    return (likelihood + log_prior - log_p_q).mean()

def det_loss(y_pred, y, mu, log_var):
    return -elbo(y_pred, y, mu, log_var)
```

```python
epochs = 1500

m = VI()
optim = torch.optim.Adam(m.parameters(), lr=0.005)

for epoch in range(epochs):
    optim.zero_grad()
    y_pred, mu, log_var = m(X)
    loss = det_loss(y_pred, Y, mu, log_var)
    loss.backward()
    optim.step()
```

With the fitted model, we can draw samples from the approximate posterior. As we see in the plot below, the aleatoric uncertainty is incorporated in the model.

```python
# draw samples from Q(theta)
with torch.no_grad():
    y_pred = torch.cat([m(X)[0] for _ in range(1000)], dim=1)
    
# Get some quantiles
q1, mu, q2 = np.quantile(y_pred, [0.05, 0.5, 0.95], axis=1)

plt.figure(figsize=(16, 6))
plt.scatter(X, Y)
plt.plot(X, mu)
plt.fill_between(X.flatten(), q1, q2, alpha=0.2)
```

{{< figure src="/img/post-27-vi-from-scratch/fit_vi.png" title="90% credible interval of $P(y|x)$." >}}

## Analytical KL-divergence and reconstruction loss
Above we have implemented ELBO by sampling from the variational posterior. It turns out that for the KL-divergence term, this isn't necessary as there is an analytical solution [for the Gaussian case). (Diederik P. Kingma and Max Welling. 2013.  Auto-encoding variational bayes.](https://arxiv.org/pdf/1802.05814.pdf) included it in Appendix B.

$$  D\_{KL}(Q(\theta)||P(\theta)) = \frac{1}{2}\sum\_{i=1}^n(1+\log \sigma\_i^2 - \mu\_i^2 - \sigma\_i^2) $$

For the likelihood term, we did implement Guassian log likelihood, this term can also be replaced with a similar loss functions. For Gaussian likelihood we can use squared mean error loss, for Bernoulli likelihood we could use binary cross entropy etc. If we do that for the earlier defined model, we can replace the loss function as defined below:

```python
def det_loss(y, y_pred, mu, log_var):    
    reconstruction_error = (0.5 * (y - y_pred)**2).sum()
    kl_divergence = (-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()))

    return (reconstruction_error + kl_divergence).sum()
```

## Final words
This post shows how you can implement variational inference and how it can be utilized to obtain uncertainty estimates over noisy data. In this post, we've only used it to implement an observed variable $y$, but as Variational Autoencoders prove, it can also be used to infer latent variables. The fact that you can combine this with neural networks seems to make it a very powerful and modular. 

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

