+++
date = "2019-09-16"
description = "Variational inference from scratch."
tags = ["algorithm breakdown", "machine learning", "python", "bayesian", "optimization"]
draft = false
keywords = ["algorithm breakdown", "machine learning", "python", "bayesian", "optimization"]
author = "Ritchie Vink"
title = "Variational inference from scratch."
og_image = "/img/post-27-vi-from-scratch/soundboard.png"
+++

{{< figure src="/img/post-27-vi-from-scratch/soundboard.png" >}}

In the posts [Expectation Maximization]({{< ref "expectation_maximization.md" >}}) and [Bayesian inference; How we are able to chase the Posterior]({{< ref "variational_inference.md" >}}), we laid the mathematical foundation of variational inference. This post we will continue on that foundation and implement variational inference in Pytorch. If you are not familiar with the basis, I'd recommend reading these posts to get you up to speed.

## Maximum likelihood baseline
Let's first start with a baseline, so that we can verify that what we will implement actually makes sense.

First, we import the needed packages and then we take an auxiliary dataset from scikit-learn. 

``` python
import numpy as np
import torch
from torch import nn
from sklearn import datasets
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

X = datasets.load_diabetes()['data']
Y = datasets.load_diabetes()['target'][:, None]

# Scale the data.
scaler = StandardScaler()
Y = scaler.fit_transform(Y)

# Go to pytorch world
X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

ds = TensorDataset(X, Y)
dl = DataLoader(ds, shuffle=True, batch_size=batch_size)
```

This data is an off the shelf regression problem. Let's make a simple multi-layer-perceptron and maximize Gaussian likelihood. This is equal to minimizing mean squared error loss. 

<div>$$ \hat{\theta}_{\text{MLE}} = \text{argmax}_\theta \prod_i^nP(x_i|\theta) $$</div>

```python
class MaximumLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    def forward(self, x):
        return self.out(x)
    
epochs = 200
batch_size = 10

m = MaximumLikelihood()
optim = torch.optim.Adam(m.parameters(), lr=0.01)

for epoch in range(epochs):
    for x, y in dl:
        optim.zero_grad()
        y_pred = m(x)
        loss = (0.5 * (y_pred - y)**2).mean()
        loss.backward()
        optim.step()

# Compute MAE loss
with torch.no_grad():
    print(torch.abs(m(X) - Y).mean())
```

``` text
>>> tensor(0.4904)
```

## Variational regression

In the baseline above we've found a function $f:x \mapsto y$. The outputs are point estimates. Now let's consider a model where we want to  obtain the distribution $P(y|x) \propto P(x|y) P(y)$. In variation al inference we accept that we cannot obtain the true posterior $P(y|x)$, but we try to approximate this distribution with another distribution $Q(\theta)$, where $\theta$ are the variational parameters. This distribution we call a variational distribution.

If we choose a factorized (diagonal) Gaussian variational distribution, $Q(\theta)$ becomes $Q(\mu, \text{diag}(\sigma^2))$. We want this distribution to be conditioned to $x$, therefore we define a function $g\_{\theta}: x \mapsto \mu, \sigma$. The function $g\_{\theta}$ will be a neural network that predicts the variational parameters. The total model can thus be described as:

$$ P(y) = \mathcal{N}(0, 1) $$

$$ Q(y|x) = \mathcal{N}(g\_{\theta, \mu}(x), \text{diag}(g\_{\theta, \sigma^2}(x))))$$

Where we set a unit Gaussian prior $P(y)$. 

## Optimization problem
Variational inference is done by maximizing the ELBO (**E**vidence **L**ower **BO**und). Which is often written in a more intuitive form:

$$ \text{argmax}\_{\theta} = E\_{\theta \sim Q}[\underbrace{\log P(D|\theta)}\_{\text{likelihood}}] + D\_{KL}(Q(\theta)||\underbrace{P(\theta)}\_{\text{prior}})$$ 

Where have a likelihood term, in Variational Autoencoders often called reconstruction loss, and the KL-divergence between the prior and the variational distribution. Let's rewrite the KL-divergence term in integral form;

$$ E\_{\theta \sim Q}[\log P(D|\theta)] + \int Q(\theta) \frac{P(\theta)}{Q(\theta)}d\theta $$

Now we observe that we can rewrite the integral form as an expectation $\theta$;

$$ E\_{\theta \sim Q}[\log P(D|\theta)] + E\_{\theta \sim Q}[ \frac{P(\theta)}{Q(\theta)}]d\theta $$

And by applying the log rule $\log\frac{A}{B}=\log A - \log B$, we get;

$$ E\_{\theta \sim Q}[\log P(D|\theta)] + E\_{\theta \sim Q}[\log P(\theta) - \log Q(\theta) ] $$

## Monte Carlo KL-divergence
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
This is all we need for implementing and optimizing this model. Below we'll define the model in Pytorch. By calling the **forward** method we retreive samples from the variational distribution.


```python
class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-4
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var

```

As we have a Gaussian prior, a Gaussian likelihood and a Gaussian variational distribution, we only need to derive the log likelihood function for the Gaussian distribution.

$$ \mathcal{L}(\mu, \sigma, x)= -\frac{n}{2}(2\pi \sigma^2) - \frac{1}{2\sigma^2}\sum\_{i=1}^n(x\_i - \mu)^2$$

```python
def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (y-mu)**2
```

Below we have the `elbo` function. In this function it all comes together. We compute the needed probabilities, and finaly we get an estimate of the expectation by taking the means. In the `det_loss` function we reverse the sign, as all the optimizers in Pytorch are minimizers, not maximizers.

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
epochs = 200
batch_size = 10

m = VI()
optim = torch.optim.Adam(m.parameters(), lr=0.01)

for epoch in range(epochs):
    for x, y in dl:
        optim.zero_grad()
        y_pred, mu, log_var = m(x)
        loss = det_loss(y_pred, y, mu, log_var)
        loss.backward()
        optim.step()
```


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

