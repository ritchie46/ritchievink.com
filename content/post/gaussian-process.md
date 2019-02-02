+++
date = "2019-02-01"
description = "Finally some intuition for Gaussian Processes."
tags = ["machine learning", "python", "gaussian processes", "bayesian"]
draft = false
keywords = ["machine learning", "python", "gaussian processes", "bayesian"]
author = "Ritchie Vink"
title = "An intuitive introduction to Gaussian processes"
og_image = "/img/post-21-gp/waves-banner.png"
+++

{{< figure src="/img/post-21-gp/waves-banner.png" >}}

Christopher Fonnesbeck had a talk about [Bayesian Non-parametric Models for Data Science using PyMC3 on PyCon 2018](https://youtu.be/-sIOMs4MSuA). In this talk he glanced over Bayes' modelling, the neat properties of Gaussian distributions and then quickly turned to the application of Gaussian Processes, a distribution over infinite functions. Wait, but what?! How does a Gaussian represent a function? I did not understand how, but the promise of what these Gaussian Processes representing a distribution over non linear and non parametric
functions really intrigued me and therefore turned into a new subject for a post. This post we'll go, a bit slower than Christopher did, through what a Gaussian Processes are.

In the first part of this post we'll glance over some properties of multivariate Gaussian distributions, then we'll examine how we can use these distributions to express our expected function values and then we'll combine both to find a posterior distribution for Gaussian processes.

## 1. Introduction: The good old Gaussian distribution.
The star of every statistics 101 college, also shines in this post because of it's handy properties. Let's walk through some of those properties to get a feel for them.

A Gaussian is defined by two parameters, the mean $\mu$ and the standard deviation $\sigma$. $\mu$ expresses our expectation of $x$ and $\sigma$ our uncertainty of this expectation.

$$ x \sim \mathcal{N}(\mu, \sigma)$$

A multivariate Gaussian is parameterized by a generalization of $\mu$ and $\sigma$ to vector space. The expected value, i.e. the mean, is now represented by a vector $\vec{\mu}$. The uncertainty is parameterized by a covariance matrix $\Sigma$. 

$$ x \sim \mathcal{N}(\mu, \Sigma) $$

The covariance matrix is actually a sort of lookup table, where we every column and row represent a dimension, and the values are the correlation between the samples of that dimension. For this reason it is symmetrical. As the correlation between dimension i and j is equal to the correlation between dimensions j and i.

$$\Sigma\_{ij} = \Sigma\_{ji}$$

Below I have plotted the gaussian distribution belonging $\mu = [0, 0]$, and $\Sigma = \begin{bmatrix} 1 && 0.6 \\\\ 0.6 && 1 \end{bmatrix}$.

{{< figure src="/img/post-21-gp/mvgs.png" >}}

### Gaussians make baby Gaussians
Gaussian processes are based on Bayesian statistics, wich requires you to compute the conditional and the marginal probability. Now with Gaussian distributions, both result in Gaussian distributions in lower dimensions.

#### Marginal probability
The marginal probability of a multivariate Gaussian is really easy. Officially it is defined by the integral over the dimension we want to marginalize over.

Given this jointly distribution:
<div class="formula-wrap">

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo>,</mo>
  <mi>y</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">N</mi>
  </mrow>
  <mrow>
    <mo>(</mo>
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mrow>
            <mo>[</mo>
            <mtable rowspacing="4pt" columnspacing="1em">
              <mtr>
                <mtd>
                  <msub>
                    <mi>&#x03BC;<!-- μ --></mi>
                    <mi>x</mi>
                  </msub>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <msub>
                    <mi>&#x03BC;<!-- μ --></mi>
                    <mi>y</mi>
                  </msub>
                </mtd>
              </mtr>
            </mtable>
            <mo>]</mo>
          </mrow>
          <mo>,</mo>
          <mrow>
            <mo>[</mo>
            <mtable rowspacing="4pt" columnspacing="1em">
              <mtr>
                <mtd>
                  <msub>
                    <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
                    <mi>x</mi>
                  </msub>
                </mtd>
                <mtd>
                  <msub>
                    <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mi>x</mi>
                      <mi>y</mi>
                    </mrow>
                  </msub>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <msubsup>
                    <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mi>x</mi>
                      <mi>y</mi>
                    </mrow>
                    <mi>T</mi>
                  </msubsup>
                </mtd>
                <mtd>
                  <msub>
                    <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
                    <mi>y</mi>
                  </msub>
                </mtd>
              </mtr>
            </mtable>
            <mo>]</mo>
          </mrow>
        </mtd>
      </mtr>
    </mtable>
    <mo>)</mo>
  </mrow>
</math>
</div>

The marginal distribution can be acquired by just reparameterizing the lower dimensional Gaussian distribution with $\mu\_x$ and $\Sigma\_x$, where normally we would need to do an integral over all possible values of $y$.

$$p(x) = \int{p(x, y)dy} = \mathcal{N}(\mu_x, \Sigma_x)$$

Below we see how integrating, (summing all the dots) leads to a lower dimensional distribution which is also Gaussian.

{{< figure src="/img/post-21-gp/marginal.png" title="Marginal probability of a Gaussian distribution" width="600px" >}}

#### Conditional probability
The conditional probability also leads to a lower dimensional Gaussian distribution. 

<div class="formula-wrap">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>y</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">N</mi>
  </mrow>
  <mo stretchy="false">(</mo>
  <munder>
    <mrow class="MJX-TeXAtom-OP MJX-fixedlimits">
      <munder>
        <mrow>
          <msub>
            <mi>&#x03BC;<!-- μ --></mi>
            <mi>x</mi>
          </msub>
          <mo>+</mo>
          <msub>
            <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>x</mi>
              <mi>y</mi>
            </mrow>
          </msub>
          <msubsup>
            <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
            <mi>y</mi>
            <mrow class="MJX-TeXAtom-ORD">
              <mo>&#x2212;<!-- − --></mo>
              <mn>1</mn>
            </mrow>
          </msubsup>
          <mo stretchy="false">(</mo>
          <mi>y</mi>
          <mo>&#x2212;<!-- − --></mo>
          <msub>
            <mi>&#x03BC;<!-- μ --></mi>
            <mi>y</mi>
          </msub>
          <mo stretchy="false">)</mo>
        </mrow>
        <mo>&#x23DF;<!-- ⏟ --></mo>
      </munder>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mtext>conditional mean</mtext>
    </mrow>
  </munder>
  <mo>,</mo>
  <munder>
    <mrow class="MJX-TeXAtom-OP MJX-fixedlimits">
      <munder>
        <mrow>
          <msub>
            <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
            <mi>x</mi>
          </msub>
          <mo>&#x2212;<!-- − --></mo>
          <msub>
            <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>x</mi>
              <mi>y</mi>
            </mrow>
          </msub>
          <msubsup>
            <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
            <mi>y</mi>
            <mrow class="MJX-TeXAtom-ORD">
              <mo>&#x2212;<!-- − --></mo>
              <mn>1</mn>
            </mrow>
          </msubsup>
          <msubsup>
            <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>x</mi>
              <mi>y</mi>
            </mrow>
            <mi>T</mi>
          </msubsup>
        </mrow>
        <mo>&#x23DF;<!-- ⏟ --></mo>
      </munder>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mtext>conditional covariance</mtext>
    </mrow>
  </munder>
  <mo stretchy="false">)</mo>
</math>
</div>

Below is shown a plot of how the conditional distribution also leads to a Gaussian distribution (in red).

{{< figure src="/img/post-21-gp/cond.png" title="Conditional probability of a Gaussian distribution" width="600px">}}

## 2. Functions described as multivariate Gaussians

A function $f$, is something that maps a specific set (the domain) $X$ to another set (the codomain) $Y$. 

$$ f(x) = y$$

The domain, and the codomain can have a infinite number of values. But let's imagine for now that the domain is finite and is defined by a set $X =$ {$  x\_1, x\_2, \ldots, x\_n$}.

We could define a multivariate Gaussian for all possible values of $f(x)$ where $x \in X$. 

<div class="formula-wrap">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow>
    <mo>(</mo>
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mi>f</mi>
          <mo stretchy="false">(</mo>
          <msub>
            <mi>x</mi>
            <mn>1</mn>
          </msub>
          <mo stretchy="false">)</mo>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>f</mi>
          <mo stretchy="false">(</mo>
          <msub>
            <mi>x</mi>
            <mn>2</mn>
          </msub>
          <mo stretchy="false">)</mo>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mo>&#x22EE;<!-- ⋮ --></mo>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <mi>f</mi>
          <mo stretchy="false">(</mo>
          <msub>
            <mi>x</mi>
            <mi>n</mi>
          </msub>
          <mo stretchy="false">)</mo>
        </mtd>
      </mtr>
    </mtable>
    <mo>)</mo>
  </mrow>
  <mo>&#x223C;<!-- ∼ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">N</mi>
  </mrow>
  <mrow>
    <mo>(</mo>
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mrow>
            <mo>(</mo>
            <mtable rowspacing="4pt" columnspacing="1em">
              <mtr>
                <mtd>
                  <msub>
                    <mi>&#x03BC;<!-- μ --></mi>
                    <mn>1</mn>
                  </msub>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <msub>
                    <mi>&#x03BC;<!-- μ --></mi>
                    <mn>2</mn>
                  </msub>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <mo>&#x22EE;<!-- ⋮ --></mo>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <msub>
                    <mi>&#x03BC;<!-- μ --></mi>
                    <mi>n</mi>
                  </msub>
                </mtd>
              </mtr>
            </mtable>
            <mo>)</mo>
          </mrow>
          <mo>,</mo>
          <mrow>
            <mo>(</mo>
            <mtable rowspacing="4pt" columnspacing="1em">
              <mtr>
                <mtd>
                  <msub>
                    <mi>&#x03C3;<!-- σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mn>11</mn>
                    </mrow>
                  </msub>
                </mtd>
                <mtd>
                  <msub>
                    <mi>&#x03C3;<!-- σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mn>12</mn>
                    </mrow>
                  </msub>
                </mtd>
                <mtd>
                  <mo>&#x22EF;<!-- ⋯ --></mo>
                </mtd>
                <mtd>
                  <msub>
                    <mi>&#x03C3;<!-- σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mn>1</mn>
                      <mi>n</mi>
                    </mrow>
                  </msub>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <msub>
                    <mi>&#x03C3;<!-- σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mn>21</mn>
                    </mrow>
                  </msub>
                </mtd>
                <mtd>
                  <msub>
                    <mi>&#x03C3;<!-- σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mn>22</mn>
                    </mrow>
                  </msub>
                </mtd>
                <mtd>
                  <mo>&#x22EF;<!-- ⋯ --></mo>
                </mtd>
                <mtd>
                  <msub>
                    <mi>&#x03C3;<!-- σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mn>2</mn>
                      <mi>n</mi>
                    </mrow>
                  </msub>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <mo>&#x22EE;<!-- ⋮ --></mo>
                </mtd>
                <mtd>
                  <mo>&#x22EE;<!-- ⋮ --></mo>
                </mtd>
                <mtd>
                  <mo>&#x22F1;<!-- ⋱ --></mo>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <msub>
                    <mi>&#x03C3;<!-- σ --></mi>
                    <mi>n</mi>
                  </msub>
                </mtd>
                <mtd>
                  <msub>
                    <mi>&#x03C3;<!-- σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mi>n</mi>
                      <mn>2</mn>
                    </mrow>
                  </msub>
                </mtd>
                <mtd />
                <mtd>
                  <msub>
                    <mi>&#x03C3;<!-- σ --></mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mi>n</mi>
                      <mi>n</mi>
                    </mrow>
                  </msub>
                </mtd>
              </mtr>
            </mtable>
            <mo>)</mo>
          </mrow>
        </mtd>
      </mtr>
    </mtable>
    <mo>)</mo>
  </mrow>
</math>
</div>

Here the $\mu$ vector contains the expected values for $f(x)$. If we are certain about a function, $f(x) \approx y$ and the $\sigma$ values would all be close to zero. Each time we sample from this distribution we'll get a function to $f$. It is important to note that **each finite value of x is another dimension** in the multivariate Gaussian. 

Let's give an example in Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


n = 50
x = np.linspace(0, 10, n)

# Define the gaussian with mu = sin(x) and negligible covariance matrix
norm = stats.multivariate_normal(mean=np.sin(x), cov=np.eye(n) * 1e-6)
plt.figure(figsize=(16, 6))

# Taking a sample from the distribution and plotting it.
plt.plot(x, norm.rvs())
```

{{< figure src="/img/post-21-gp/sing.png" title="A function sampled from a Gaussian distribution.">}}

We can also define a distribution of functions with $\vec{\mu} = 0$ and $\Sigma = I$ (the identity matrix). Now we do have some uncertainty, because the diagonal of $\Sigma$ has a standard deviation of 1. Second thing to note is that all values of $f(x)$ are completely unrelated to each other, because the correlation between all dimensions is zero. In the example below we draw 3 functions from this distribution.

```python
# Define the gaussian with mu = 0 and a covariance matrix w/o any correlation,
# but w/ uncertainty
norm = stats.multivariate_normal(mean=np.zeros(n), cov=np.eye(n))
plt.figure(figsize=(16, 6))

# Taking 3 sample from the distribution and plotting it.
[plt.plot(x, norm.rvs()) for _ in range(3)]
```

{{< figure src="/img/post-21-gp/draw3.png" title="Three functions sampled from $\mathcal{N}(\vec{0}, I)$ " >}}

## 3. Controlling the functions with kernels

As you can see we've sampled different functions from our multivariate Gaussian. In fact we can sample an infinite amount of functions from this distribution. And if we woul want a more fine grid of values, we could also reparameterize our Gaussian to include a new set of $X$.

However, these functions we sample now are pretty random and maybe doesn't seem likely for some real world processes. Let's say we only want to sample functions that are **smooth**. Besides that smoothness looks very slick, it is also a reasonable assumption. Values that are close to each other in domain $X$, will also be mapped close to each other in the codomain $Y$. We could construct such functions by defining the covariance matrix $\Sigma$ in such a way that values close to
each other have larger correlation than values with a larger distance between them.

### Squared exponential kernel
A way to create this new covariance matrix is by using a **squared exponential kernel**. This kernel does nothing more than assigning high correlation values to $x$ values close toghether. 

$$k(x, x') = exp(- \frac{(x-x')^2}{2l^2})$$

If we now define a covariance matrix $\Sigma = k(x, x)$, we sample much smoother functions.

```python
def kernel(m1, m2, l=1):
    return np.exp(- 1 / (2 * l**2) * (m1[:, None] - m2)**2)

n = 50
x = np.linspace(0, 10, n)
cov = kernel(x, x, 0.44)

norm = stats.multivariate_normal(mean=np.zeros(n), cov=cov)
plt.figure(figsize=(16, 6))
[plt.plot(x, norm.rvs()) for _ in range(3)]
```

{{< figure src="/img/post-21-gp/smoes.png" title="Three functions sampled from $\mathcal{N}(\vec{0}, k(x, x))$ " >}}

## Gaussian Processes
Ok, now we have enough information to get started with Gaussian processes. GPs are use to define a prior distribution of the functions that could explain our data. And conditional on the data we have observed we can find a posterior distribution of functions that fit the data. Let's say we have some **known function outputs $f$** and we want to infer **new unknown data points $f\_\*$**. With the kernel we've described above, we can define the joint distribution $p(f, f\_\*)$. 

<div class="formula-wrap">
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow>
    <mo>(</mo>
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mi>f</mi>
        </mtd>
      </mtr>
      <mtr>
        <mtd>
          <msub>
            <mi>f</mi>
            <mo>&#x2217;<!-- ∗ --></mo>
          </msub>
        </mtd>
      </mtr>
    </mtable>
    <mo>)</mo>
  </mrow>
  <mo>=</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mi class="MJX-tex-caligraphic" mathvariant="script">N</mi>
  </mrow>
  <mrow>
    <mo>(</mo>
    <mtable rowspacing="4pt" columnspacing="1em">
      <mtr>
        <mtd>
          <mrow>
            <mo>[</mo>
            <mtable rowspacing="4pt" columnspacing="1em">
              <mtr>
                <mtd>
                  <mi>&#x03BC;<!-- μ --></mi>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <msub>
                    <mi>&#x03BC;<!-- μ --></mi>
                    <mo>&#x2217;<!-- ∗ --></mo>
                  </msub>
                </mtd>
              </mtr>
            </mtable>
            <mo>]</mo>
          </mrow>
          <mo>,</mo>
          <mrow>
            <mo>[</mo>
            <mtable rowspacing="4pt" columnspacing="1em">
              <mtr>
                <mtd>
                  <mi>K</mi>
                </mtd>
                <mtd>
                  <msub>
                    <mi>K</mi>
                    <mo>&#x2217;<!-- ∗ --></mo>
                  </msub>
                </mtd>
              </mtr>
              <mtr>
                <mtd>
                  <msubsup>
                    <mi>K</mi>
                    <mo>&#x2217;<!-- ∗ --></mo>
                    <mi>T</mi>
                  </msubsup>
                </mtd>
                <mtd>
                  <msub>
                    <mi>K</mi>
                    <mrow class="MJX-TeXAtom-ORD">
                      <mo>&#x2217;<!-- ∗ --></mo>
                      <mo>&#x2217;<!-- ∗ --></mo>
                    </mrow>
                  </msub>
                </mtd>
              </mtr>
            </mtable>
            <mo>]</mo>
          </mrow>
        </mtd>
      </mtr>
    </mtable>
    <mo>)</mo>
  </mrow>
</math>
</div>

So now we have a joint distribution, which we can fairly easy compute for any new $x\_\*$ we are interested in. Assuming standardized data, $\mu$ and $\mu\_\*$ can be initialized as $\vec{0}$. And all the covariance matrices $K$ can be computed for all the data points we're interested in.

* $K = k(x, x)$
* $K\_\* = k(x, x\_\*)$
* $K\_{\*\*} = k(x\_\*, x\_\*)$

Because this distribution only forces the samples to be smooth functions, there should be infinitely many functions that fit $f$.

And now comes the most important part. **We know the values of $f$, so we are interested in the conditional probabilty $p(f\_\*| f)$**. Which we can calculate in closed form!

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
