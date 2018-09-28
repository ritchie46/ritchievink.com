+++
date = "2018-09-26"
description = ""
tags = ["machine learning", "python", "algorithm breakdown", "time series"]
draft = false
author = "Ritchie Vink"
title = "Algorithm Breakdown: AR, MA and ARIMA models."
og_image = "/img/post-18-arima/random-walk.png"
+++

{{< figure src="/img/post-18-arima/random-walk.png" >}}

Time series are a quite unique topic within machine learning. In a lot of problems the dependent variable $y$, i.e. the thing we want to predict, is dependent on very clear inputs, such as pixels of an image, words in a scentence, properties of a persons buying behavior, etc. In time series these indepent variables are often not known. For instance in stock markets we don't have a clear independent
set of variables where we can fit a model on. Are stock markets dependent on properties of a company, or properties of a country, or are they dependent on the sentiment in the news? Surely we can try to find ar relation between those indepedent variables and stock market results, and maybe we are able to find some good models that map those relations. Point is that those relations are not very clear, nor is the independent data easily obtainable.

A common approach to model time series is to regard the label at current time step $X\_{t}$ as a variable dependent on previous time steps $X\_{t-k}$. We thus analyze the time series on nothing more than the time series. 

One of the most used models when handling time series are ARIMA models. In this post we'll explore how these models are defined and we are going to develop such a model in Python with nothing else but the numpy package.

## Stochastic series
ARIMA models are actually a combination of two, (or three if you count differencing as a model) processes that are able to generate series data. Those two models are based on an Auto Regressive (AR) process and a Moving Average process. Both AR and MA processes are stochastic processes. Stochastic means that the values come from a random probability distribution, which can be analized statisticly but may not be predicted precisely. In other words, both processes have some
uncertainty. 

## White noise
Let's look at a very simple stochastic process called white noise. White noise can be drawn from many kind of distributions, here we draw from a univariate gaussian.

$$ \epsilon \sim N(0, 1) $$

``` python
# fix some imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

n = 500
fig, ax = plt.subplots(1,2, figsize=(16, 6), gridspec_kw={'width_ratios':[3, 1]})
eps = np.random.normal(size=n)
ax[0].plot(eps)
sns.distplot(eps, ax=ax[1])
```

{{< figure src="/img/post-18-arima/white-noise.png" title="White noise signal." >}}
This process is completely random, though we are able to infer some properties from this series. By making a plot of the distribution we can assume that these variable come from a single normal distribution with zero mean and unit variance. Our best guess for any new variables value would be 0. A better model for this process doesn't exist as every new draw from the distribution is completely random and independent of the previous values. White noise is actually something we want
to see on the residuals after we've defined a model. If the residuals follow a white noise pattern, we can be certain that we've declared all the possible variance.

## MA process
A moving average process is actually based on this white noise. It is defined as a weighted average of the previous white noise values.

$$ X\_t = \mu + \epsilon\_t + \sum\_{i=1}^{q}{\theta\_i \epsilon\_{t-i}} $$

Where $\theta$ are the parameters of the process and $q$ is the order of the process. With order we mean how many time steps $q$ we should include in the weighted average.

Let's simulate an MA process. For every time step $t$ we take the $\epsilon$ values up to $q$ time steps back. First we create a function that given an 1D array creates a 2D array with rows that look $q$ indices back.

``` python
def lag_view(x, order):
    """
    For every value X_i create a row that lags k values: [X_i-1, X_i-2, ... X_i-k]
    """
    y = x.copy()
    # Create features by shifting the window of `order` size by one step.
    # This results in a 2D array [[t1, t2, t3], [t2, t3, t4], ... [t_k-2, t_k-1, t_k]]
    x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])
    
    # Reverse the array as we started at the end and remove duplicates.
    # Note that we truncate the features [order -1:] and the labels [order]
    # This is the shifting of the features with one time step compared to the labels
    x = np.stack(x)[::-1][order - 1: -1]
    y = y[order:]

    return x, y
```
In the function above we create that 2D matrix. We also truncate the input and output array so that all rows have lagging values. If we call this function on an array ranging from 0 to 10, with order 3, we get the following output.

``` text
>>> lag_view(np.arange(10), 3)[0]

array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4],
       [3, 4, 5],
       [4, 5, 6],
       [5, 6, 7],
       [6, 7, 8]])
```

Now we are able to easily take a look at different lags back in time, let's simulate 3 different MA processes with order $q=1$, $q=6$ and $q=11$.

``` python
def ma_process(eps, theta):
    theta = np.array([1] + list(theta))[:, None]
    eps_q, _ = lag_view(eps, len(theta))
    return eps_q @ theta

fig = plt.figure(figsize=(18, 4 * 3))
a = 310
for i in range(0, 11, 5):
    a += 1
    theta = np.random.uniform(0, 1, size=i + 1)
    plt.subplot(a)
    plt.title(f'$\\theta$ = {theta.round(2)}')
    plt.plot(ma_process(eps, theta))
```


{{< figure src="/img/post-18-arima/ma-signal.png" title="MA processes from different orders." >}}

Note that I've chosen positive values for $\theta$ which isn't required. An MA process can have both positive and negative values for $\theta$. In the plots above can be seen that when the order of $MA(q)$ continues the values are longer correlated with previous values. Actualy, because the process is a weighted average of the $\epsilon$ values until lag $q$, the correlation drops after this lag. Based on this property we can make an educated guess on the which order an $MA(q)$
process is. This is great, because it is very hard to infer the order by looking at the plots directly. 

## Autocorrelation
When a value $X\_t$ is correlated with a previous value $X\_{t-k}$, this is called autocorrelation. The autocorrelation function is defined as:

$$ACF(X\_t, X\_{t-k}) = \frac{E[(X\_t - \mu\_t)(X\_{t-k} - \mu\_{t-k})]}{\sigma\_t \sigma\_{t-k}}$$

Nummerically whe can approximate it by determining the correlation between different arrays, namely $X\_t$ and array $X\_{t-k}$. By doing so whe do need to truncate both array by $k$ elements in order to maintain equal length.

``` python
def pearson_correlation(x, y):
    return np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std())

def acf(x, lag=40):
    """
    Determine auto correlation factors.
    :param x: (array) Time series.
    :param lag: (int) Number of lags.
    """
    return np.array([1] + [pearson_correlation(x[:-i], x[i:]) for i in range(1, lag)])

lag = 40
# Determine the ACF of an ma(1) and an ma(2) process.
acf_1 = acf(ma_process(eps, [1]), lag)
acf_2 = acf(ma_process(eps, [0.2, -0.3, 0.8]), lag)
```

Above we have applied the ACF function on an $MA(1)$ and an $MA(2)$ process with different weights $\theta$. The weights for the models are:

* MA(1): [1]
* MA(2): [0.2, -0.3, 0.8]

Below we plot the result of both acf functions. We've also defined a helper function `bartletts_formula` which we use as a null hypothesis to determine if the correlation coefficients we've found are significant and not a statistical fluke. With this function we determine a confidence interval $CI$.

$$CI = \pm z\_{1-\alpha/2} \sqrt{\frac{1+2 \sum\_{1 < i< h-1 }^{h-1}r^2\_i}{N}} $$

where $ z\_{1-\alpha/2} $ is the quantile function from the normal distribution. Quantile functions are the inverse of the cumulative distribution function and can be called with `scipy.stats.norm.ppf`. Any values outside of this confidence interval (below plotted in orange) are statistical significant.

``` python
def bartletts_formula(acf_array, n):
    """
    Computes the Standard Error of an acf with Bartlet's formula
    Read more at: https://en.wikipedia.org/wiki/Correlogram
    :param acf_array: (array) Containing auto correlation factors
    :param n: (int) Length of original time series sequence.
    """
    # The first value has autocorrelation with it self. So that values is skipped
    se = np.zeros(len(acf_array) - 1)
    se[0] = 1 / np.sqrt(n)
    se[1:] = np.sqrt((1 + 2 * np.cumsum(acf_array[1:-1]**2)) / n )
    return se


# Statistical significance for confidence interval
alpha = 0.05

plt.figure(figsize=(16, 4 * 2))
plt.suptitle('Correlogram')
a = 210
for array in [acf_1, acf_2]:
    a += 1
    plt.subplot(a)
    plt.vlines(np.arange(lag), 0, array)
    plt.scatter(np.arange(lag), array, marker='o')
    plt.xlabel('lag')
    plt.ylabel('auto correlation')

    # Determine confidence interval
    ci = stats.norm.ppf(1 - alpha / 2.) * bartletts_formula(array, len(eps))
    plt.fill_between(np.arange(1, ci.shape[0] + 1), -ci, ci, alpha=0.25)
```

{{< figure src="/img/post-18-arima/acf.png" >}}

As we mentioned earlier, these plots help us infer the order of the $MA(q)$ model. In both plots we can see a clear cut off in significant values. Both plots start with an auto correlation of 1. This is the auto correlation at lag 0. The second value is the auto correlation at lag 1 etc. The first plot the cut off is after 1 lag and in the second plot the cut off is at lag 3. So in our artificial data set we are able to determine the order of different $MA(q)$ models by looking
at the ACF plot!

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
