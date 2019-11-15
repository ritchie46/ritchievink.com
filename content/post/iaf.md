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

Where $f(\cdot)$ is an invertible transformation $f: \mathbb{R}^m \mapsto \mathbb{R}^m$. In order to have a tractable distribution, the determinant Jacobian $\left| \det \frac{\partial{f}}{\partial{z}} \right|$ should be cheaply computable. 

This tractability constraint, isn't easily met, and therefore there is now a field of research that focusses on transformations $f(\cdot)$ that have tractable determinant Jacobians. With radial/ planar flows tractability is achieved by applying the [matrix determinant lemma](https://en.wikipedia.org/wiki/Matrix_determinant_lemma).

## Autoregression
Autoregressive functions do have tractable determinant jacobians. Let $z'$ be our our transformed random variable $f(z) = z'$. An autoregressive function would transform every dimension $i$ by:

\begin{eqnarray}
z'\_i = f(z\_{1:i})
\end{eqnarray}

The Jacobian matrix $J = \frac{\partial{z'}}{\partial{z}}$ will then be lower triangular. Which is cheaply computed by the product of the diagonal values:

\begin{eqnarray}
\det J = \prod\_{i=1}^D J\_{ii}
\end{eqnarray}

Tractable determinant Jacobians are thus possible by using autoregressive transformations. First time I read *this paper*<sup>[3]</sup>, I wondered how we could ensure a neural network would be an autoregressive transformation without using RNNs. It turns out you can mask the connection of an autoencoder in such a way that that the output $x_i$ is only connected to the previous inputs $x\_{i:k}$. This was shown in the MADE<sup>[4]</sup> paper, which was the topic of my previous post [Distribution estimation with Masked Autoencoders]({{< ref "made.md" >}}). *Note: In the MADE post we modelled $x\_i = f(x\_{1:i-1})$, which would lead to a lower triangular Jacobian with zeros on the diagonal, and thus leading to $\det J = 0$. Therefore we will model a transformation $x\_i = f(x\_{1:i})$.

## Autoregressive transformation
The radial/ planar transformation are effectively a single layer with one activation node, and therefore not very complex relation between the dimensions of $z$. With autoregressive transformations, $z_i$ is dependent on $z\_{1:i-1}$, and thus connected to all lower dimensions of $i$. Let's define such a transformation. Let $z' = f(z)$, and the first dimensions of the transformed variable be defined by:

\begin{eqnarray}
z &\sim& \mathcal{N}(0, I) &\qquad \tiny{\text{base distribution: } Q(z) } \\\\ 
z'_0 &=& \mu_0 + \sigma\_0 \odot z\_0 \label{art0}
\end{eqnarray}

Where $\epsilon \in \mathbb{R}^D$. Effectively meaning that the first dimension of the transformation is randomly sampled. The other dimensions $i \gt 0$ are computed by:

<div>
\begin{eqnarray}
z'_i = \mu_i(z'_{1:i-1}) + \sigma_i(z'_{1:i-1}) \cdot z_i  \label{art}
\end{eqnarray}
</div>

Eq. \eqref{art} clearly depicts the downside of this transformation. Applying the transformation has a complexity of $\mathcal{O}(D)$. Which can become rather expensive as we will apply a flow of $k$ transformations. Actually sampling from this distribution would be proportional to $D \times k$ operations.


## Inverse Autoregressive Transformation
We can inverse the operations defined in eq. \eqref{art0} and eq. \eqref{art}.

<div>
\begin{eqnarray}
z_0 &=& \frac{z'_0 - \mu_0}{\sigma_0} \label{iart0} \\
z_i &=& \frac{z'_i - \mu(z'_{1:i-1}) }{\sigma(z'_{1:i-1})} \label{iart}
\end{eqnarray}
</div>

*Note that due to the inversion $z' = f(z)$ now is $z = f(z')$!*

The inverse is only dependant on $z'$ (in this case, the value before the transformation), and can thus easily be parallelized. Besides cheap sampling we also need to derive the determinant Jacobian. The Jacobian lower triangular, which means we only need to compute the partial derivatives of the diagonal to compute the determinant.

<div>
\begin{equation}
\frac{d{\bf z}}{d{\bf z'}} =
\begin{bmatrix}
\frac{z_0}{z'_0 } & 0 & \dots & 0 \\
\frac{z_0}{z'_1 } & \frac{z_1}{z'_1} & \dots  & 0 \\
\vdots & \ddots & \ddots & \vdots \\
\frac{z_0}{z'_D } & \dots & \frac{\partial z_{D-1}}{\partial z'_D} &  \frac{\partial z_{D}}{\partial z'_D} \\
\end{bmatrix}
\end{equation}
</div>

The diagonal is <span>$\\{{ \frac{z_0}{z'_0}, \frac{z_1}{z'_1}, \dots, \frac{z_D}{z'_D }   \\}}$</span>. 

As $\mu(\cdot)$ and $\sigma(\cdot)$ are not dependant of $z'_i$, the partial derivatives evaluate to <span> $ \\{{  \frac{1}{\sigma\_0}, \frac{1}{\sigma_1(z'\_1)}, \dots, \frac{1} {\sigma\_D(z'\_{1:D-1}) } \\}}$ </span>, and thus we also have a tractable determinant Jacobian;


<div>
\begin{eqnarray}
\det \left| \frac{d{\bf z}}{d{\bf z'}} \right| &=& \prod_{i=1}^D  \frac{1}{\sigma_i(z_{1:i-1})} \\
\log \det \left| \frac{d{\bf z}}{d{\bf z'}} \right| &=& \sum_{i=1}^D - \log \sigma_i(z_{1:i-1}) \\
\end{eqnarray}
</div>

## Inverse Autoregressive Flow
That wraps all the ingredients we need to build a tractable flow. We will use the inverse functions defined in eq. $\eqref{iart0}$ and eq. $\eqref{iart}$. Let $t$ be one of $k$ flows. Instead of modeling $\mu_t(\cdot)$ and $\sigma_t(\cdot)$ directly, we will define two functions $s_t = \frac{1}{\sigma_t(\cdot)}$, and $m_t = \frac{-\mu_t(\cdot)}{\sigma_t(\cdot)}$. These will be modeled by **autoregressive neural networks**.


<div>
\begin{eqnarray}
z_t &=& \frac{z_{t-1} - \mu_t(z_{t-1}) }{\sigma_t(z_{t-1})} \\
    &=& \frac{z_{t-1}}{\sigma_t(z_{t-1})} - \frac{\mu_t(z_{t-1})) }{\sigma_t(z_{t-1}) } \\
    &=& z_{t-1} \odot {\bf s}_t(z_{t-1})  + {\bf m}_t (z_{t-1})
\end{eqnarray}
</div>



## References
&nbsp; [1] Rezende & Mohammed (2016, Jun 14) *Variational Inference with Normalizing Flows*. Retrieved from https://arxiv.org/pdf/1505.05770.pdf <br>
&nbsp; [2] Adam Kosiorek (2018, Apr 3) *Normalizing Flows*. Retrieved from http://akosiorek.github.io/ml/2018/04/03/norm_flows.html <br>
&nbsp; [3] Kingma, Salimans, Jozefowicz, Chen, Sutskever, & Welling (2016, Jun 15) *Improving Variational Inference with Inverse Autoregressive Flow*. Retrieved from https://arxiv.org/abs/1606.04934 <br>
&nbsp; [4] Germain, Gregor & Larochelle (2015, Feb 12) *MADE: Masked Autoencoder for Distribution Estimation*. Retrieved from https://arxiv.org/abs/1502.03509 <br>


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
