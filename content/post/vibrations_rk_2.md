---
author: Ritchie Vink
date: 2017-04-13T13:02:56+02:00
description: Introduction to Runga Kutta.
draft: false
tags :
- Python
- Vibrations
- Engineering
title: Writing a fourth order Runga Kutta solver for a vibrations problem in Python (Part 2)
topics:
- topic 1
type: post
---

This post continues where [part 1]({{< ref "post/vibrations_rk.md" >}}) ended. In order to increase the accuracy of our function solver we are going to use a 4th order Runga Kutta algorithm. The basics are the same as with the Euler method. However the dy part of the 4th order method is more accurately computed.

## Definition

The incremental values of this method are defined as:

<div>\[ y_{n+1} = y_{n} + \frac{h}{6}(k_{1} + 2k_{2} +2k_{3} + k_{4})\] </div>
<div>\[ t_{n+1} = t_{n} + h \] </div>

With the factors k<sub>1</sub> - k<sub>4</sub> being:

<div>\[k_{1} = f(t_{n}, y_{n}) \]</div>
<div>\[k_{2} = f(t_{n} + \frac{h}{2}, y_{n}) + \frac{h}{2}k_{1}) \]</div>
<div>\[k_{3} = f(t_{n} + \frac{h}{2}, y_{n}) + \frac{h}{2}k_{2}) \]</div>
<div>\[k_{4} = f(t_{n} + h, y_{n}) + hk_{3}) \]</div>

The function f is again the derivative of y.

<div>\[y'= f(t, y)\]</div>

## Code

Lets see how this looks in Python.

```python
def runga_kutta_4(t, f, initial=(0, 0)):
    """
    Runga Kutta nummerical method.

    Computes dy and adds it to previous y-value.
    :param t: (list/ array) Time values of function f.
    :param f: (function)
    :param initial:(tpl) Initial values
    :return: (list/ array) y values.
    """
    # step size
    h = t[1] - t[0]
    y = np.zeros((t.size, ))

    t[0], y[0] = initial

    for i in range(t.size - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h * 0.5, y[i] + 0.5 * k1)
        k3 = h * f(t[i] + h * 0.5, y[i] + 0.5 * k2)
        k4 = h * f(t[i + 1], y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y
```

As you can see above the input values are the same as with euler function. Again the starting point of the curve must be set by passing the initial values for t<sub>0</sub> and y<sub>0</sub>. 

By plotting all 3 curves, the Euler method, the 4th order Runga Kutta method and the function y = e<sup>t</sup> than we find that the curve of the Runga Kutta method is plotted above the curve of the solution and thus has far more acccuracy than the Euler method.

{{< figure src="/img/post-4/figure_2.png" title="Euler method, Runga Kutta method and solution." >}}

In the [next post]({{< ref "post/vibrations_rk_3.md" >}}) we are going to apply the Runga Kutta solution to the vibrations problem. 

[READ HERE FOR THE NEXT PART!]({{< ref "post/vibrations_rk_3.md" >}})

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
