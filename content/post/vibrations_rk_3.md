---
author: Ritchie Vink
date: 2017-04-13T13:02:56+02:00
description: Introduction to Runga Kutta.
draft: false
keywords:
- key
- words
tags :
- "Python"
- "Vibrations"
title: Writing a fourth order Runga Kutta solver for a vibrations problem in Python (Part 3)
topics:
- topic 1
type: post
---

This post continues where [part 2]({{< ref "post/vibrations_rk_2.md" >}}) ended. The Runga Kutta algorithm described in last post is only able to solve first order differential equations.

The differential equation (de) for a single mass spring vibrations problem is a second order de.

<div>\[ mu'' + cu' + ku = F\] </div>

Note that in this equation:

u'' = acceleration a

u' = velocity v

u = displacement

Before we can solve it with a Runga Kutta algorithm we must rewrite the base equation to a system of two first order ode's.

Rewrite:

<div>\[ u' = v\] </div>
<div>\[ v' = \frac{1}{m}(F - cv - ku) \] </div>

Now there a set of coupled ode's. Both need to be solved together. The solution of both ode's is determined by:

<div>\[ u_{i+1} = u_{i} + \frac{h}{6}(v_{1} + 2v_{2} + 2v_{3} + v_{4})\] </div>
<div>\[ v_{i+1} = v_{i} + \frac{h}{6}(a_{1} + 2a_{2} + 2a_{3} + a_{4})\] </div>

Where v<sub>1</sub> - v<sub>4</sub> and a<sub>1</sub> - a<sub>4</sub> are determined as described in the table below. Note that:

<div>\[ f(t, u, v) = v' = \frac{1}{m}(F - cv - ku) \] </div>

**Column values**|**t**|**u**|**v**|**a**
:-----:|:-----:|:-----:|:-----:|:-----:
t<sub>1</sub>, u<sub>1</sub>, v<sub>1</sub>,  a<sub>1</sub>|t<sub>i</sub>|t<sub>i</sub> |v<sub>i</sub> |f(t<sub>1</sub>, u<sub>1</sub>, v<sub>1</sub>)
t<sub>2</sub>, u<sub>2</sub>, v<sub>2</sub>,  a<sub>2</sub>|t<sub>i</sub> + h/2|t<sub>i</sub>  + v<sub>1</sub>h/2|v<sub>i</sub> + a<sub>1</sub>h/2|f(t<sub>2</sub>, u<sub>2</sub>, v<sub>2</sub>)
t<sub>3</sub>, u<sub>3</sub>, v<sub>3</sub>,  a<sub>3</sub>|t<sub>i</sub> + h/2|t<sub>i</sub>  + v<sub>2</sub>h/2|v<sub>i</sub> + a<sub>2</sub>h/2|f(t<sub>3</sub>, u<sub>3</sub>, v<sub>3</sub>)
t<sub>4</sub>, u<sub>4</sub>, v<sub>4</sub>,  a<sub>4</sub>|t<sub>i</sub> + h|t<sub>i</sub>  + v<sub>3</sub>h|v<sub>i</sub> + a<sub>3</sub>h|f(t<sub>4</sub>, u<sub>4</sub>, v<sub>4</sub>)

## Code

If we make this in python we get the function below. The input is an array with time values; t.
We set the displacement and velocity at the first value of the times array.
Furthermore we pass the mass, the damping and the spring stiffness of the system.


```python

def runga_kutta_vibrations(t, u0, v0, m, c, k, force):
    """
    :param t: (list/ array)
    :param u0: (flt)u at t[0]
    :param v0: (flt) v at t[0].
    :param m:(flt) Mass.
    :param c: (flt) Damping.
    :param k: (flt) Spring stiffness.
    :param force: (list/ array) Force acting on the system.
    :return: (tpl) (displacement u, velocity v)
    """

    u = np.zeros(t.shape)
    v = np.zeros(t.shape)
    u[0] = u0
    v[0] = v0
    dt = t[1] - t[0]
    
    # Returns the acceleration a
    def func(u, V, force):
        return (force - c * V - k * u) / m

    for i in range(t.size - 1):
        # F at time step t / 2
        f_t_05 = (force[i + 1] - force[i]) / 2 + force[i]

        u1 = u[i]
        v1 = v[i]
        a1 = func(u1, v1, force[i])
        u2 = u[i] + v1 * dt / 2
        v2 = v[i] + a1 * dt / 2
        a2 = func(u2, v2, f_t_05)
        u3 = u[i] + v2 * dt / 2
        v3 = v[i] + a2 * dt / 2
        a3 = func(u3, v3, f_t_05)
        u4 = u[i] + v3 * dt
        v4 = v[i] + a3 * dt
        a4 = func(u4, v4, force[i + 1])
        u[i + 1] = u[i] + dt / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        v[i + 1] = v[i] + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)

    return u, v
```

Lets say we have got a mass spring system with the following parameters:

* Mass			m = 10 kg
* Stiffness		k = 50 N/m
* Viscous damping 	c = 5 Ns/m

On this system we are going to apply a short pulse. The force on the system is described by the following array:

```python

n = 1000
t = np.linspace(0, 10, n)
force = np.zeros(n)

for i in range(100, 150):
    a = np.pi / 50 * (i - 100)
    force[i] = np.sin(a)
```

It is an array of zeros for most of the time values t. Only a small part off the array will be a peak. 
Now we are going to define the parameters of the system, call the runga_kutta_vibrations function and plot the result together with the force.

```python
# Parameters of the mass spring system
m = 10
k = 50
c = 5

u, v = runga_kutta_vibrations(t, 0, 0, m, c, k, force)

# Plot the result
fig, ax1 = plt.subplots()
l1 = ax1.plot(t, v, color='b', label="displacement")
ax2 = ax1.twinx()
l2 = ax2.plot(t, force, color='r', label="force")

lines = l1 + l2
plt.legend(lines, [l.get_label() for l in lines])
plt.show()
```

{{< figure src="/img/post-4/figure_3.png" title="Vibration resulting from a pulse." >}}

If we look at the plot figure. We can see that before the pulse is acting on the system the amplitude is zero. There is no force acting on the system and therefore there is no oscillation. At approximately 1 s, a short puls acts on the system and the system responds. The rest of the plot shows the system oscillating in its natural frequency. 

As you can see, we now can describe a vibrations problem of a singe degree of freedom system nummerically. The Runga Kutta method can be used with predefined force functions as we did (We described the pulse with a sine function, but it can also be used with vibration data resulting from measurements. 

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
