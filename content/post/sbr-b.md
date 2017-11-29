---
author: Ritchie Vink
date: 2017-05-07T20:27:28+02:00
description: This post eloborates on two formula's desbribed in the SBR guideline.
draft: false
keywords:
- Guidelines
- SBR-B
tags:
- Vibrations
- Python
- Engineering
title: What should be explained in the Dutch SBR-B Guideline!
topics:
- topic 1
type: post
---
The Dutch SBR guideline is intended to help you process vibration data and help you determine when a vibration signal can cause discomfort to persons. It seems to me however, that the SBR-B guideline does not have the intention to be understood. They seem to help you by making a super abstract of scientific papers and by giving you a few keywords so you can Google it yourself.

This post will elaborate on two formula's given in the guideline. It took me a while to find out what they really ment. But thanks to some help from my colleague [Lex van der Meer](https://www.linkedin.com/in/lex-van-der-meer-phd-0546906/), and some papers he found, I could make sense of it.

<br/>

## Problem

The guideline gives two formula's that should be used to turn your raw data from a vibrations measurement into design values for further processing. 
Roughly translated, in about the same amount of words, it says: 

*The vibration data needs to be weighed by:*

<div>\[|H_a(f)| = \frac{1}{v_0} \cdot \frac{1}{\sqrt{1 + (f_0/f)^2}}\]</div>


*In which:*

|                   |                  |
|-------------------|------------------|
| f                 | frequency in Hz  |
| f<sub>0</sub>     | 5.6 Hz           |
| v<sub>0</sub>     | 1 mm/s           |

<br/>

*From the result of the formula above the effective value is determined by:*


<div>\[v_{eff}(t) = \sqrt{ \frac{1}{\tau} \int_0^tg(\xi)v^2(t-\xi)d\xi}\]</div>

<br/>

*In which:*


<div>\[\tau = 0.125 s\]</div>
<div>\[g(\xi) = e^{-\xi/\tau}\]</div>

<br/>

In the first formula a frequeny in Hz is required. They do not specify which frequency. I thought my measured vibration signal had infinity frequencies, or at least more than one? In the second formule we integrate the output of the first formula over d&xi;, again not specifying what &xi; is. That's about all the attention the guideline spents on it. Well good luck with that!

<br/>

## Time signal

First we need some 'measured' data. In the following code snippet some fake data is created by adding 5 sine waves. The sine waves' amplitudes and frequency are random.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
t = np.linspace(0, 2.5, 500)
vibrations = np.zeros_like(t)

for i in range(5):
    vibrations += np.sin((10 * np.random.rand())**2 * 2 * np.pi *t) * 10 * np.random.rand()

fig = plt.figure(figsize=(12, 6))
plt.title("Measured vibration")
plt.ylabel("v [mm/s]")
plt.xlabel("t [s]")
plt.plot(t, vibrations)
plt.show()

```

{{< figure src="/img/post-6-sbr/fig_1.png" title="Vibration signal" >}}

Above is the fake vibration data we've just created shown. Now we have a vibrations signal we can take apart the two give formula's and see what their use is.

</br>

## Weighted signal

The first formula is used to weight the signal by the frequencies that are most likely to cause hindrance. This becomes more clear if we plot the function first. Let's plot the result of the function in the frequency range 1 - 100 Hz. Note that 1 / v<sub>0</sub> = 1, thus let's ignore that.


```python
f = np.arange(0, 100)
f0 = 5.6
y = 1 / np.sqrt(1 + (f0/f)**2)

plt.plot(f, y)
plt.xlabel("f [Hz]")
plt.show()
```

{{< figure src="/img/post-6-sbr/fig_3.png" title="Weight function" >}}

By plotting <span>\\(\frac{1}{\sqrt{1 + (f/5.6)^2}}\\)</span> in the range 1-100 Hz we get the curve shown above. Apparently the lower frequencies will be weighted much more than the higher ones, as the the curve will tend to go to zero by increasing the frequency. This weighting is done by multiplying the original signal with this function.

What the guideline does not mention is that before you are able to do so, you must convert the signal from the time domain to the frequency domain. Well, this can be done by taking the Fast Fourier Transform! [You can read more about this in the last post.]({{< ref "post/understanding-fft.md" >}})

By taking the FFT we retrieve the frequency bins. Each bin can be multiplied with the weights function. Shown below is the frequency spectrum and the curve that will scale down this spectrum.

```python
vibrations_fft = np.fft.fft(vibrations)
T = t[1] - t[0]
N = t.size

f = np.linspace(0, 1 / T, N)
a = N // 2

weight = 1 / np.sqrt(1 + (5.6/f)**2)
vibrations_fft_w = weight * vibrations_fft

fig = plt.figure(figsize=(14, 4))
plt.subplot(121)
plt.title("Frequency spectrum")
plt.ylabel("[-]")
plt.xlabel("f [Hz]")
plt.bar(f[:a], np.abs(vibrations_fft[:a]) / np.max(vibrations_fft[:a]), width=0.3)
plt.plot(f[:a], weight[:a], c="r")

plt.subplot(122)
plt.title("Weighted frequency spectrum")
plt.ylabel("[-]")
plt.xlabel("f [Hz]")
plt.bar(f[:a], np.abs(vibrations_fft_w[:a]) / np.max(vibrations_fft[:a]), width=0.3)
plt.ylim(0, 1)
plt.show()
```

{{< figure src="/img/post-6-sbr/fig_4.png" title="Weighted frequency spectrum" >}}


The above figure shows that all frequencies are downscaled. However the lower frequencies are downscaled most. The higher frequencies are probably leading to the most hindrance. And very low frequencies will probably just rock you to sleep.

By transforming the signal back to the time spectrum we can see how the frequency scaling affected the signal.

```python
vibrations_w = np.fft.ifft(vibrations_fft_w).real

plt.title("Weighted vibration")
plt.ylabel("v [mm/s]")
plt.xlabel("t [s]")
plt.plot(t, vibrations_w)
plt.show()
```
{{< figure src="/img/post-6-sbr/fig_5.png" title="Weighted time spectrum" >}}


By comparing the above figure with the original signal we can see it has changed a bit. By weakening the lower frequencies the signal has decreased in amplitude. The maximum amplitude has dropped from ~ 30 mm/s to ~ 25 mm/s.

</br>

## Effective value

The second formula describes how you can compute the effective value of the vibration signal, or 'voortschrijdende effectieve waarde' in Dutch. This formula looks a lot like the formula of the [Root Mean Square](https://en.wikipedia.org/wiki/Root_mean_square) (RMS) of a signal. The formula of the RMS given by:

<div>\[RMS = \sqrt{\frac{1}{T}\int^T_0 v(t)^2dt}\]</div>

It resembles the first formula. However for <span>\\(v(t - \xi)\\)</span> the velocity signal is multiplied with <span>\\(e^{-\xi/\tau}\\)</span>. Also the signal is not an integral with steps dt, but an integral with steps d&xi;

The &xi; is actually another parameter for the time t. Every increment in time dt a new integral is computed from t<sub>0</sub> to t<sub>i</sub> with steps d&xi; (which are the same size as dt); The function <span>\\(e^{-\xi/\tau}\\)</span> is another scaling function. The larger &xi; becomes, the smaller the multiplication factor becomes. 


```python
v_sqrd_w = vibrations_w**2
a = 55
c = ["#1f77b4" for i in range(a)]

current = 51
c[current] = "#d62728"

xi = t[:current + 2]
g = np.exp(-xi / 0.125)

plt.subplot(211)
plt.title("Squared signal")
plt.plot(t[:current + 2], g[::-1][:a] * v_sqrd_w[current + 1], color="r")
plt.bar(t[1:a], v_sqrd_w[1:a], width=0.002, color=c)
plt.ylim(0, np.max(v_sqrd_w))

for i in range(g.size):
    v_sqrd_w[i] *= g[-i]

plt.subplot(212)
plt.title("Weighted squared signal")
plt.xlabel("t [s]")
plt.bar(t[1:a], v_sqrd_w[1:a], width=0.002, color=c)
plt.ylim(0, np.max(vibrations_w**2))
plt.show()
```

{{< figure src="/img/post-6-sbr/fig_6.png" title="Scaled down time signal per time step" >}}

In the code snippet above the signal is squared and plotted. The red bar in the plot is the current time inteval t<sub>i</sub>. All the preceding values of, and including the value for v(t)<sup>2</sup>, will be multiplied with the red function. This function ranges from 0 to 1. Values close to the current time interval t<sub>i</sub> will keep their value. Values further away will be scaled down more. This multiplication is done for every time step t<sub>i</sub>. The scaled down signal for the current time step is shown the in the second figure.

When the weighted value for every time step is determined the RMS can be computed for this weighted signal.

```python
Ts = 0.125
v_eff = np.zeros(t.size)
dt = t[1] - t[0]

for i in range(t.size - 1):
    g_xi = np.exp(-t[:i + 1][::-1] / Ts)
    v_eff[i] = np.sqrt(1 / Ts * np.trapz(g_xi * v_sqrd_w[:i + 1], dx=dt))

plt.plot(t, v_eff)
plt.title("Effective value")
plt.ylabel("v [mm/s]")
plt.xlabel("t [s]")
plt.show()
```

{{< figure src="/img/post-6-sbr/fig_7.png" title="Effective value time signal" >}}

</br>

## Conclusion

We have computed the effective value (voortscrhijdende effectieve waarde) for a random time signal. It was quite a hassle for me to find out what should be done. The guideline does not mention that you need to switch between the frequency and the time spectrum two times. Also the interpretation of &xi; in the second formula could really use a calculation example. 

What you eventually can do with the computed effective value is something I will leave to the guideline. I hope this helps someone a few hours when dealing with the SBR!


<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
