+++
date = "2018-05-18"
description = ""
tags = ["python", "machine learning", "algorithm breakdown"]
draft = true
author = "Ritchie Vink"
title = "Algorithm Breakdown: Affinity propagation"
+++
{{< figure src="/img/post-14-affinity_propagation/fw.jpg">}}

On a project I worked on at the ANWB (Dutch road side assistence company) we mined driving behavior data. We wanted to know how many persons were likely to drive a certain vehicle on a regular basis. Naturally k-means clustering came to mind. The k-means algoritm finds clusters with the least inertia for a given `k`.

A drawback is that often, k is not known. For the question about the numbers of persons driving a car, this isn't that big of a problem as we have a good estimate of what k should be. The number of persons driving on a regular basis would probably be in a range of 1 to 4. 

Nevertheless, this did make me wonder if there were algorithms that would define k for me. 

## Affinity propagation
Affinity propagation is a clustering method that next to qualitative cluster, also determines the number of clusters, k, for you. Let's walk through the implementation of this algorithm, to see how it works.

As it is a clustering algorithm, we also give it random data to let it's OCD work on.

```python
n = 20
size = (n, 2)
np.random.seed(3)
x = np.random.normal(0, 1, size)

for i in range(4):
    center = np.random.rand(2) * 10
        x = np.append(x, np.random.normal(center, .5, size), axis=0)
            
            c = [c for s in [v * n for v in 'bgrcmyk'] for c in list(s)]

            plt.figure(figsize=(15, 6))
            plt.title('Some clusters in 2d space')
            plt.scatter(x[:, 0], x[:, 1], c=c)
            plt.show()
```

{{< figure src="/img/post-14-affinity_propagation/data.png">}}

Affinity propagation the data points can be seen as a network where all the data points send messages to all other points. The subject of these messages are exemplars. Exemplars are points that explain the other data points and are the most significant of their cluster. A cluster only has one exemplar. All the data points want to collectively determine which data points are an exemplar for them. These messages are stored in two matrices. 

* The 'responsibility' matrix R. In this matrix, $r(i, k)$ reflects how well-suited point $k$ is to be an exemplar for point $i$.
* The 'availability' matrix A. $a(i, k)$ reflects how appropriate it would be for point $i$ to choose point $k$ as its exemplar.

Both matrices can be interpreted as log probabilities and are thus negative.

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
  });
  </script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
