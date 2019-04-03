+++
date = "2019-04-02"
description = "Location based soil classification"
tags = ["engineering", "python"]
draft = false
keywords =["web", "python"]
author = "Ritchie Vink"
title = "Soil classification with Convolutions and Location embeddings"
og_image = "/img/soil-classification/header.png"
+++

{{< figure src="/img/soil-classification/header.png" >}}

<br>

Soil classification is, in practice, a human process. A geotechnical engineer interprets results from a [Cone Penetration Test](https://en.wikipedia.org/wiki/Cone_penetration_test) (https://en.wikipedia.org/wiki/Cone_penetration_test) and comes up with a plausible depiction of the existing soil layers. These interpretations will often be used throughout a project and are input for many following calculations.

Just as the polio virus, the process of manually mapping data from $x$ to $y$, belongs to the list of things that humanity tries to eradicate from earth.

## First few iterations
There exist automatic soil classification algorithms. The defacto standard is [Robertson et al. (1990) Soil Classification Using The Cone Penetration Test](https://www.cpt-robertson.com/publications/). The classicications resulting from their work, often aren't satisfactory for engineering purposes in the Netherlands. This is due to two reasons.

* The classifications under/ over represent a certain class (Peat/ Silt)
* Engineers want aggregated layers and the Robertson classifications are per measured layer.

*The last point, isn't criticism of the Robertson classification algorithm, as classification and aggregation are two seperate tasks.*

If we want to obtain automated classifications that add more value in practice, we need to focus on the points above.

## Data driven iteration
I believe that a data driven approach can yield better results than the status quo, but we do need labeled data to achieve these results. As I don't have a laboratory nor thousands of hand made labels, I chose to obtain a dataset by quering a database for CPT's and Borehole data within 6 meters apart. Any CPT and Borehole that are less than 6 meters apart are assumed to be from the same soil. The 6 meters cut of is chosen arbitrarily. 
The resulting **dataset has got flawed labels**, but most of the labels are correct.

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
