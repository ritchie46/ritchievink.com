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

If we want to obtain automated classifications that add more value in practice, we need to focus on the points above. So our solution should be able to map input data to a soil distribution and secondly detect changes in the data that represent soil layers.

## Data driven iteration
I believe that a data driven approach can yield better results than the empirical research based approach. But to do so we do need labeled data to achieve these results. As I don't have a laboratory nor thousands of hand made labels, I chose to obtain a dataset by quering a database for CPT's and Borehole data within 6 meters apart. Any CPT and Borehole that are less than 6 meters apart are assumed to be from the same soil. The 6 meters cut of is chosen arbitrarily. 
The resulting **dataset has got flawed labels**, but by scraping large quantities of data we hope to approximate the real soil distribution and make the flaws negligible.

For this project I had roughly 49,000 CPTs and 40,000 boreholes, from which **1,800 pairs** met the condition of being less than 6 meters apart.

## Cone penetration tests
Cone penetration tests are in situ procedures where a steel bar with a cone is pressed into the ground at a constant speed. During the soil penetration different thing are measured at the cone of the penetration bar.

Shown below is a typical graph resulting from a cone penetration test. Typically two lines are drawn, the **qc** values and the **Fr** values. These can be interpreted as

* **qc:** Resistance measured at the tip of the cone. 
* **Fr:** Friction measured at the cone. The friction is normalized by $\frac{1}{q\_c}$ as these are heavily correlated.

<img src="/img/soil-classification/cpt-plot.png" height=450px>

## Decision factors
Machine learning is often beneficial with high dimensional data. In this problem amount of data dimensions is relatively low. And when training powerful ML models, like Neural Networks (Multi Layer Perceptron architecture) or Gradient Boosting Trees, out of the box on this data, we see comparable results. The results are quite reasonable, but they don't reflect the decision factor of a geotechnical engineer. 

A huge part of the decision making is based on the location where the
CPT is taken. And whe a CPT, for instance, is taken at sea, they can tell by the curve of the line that a certain layer consists of seashells. 

If we want a model to be able to take the same decision factors into account as a human does it needs to be able to make decision based on the same information. 

## Model architecture
For this reason, I've chosen a Neural Network architecture with **convolutional layers** that can apply feature extraction on the input signal. Secondly the model was enhanced with **location based embeddings**. This way the model could learn it's own location embeddings and could learn the probabilities of soil conditional on a certain location. Finally, most of the bore hole data shows that layers concise of multiple volumic parts of soil types. Therefore we should
predict the total soil distribution per layer. These adaptations have led to better results than the approaches based on only the **qc** and **Fr** values.

Below is shown a result from the model. The model predicts the soil distribution over the depth. Qualitatively the predictions of the model seem very reasonable and align with a geotechnical mapping. Later in the post we will make a quantitative evaluation.

{{< figure src="/img/soil-classification/example_prediction.png" title="Example prediction" >}} 

## Location embeddings
From the 1800 pairs 48 location clusters were created by applying K-means on the location data. 

Below the location of the clusters are mapped on the map of the Netherlands. Sadly the dataset isn't sufficiently large to fill the entire Netherlands, but it is a good start!

{{< figure src="/img/soil-classification/cluster-map.png" title="Location clusters" >}}

The colors represent a similarity measure between the clusters based on the cosine similarity. Clusters close to each other on the color scale are likely to have similar soil distributions. 

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
