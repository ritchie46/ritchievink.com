+++
title = "Transfer learning with Pytorch: Assessing road safety with computer vision"
date = "2018-04-12"
description = ""
categories = ["category"]
tags = ["machine learning", "python"]
draft = false
author = "Ritchie Vink"
+++

For a project at Xomnia, I had the oppertunity to do a cool computer vision assignment. We tried to predict the input of a road safety model. [Eurorap](http://www.eurorap.org/) is such a model. 
In short, it works something like this. You take some cars, mount them with cameras and drive around the road you're interested in. The 'Google Streetview' like material you've collected is sent to a 'mechanical turk' workforce to manually label the footage. 

{{< figure src="/img/post-13-eurorap/mechanical_turks.jpg" title="Mechanical Turks labeling images of Dutch Roads.">}}

This manually labeling of pictures is of course very time consuming, costly, not scalable and last but not least boooring!

Just imagine yourself clicking through thousands of images like the one shown below.

{{< figure src="/img/post-13-eurorap/road_drenthe.jpg" title="The thrilling working life of a mechanical Turk.">}}

This post describes the proof of concept we've done, and how we've implemented it in Pytorch.

## Dataset

The video material is divided in images shot every 10 meters. Even a Mechanical Turk has trouble not shooting itself of boredom when has to fill in 300 labels of what he sees every 10 meters. To make his work a little bit more pleasant he only had to fill in the labels every 100 meter for what they had seen in the 10 pictures.

This working method has led to terrible labeling of the footage however. Every object that you drive by within these 100 meters is labeled to all 10 images, whilst it most of the time only is visible in 2~4 images!

For the project we did a proof of concept and we've only regarded a small subset of all labels. In total there are approximately 300. We have done our analysis on 28 labels. And because of the problem of the mislabeled objects, we've chosen mostly labels that are parallel to the road. As it seems likely that those labels hold true for all 10 images. For instance, a road that has emergency lane in one image, is likely to have an emergency lane in the subsequent images.

Below you'll see a plot with the labels we have trained our model on, and the amount of occurrences of that specific label. In total there were approximately 100,000 images in our dataset.

{{< figure src="/img/post-13-eurorap/subset.png" title="Counts of the labels we've used.">}}

## Transfer learning
There is a reasonable amount of images to work with, however the large inbalance in the labels, the errors in the labeling, and the amount of data required for the data hungry models called neural networks make it unlikely that we have enough data to properly train a robust computer vision model. 

To give you an idea of the scale here. The record breaking convolutional neural networks, you read about in the news, are often trained on the imageNet dataset, which contains 1.2 million images. These modern ConvNets (convolutional neural networks) are trained for ~2 weeks on heavy duty GPU's to go from their random initialization of weights, to a configuration that is able to win prices.

It would be nice if we don't have to waste all this computational effort and somehow could transfer what we have learned in one problem (imageNet) to another (roadImages). Luckely, with ConvNets this isn't a bad idea and there is quite some logical intuition in doing so. 
Neural networks are often regarded as black box classifiers. And for the classification part, this is currently somewhat true. But for traditional ConvNet architectures there is a clear distinction between the classification and the feature extraction.
