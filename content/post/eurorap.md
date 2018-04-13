+++
title = "Transfer learning with Pytorch: Assessing road safety with computer vision"
date = "2018-04-12"
description = ""
categories = ["category"]
tags = ["machine learning", "python"]
draft = true
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

And this feature extraction part is:

1. Not so black box at all.
2. Very transferable.

### Not so black box at all

When taking a look at which pixels of an image result in activations (i.e. neurons having a large output) in convolutional layers, we can get some intuition on what is happening and meant with feature extraction.


{{< figure src="/img/post-13-eurorap/activated_features.jpg" title="Feature extraction.">}}

When we look at the activated neurons in the first layers, we can clearly identify the car. This means that there are a lot of low level features activated. Even so much that we can still identify the car in the activated neurons. The lower level features are for instance, edges, shadows, textures, lines, colors, etc.

{{< figure src="/img/post-13-eurorap/fe_start.png" title="Feature extractions in the first layer.">}}

Based on these activations of lower features, new neurons will activate, growing in hierarchy and complexity. Activated edges combine in activations of curves, which can on itself in combination with other features lead to activations of objects, like a tyre or a grill.
Lets say that in the image above a neuron is highly activated because a complex feature like a car grill. We wouldn't be able to see this in an image of the activations. As there is only one grill in car (and one grill detector in our network), we would only see one activated neuron, i.e. one white pixel!

{{< figure src="/img/post-13-eurorap/fe_end.png" title="Feature extractions in the last layer.">}}

### Very transferable

Continuing in this line of thought, it seems pretty reasonable to assume that these lower level features extend to more objects than cars. 
Almost every picture has got edges, shadows, colors and so forth. This means that a lot of the feature extraction neurons weigths in a price winning ConvNet are set just right, or at least close by, to our wanted configuration of the weights.

This works so well, that we can pretrain a ConvNet on a large dataset like ImageNet and transfer the ConvNet to a problem with a relatively small dataset (that normally would be insufficient for this ConvNet architecture). The pretrained model still needs to be finetuned. There are a few options like freezing the lower layers and retraining the upper layers with a lower learning rate, finetuning the whole net, or retraining the classifier.

In our case we finetuned the feature extraction layers with low learning rates, and changed the classifier of the model.

## Implementation

For this project we've tried various pretrained ConvNet architectures like GoogleNet, ResNet and VGG and found VGG to produce the best result, closely followed by ResNet. In this section we'll go through the VGG implementation in Pytorch. 

```python
import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self):
        num_classes = 1000
        super(VGG, self).__init__()
        self.features = self.make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)



```


