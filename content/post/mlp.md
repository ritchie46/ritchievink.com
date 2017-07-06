---
author: author
date: 2017-07-01T12:11:12+02:00
description: description
draft: true
keywords:
- key
- words
tags:
- one
- two
title: Programming a neural network from scratch
topics:
- topic 1
type: post
---

## Intro

At the moment of writing this post it has been a few months since I've lost myself in the concept of machine learning. I have been using packages like TensorFlow, Keras and Scikit-learn to build a high conceptual understandig of the subject. I did understood intuition of the backpropagation algorithm and the idea of minimizing costs, but I haven't programmed it myself. Tensorflow is regarded as quite a low level machine learning package, but it still abstracts the backpropagation algorithm for you. In order to better understand the underlying concepts I've decided to build a simple neural network without any deep learning framework. In this post I will show you how to program this in python. We'll be only using the numpy package for the linear algebra abstraction.

</br>

## Network

We are going to build a three layer neural net. This is the minimum required amount of layers when talking of a multi layer perceptron network. Every net requires an input layer and an output layer. The remaining layers are the so called hidden layers. Lets assume are in the possession of a pretrained neural network. The network predicts certain outputs based on input data. 
	

{{< figure src="/img/post-9-mlp/nn_diagram_1.png" title="Image 1: Feed forward pass of a neural network" >}}


The figure above shows the concept of a neural network. It has got a certain input vector <span>\\( \vec{x} \\)</span> representing the data. This data is multiplied with the weigths <span>\\( \vec{w} \\)</span> and shifted over <span>\\( \vec{b} \\)</span>. The output is: 

<div>$$ \sum{\vec{x} \odot \vec{w} + \vec{b}} = z $$</div>

Note that <span>\\( \odot \\)</span> means the Hadamard product and is just elementwise multiplication. If we would make this visual in vector or neuron form it would look something like this.

<div style="text-align:center">{{< figure src="/img/post-9-mlp/pc_1.png" title="Vector form">}}</div>

<div style="text-align:center">{{< figure src="/img/post-9-mlp/pc_2.png" title="Neuron form">}}</div>

The figures above show the connection of three input nodes to one node of a hiden layer. The output vector is eventually summed resulting in one scalar output in the hidden layer node. The output of a hidden layer node is the input for the next layer. As you can see the process repeats until the neural networks generates an output. 

Note that we can also replace the bias vector with one bias scalar. By doing so we can use the dot product of the weights and the activations of the previous layer <span>\\( x \\)</span>.

<div>$$ \vec{x} \cdot \vec{w_1} + b_1 = z $$</div>

</br>

### Non linearity
Image 1 shows a network that consists of inputs. The input vector <span>\\( \vec{x}\\)</span> is multiplied with <span>\\( \vec{w} \\)</span> and shifted along <span>\\( \vec{b} \\)</span>, the weights and biases of the first layer. The second layer seems to repeat the same math operation only with other weights and biases. This is a concatenation of two linear functions. <span>\\( w(x) + b \\)</span> is a linear function and feeding the output of <span>\\( w_1(x) + b_1 \\)</span> in <span>\\( w_2(x) + b_2 \\)</span> will result in another linear function. If we assume that our network is properly trained, we can conclude that the last layer is redundant as the first layer could have already the linear function we are looking for. 

A concatenation of multiple linear functions can thus be replaced by one linear function and isn't beneficial to the model. To give the second layer any purpose the output of the first layer <span>\\( z \\)</span> is multiplied with a non linear function <span>\\( f(x) \\)</span> resulting in:

<div>$$ f(\vec{x} \cdot \vec{w_1} + b_1) = a_2 $$</div>

The final output a<sub>2</sub> is called the activation of the neuron. In this case we use the subscript 2, as it is the activation of the node in the second layer. 

The non linear functions we are going to use is the Relu function and the Sigmoid function. 

Relu function:

<div>$$ f(x) = max(0, x) $$</div>

<div style="text-align:center">{{< figure src="/img/post-9-mlp/relu.png">}}</div>

Sigmoid function:

<div>$$ f(x) = \frac{1}{1 + e^{-x}} $$</div>

<div style="text-align:center">{{< figure src="/img/post-9-mlp/sigmoid.png">}}</div>

These non linear functions give the network the ability to learn non linear relations in the data and make it possible to learn in more depths of abstraction. The relu function will let every positive value pass through the network. Negative values will be changed to zero so that that neuron doesn't activate at all. The sigmoid functions squashes the outputs to value between zero and one. We can think of this as probabilities. We will use the relu function in the hidden layer and the sigmoid function at the output layer. 

</br>

## Code

So now we have got a sort of high level of what is happening in the net we can write some code. Just a few variable notations up front:

* **w**: weights
* **b**: biases
* **z**: output of a neuron: <span>\\(x \cdot w + b\\)</span>
* **a**: activations of z:  <span>\\(f(z)\\)</span>

### Activation functions
First we write the two activation functions. The @staticmethod decorator makes it possible to call the two methods directly from class level so we don't have to create any objects from the two classes.



```python
import numpy as np


class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

```

</br>

### Network class
Next is the container that will keep the hidden state and perform all the logic for the neural network, the Network class.

```python

class Network:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.

        """

        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}

        for i in range(len(dimensions) - 1):
            self.w[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

```

In the `__init__` function we initiate the neural network. The `dimensions` argument should be an iterable with the dimensions of the layers. Or in other words the amount of nodes per layer. The `activations` argument should be an iterable containing the activation class objects we want to use. We need to create some inner state of weights and biases. This inner state is represented with two dictionaries `self.w` and `self.b`. In the for loop we assing the chosen dimensions to the layer numbers. A neural network containing 3 layers; input layer, hidden layer, output layer will have weights and biases assigned in layer 1 and layer 2. Layer 3 will be the output neuron.

We can see that the biases are initiated as zero and the weights are drawn from a random distribution. The dimensions of the weights are determined by the dimensions of the layers. The drawn weights are eventually divided by the square root of the current layers dimensions. This is called [Xavier initialization](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization) and helps prevent neuron activations from being too large or too small.

So if we want to initiate an neural network with the same dimensons as the one in figure 1 we could create an object from the Network class. In order to reproduce the same outputs, I've set the random seed to 1.

#### initialization
```
np.random.seed(1)
nn = Network((2, 3, 2), (Relu, Sigmoid))
```

If we print the weights and biases we will recieve the following dictionaries. As said in the first part of the post, every input neuron is connected to all the neurons of the next layer, resulting in 2 * 3 = 6 weights in the first layer and resulting in 3 * 2 = 6 weights in de second layer. Because we've chosen to add the biases after the summation has taken place, we only need as many biases as there are nodes in the next layer. 

The following snippet shows the location of the internal state of the network. The keys represent the layers, the values represent the weights, biases and activation classes.

#### internal state
```python
Weights:
{
	1: array([[ 1.14858562, -0.43257711, -0.37347383],
	       [-0.75870339,  0.6119356 , -1.62743362]]),
	2: array([[ 1.00736754, -0.43948301],
	       [ 0.18419731, -0.14397405],
	       [ 0.84414841, -1.18942279]])
}

Biases:
{
	1: array([ 0.,  0.,  0.]), 
	2: array([ 0.,  0.])
}

Activation classes:
{
	2: <class '__main__.Relu'>, 
	3: <class '__main__.Sigmoid'>
}
```

</br>

## Feeding forward
Now we are going to implement the forward pass. The forward pass is how the network generates output. The inputs are fed into the network, are multiplied with the weights and shifted along the biases of every layer (if they pass through the Relu function) and finaly they pass the sigmoid function resulting in an output between 0 and 1. The mathematics of the forward pass is the same for every layer. The only variable is the activation function <span>\\( f(x) \\)</span>.

<div>$$ \vec{a_(i - 1)} \cdot \vec{w_i} + b_i = z_i $$</div>
<div>$$ f(z_i) = a_i $$</div>

However algorithmicly we have abstracted the activation function with activation classes that all have the method `.activation()` meaning that we can loop over all the layers doing the same mathematical operation and finally call the varying activation function with the `.activation()` method. We add a `.feed_forward()` method to the `Network` class.

```
def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

	# w(x) + b
	z = {}

	# activations: f(z)
	a = {1: x}  # First layer has no activations as input. The input x is the input.

	for i in range(1, self.n_layers):
	    # current layer = i
	    # activation layer = i + 1
	    z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
	    a[i + 1] = self.activations[i + 1].activation(z[i + 1])

	return z, a
```

We create two new dictionaries in the function `z` and `a`. In these dictionaries we append the outputs of every layer, thus again the keys of the dictionaries map to the layers of the neural network. Note that the first layer has no 'real' activations as it is the input layer. Here we consider the the inputs `x` as the activations of the previous layer.

The dictionaries structure would be like this.

```python
a:
{
	1: "inputs x",
	2: "activations of relu function in the hidden layer",
	3: "activations of the sigmoid function in the output layer"
}

z:
{
	2: "z values of the hidden layer", 
	3: "z values of the output layer"
}


```

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
