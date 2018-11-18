+++
date = "2018-10-09"
description = "Analysis breakdown of the Facebooks prophet model. In this post we implement prophet in PyMC3."
tags = ["machine learning", "python", "algorithm breakdown", "gradient boosting"]
draft = false
keywords =["machine learning", "python", "numpy", "gradient", "boosting"]
author = "Ritchie Vink"
title = "Algorithm breakdown: GBM"
og_image = "/img/post-20-gbm/stacking.png"
+++

{{< figure src="/img/post-20-gbm/stacking.png" >}}
<br>
We were making a training at work about ensemble models. When we were discussing different techniques like bagging, boosting, and stacking, we also came on the subject of gradient boosting. Intuitevely, gradient boosting, by training on the residuals made sense, however the name gradient boosting did not right away. This post we are exploring the name of gradient boosting and of course also the model itself!

## Intuition

### Single decision tree

Gradient boosting is often used as a optimization technique for decision trees. Decision trees are rule based models. Using a single decision tree, often leads to models with high variance, i.e. overfitting. Below we can see why this happens. 

{{< figure src="/img/post-20-gbm/decision-tree.svg" title="50% split at every node.">}}

You can imagine a tree, where at every node aproximately 50% of the data is being split. The nodes that are deep in the tree, will only see a small subset of the data and make splits on these small subsets. The chances that these subsets don't represent the real distribution of the data increase by depth, and so does the chance of overfitting. Another limitation of decision trees is that they do not optimize an overall loss function. Decision trees are trained 'greedy', which means that
they minimize loss at every node. However minimizing loss at a node level does not guarantee minimal loss at a tree level.

### Gradient boosting trees

Gradient boosting trees solves part of the limitation mentioned above. Instead of training a single tree, multiple trees are trained sequentially. To lower the variance of the trees, they are however restricted. They are turned into weak learners by setting limits on the depth of the trees. The decision trees depth if often chosen between 3 and 6 layers. We allow a little bit of depth, so that we can compare jointly occurring of variables. If a tree has a depth of three layers, we can
compare conditions like this:

`if A and B and C then;`

Okay, now we have talked about the limitation of decision trees, lets look  how gradient boosting uses trees and tries to overcome these problems. Note that we will focus this entire post on regression problems and therefore assumme numericall data, not categorical.

Gradient boosting trees is recurrently defined as a set of $M$ trees

$$ F\_m(x) = F\_{m-1}(x) + h\_m(x) $$

$F\_m(x)$ is an iterative boost of the model, by adding a decision tree $h\_m(x)$ to previous iteration of the model $F\_{m-1}(x)$. The tree $h\_m(x)$ is trained on the residuals of $F\_{m-1}(x)$. Residuals are the difference with the true labels $y$ and the predictions of the model $\hat{y}$.

$$h\_m(x) = y - F\_{m-1}(x) $$

Intuitively this makes a lot of sense. We missed a few spots in our previous model $F\_{m-1}(x)$ and therefore we let the next model $h\_m(x)$ focus on those spots. And by iterating a few times, we will aproach closer to $y$ until convergence.

It also takes into account the problems we saw at a single decision tree. 

* Greedy learning, no overall loss function is optimized.
* Decisions are made on small subsets of the data.

The overall loss is now minimized, but we'll get to proof of that later! The decisions are now not made on small subsets, because we don't let the trees get too deep and every iteration a new tree is trained on all the data (now being residuals). Every new tree, trains acutally on new rescaled data. The new tree will focus on the samples where the previous iteration $F\_{m-1}$ is very wrong as this leads to large residuals. Every new tree focusses on the errors of the previous one, by taking into account all the data, and not a subset of the data! This distinction is very important as this reduces the chance of overfitting a lot.

### Implementation
Below we implement the gradient boosting as defined above, with one little adjustment called shrinkage. This is nothing more than adding a learning rate $\eta$ when adding a new tree. The definition then becomes

$$ F\_m(x) = F\_{m-1}(x) + \eta h\_m(x) $$

```python
from sklearn import datasets, tree, model_selection, metrics
import numpy as np

class GradientBooster:
    def __init__(self, n_trees=20):
        self.f = []
        self.learning_rates = []
        self.n_trees = n_trees
    
    def fit(self, x, y, lr=0.1):
        class F0:
            predict = lambda x: np.mean(y) * np.ones(x.shape[0])
        self.f.append(F0)
        self.learning_rates.append(1)
        
        for _ in range(self.n_trees):
            m = tree.DecisionTreeRegressor(max_depth=5)
            res = y - self.predict(x)
            m.fit(x, res)
            self.f.append(m)
            self.learning_rates.append(lr)
            
    def predict(self, x):
        return sum(f.predict(x) * lr for f, lr in zip(self.f, self.learning_rates))

```

Let's quickly verify if it works by trying to outperform a decision tree model with a regression problem.

```python
# Some data
np.random.seed(123)
x = datasets.load_diabetes()['data']
y = datasets.load_diabetes()['target']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

def evaluate(m):
    print('Training score:', metrics.r2_score(y_train, m.predict(x_train)), 
    '\tTesting score:', metrics.r2_score(y_test, m.predict(x_test)))
```

``` python
# Algorithm to beat
p = {'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 3, 7],
    'min_samples_leaf': [1, 3, 7]}

m = model_selection.GridSearchCV(tree.DecisionTreeRegressor(), p)
m.fit(x_train, y_train, )

evaluate(m)
```
`>>> Training score: 0.6595521969069875 	Testing score: 0.14972533215961115`

```python
m = GradientBooster(20)
m.fit(x_train, y_train)
evaluate(m)
```
`>>> Training score: 0.8248281659615961 	Testing score: 0.43006412209704451`

That seems to work fine! On both the training set as the test set we greatly outperform the decision tree. This is acutally all there is to it to implement a gradient boosting algorithm. We take a weak learner and boost the models performance by training on the residuals.

## Why is it called gradient boosting?
In the definition above, we trained the additional models only on the residuals. It turns out that this case of gradient boosting is the solution when you try to optimize for MSE (mean squared error) loss. But gradient boosting is agnostic of the type of loss function. It works on all differentiable loss functions. We could see gradient boosting as a generalization of the algorithm we've defined in the section above.

### Gradient descent
Gradient boosting has quite some similarities with gradient descent. With gradient descent we try to optimize the parameters $\theta$ of function $F(x|\theta)$. We do this by minimizing a loss function $L(y, \hat{y})$. The loss function is a function that takes the true value $y$ and the predicted value $\hat{y}$ as input and returns a loss value. The function decreases as the predictions $\hat{y}$ get better.

$$\theta\_t = \theta\_{t-1} - \eta \nabla\_{\theta} L(y, \hat{y})$$

Every iteration $t$ we adjust the parameters of time step $t-1$ by the a factor $\eta$ of the gradient of the loss with respect to the parameters $\theta$. We add a minus sign because we want to minimize the loss, not maximize it. That is gradient descent in a nutshell!

### Gradient boosting

Now, let's compare it with gradient boosting! Let's rewrite $F(x)$ as $F$ for succinctness purposes.

$$F\_m = \hat{y} - \eta \nabla\_{\hat{y}} L(y, \hat{y})$$


$$F\_m = F\_{m-1} - \eta \nabla\_{F\_{m-1}} L(y, F\_{m-1})$$

See the similarities? Instead of optimizing the parameters of a function we, optimize the function architecture (and its parameters) itself! Incrementally we add the partial derivatives with respect to $F\_{m-1}$. 

We can go back to our earlier notation with $h\_m$


$$ F\_m = F\_{m-1} - \eta h\_m $$

where $h\_m = \nabla\_{F\_{m-1}} L(y, F\_{m-1})$

## L2 boosting

We've already mentioned that training a new decision tree on the residuals of previous iteration is actually gradient boosting when minimizing the MSE loss. Let's explore why that is.

The MSE loss is defined as

$$L = \frac{1}{2}(y - \hat{y})^2$$

The $\frac{1}{2}$ constant is added so that the partial derivative is easier to work with. Don't worry, it is not cheating. This has no influence on the working of the algorithm.

Now let's define the partial derivate w.r.t. to functions output. $ \nabla\_{F\_{m-1}} L(y, F\_{m-1})$. By applying the chain rule a few times we'll come to a solution.

$$\frac{\partial L}{\partial \hat{y}} = (y - \hat{y}) \cdot -1 = \hat{y} - y$$

And thats it. Training a tree $h\_m$ on the partial derivative $\nabla\_{F\_{m-1}} L(y, F\_{m-1})$, is the same as training a tree on the resiudals $\hat{y} - y$! 

## L1 boosting

Now this solution was very easy. Let's look how this definition of gradient boosting holds with another loss function. The MAE (mean absolute error) loss.

MAE is defined by

$$L = |y-\hat{y}|$$

If we rewrite the absolute signs, we get

$$L = \sqrt{(y - \hat{y})^2} = ((y - \hat{y})^2)^\frac{1}{2}$$

Again we'll apply the chain rule


$$ \frac{\partial L}{\partial \hat{y}} = \frac{\partial L}{\partial (y - \hat{y})^2} \cdot \frac{\partial (y - \hat{y})^2}{\partial y - \hat{y}} \cdot \frac{\partial y - \hat{y}}{\partial\hat{y}} $$

$$ \frac{\partial L}{\partial \hat{y}} = \frac{\frac{1}{2}}{\sqrt{(y - \hat{y})^2}} \cdot 2(y - \hat{y}) \cdot -1$$

$$ \frac{\partial L}{\partial \hat{y}} = \frac{\hat{y} - y}{\sqrt{(\hat{y} - y)^2}} = \frac{\hat{y} - y}{|\hat{y} - y|} = sign(\hat{y} - y)$$

And that is our solution! Now we'll train the new decision tree $h\_m$ on the sign on the residuals. 

However, if we think about this solution as a model, it is not very pratical. With every new tree, we now take steps of approximately $\pm1 \cdot \eta$. For instance, if we try to predict housing prices and our data is not scaled and our data is not scaled, we could be adding hundred thousands of trees to our model! Both memorywise and computationalwise this isn't a realistic solution.

## TreeBoost

For this specific problem, an algorithm TreeBoost was proposed by [Friedman, J. H.](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf).


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
