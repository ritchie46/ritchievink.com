+++
title = "Computer build me a bridge"
date = "2018-01-14"
description = ""
categories = ["category"]
tags = ["tag1", "tag2"]
draft = true
author = "Ritchie Vink"
+++

In earlier posts I've analized simple structures with a Python fem package called [anaStruct](https://github.com/ritchie46/anaStruct). And in this [post]({{< ref "post/nl_water_acc.md" >}}) I've used anaStruct to analize a very non linear roof ponding problem.

Modelling a structure in Python may seem cumbersome in relation to some programs that offer a graphical user interface. For simple structures this may well be the case. However now we've got a simple way to programmatically model 2D structures, I was wondering if we could let a computer model these structures for us.

I've set the goal of building (sub) optimal bridges by letting a model determine the optimal solutions. During this process I've first tried Reinforcement Learning and had limited success with that approach. A technique that had more succes, and turned out to be much simpler, was Genetic Algorithms (GA). This post describes how we can let a computer find (sub) optimal solutions for our problem.

# Genetic Algorithms
Before going to the bridges let's first look at the genetic algorithms model. This is actually a sort of hill climbing algorithm that tries to find the optimal solution by random changes. 

## Fitness
The height of the hill is described with a fitness function. If we would compare this with Darwin's survival of the fittest, the fitness function would map your DNA to the probability of you:

1. Living long enough to reproduce.
2. Convincing the other sex to have coitis with you.

This guy would have a fitness score of over 9000:

{{< youtube owGykVbfgUE >}}

<br>
<br>

And this guy would have a score of 1:

{{< youtube id="_9umwdE8VoY?t=19s _" >}}

<br>
He does get a point, because he misses. 

The fitness function can be anything though. It is a function that outputs positive numbers. The output of the function should be higher if the result is more optimal or closer to our desired behaviour. When you want to build a brigdge the fitness function could for instance relate to:

* Is the structure in equillibrium (does it not collapse).
* Amount of elements used (We probably don't want to spent high material costs).
* Amount of deflection.

## DNA
The input of the fitness function is DNA. For humans DNA has 4 possible letter states. For the DNA of the bridge we are going to use only 2 states, on or off. We can think of the DNA as a long sequence of binary values. Every index of the sequence belongs to an element on a grid. Such an element is a beam/ rod in the bridge and can be either on or off. If it is on, the element will be build. If it is off, it won't. Simple right!

Let's make it more visual with a small example. The figure below shows a grid containing possible locations to place an element. Dashed lines mean that there can be an element, but it isn't build yet. The element locations are numbered from one to eleven.

{{< figure src="/img/post-12-ga-bridge/empty_grid.svg" title="Empty grid." >}}

The DNA string belonging to this grid of empty elements is an array of eleven zeros. The figure below shows:

* On the first row:  the DNA sequence.
* On the second row: the corresponding beam of the DNA state.
 
{{< figure src="/img/post-12-ga-bridge/empty_dna.svg" title="DNA sequence belonging to the empty grid." >}}

If we randomly change some DNA values from zero to one, we activate the element and the element will be build on the grid. The figure below shows the grid and the corresponding DNA when beam number 3, 7 and 10 are activated.

{{< figure src="/img/post-12-ga-bridge/active_grid.svg" title="Grid with activated elements." >}}

The DNA belonging to this grid is shown below.
{{< figure src="/img/post-12-ga-bridge/active_dna.svg" title="DNA sequence belonging to the grid with activated elements." >}}

# Optimization
This is how the DNA sequence will compare to the eventual bridge. In the first generation the DNA will be randomly generated. The probability that a random generated sequence will yield a high fitness score is very low, therefore we will generate an entire population of DNA.

We can compute the fitness score for the entire population and determine which DNA sequences build better bridges than others. The first generation probably won't contain any optimal structures. Therefore we want to be able to change the DNA randomly whilst maintaining the DNA parts that lead to a high fitness score. 
Genetic Algorithms are able to do so via selection, crossover, and mutation.

## Selection
When the fitness of a generations population is determined, we select parents that may exchange DNA. There are different strategies for selecting parents. The chance of a parent being selected can for instance be proportional to their fitness score. However, when we would have a distribution with a fitness score as shown below, a selection procedure that is proportional to the fitness score would lead to a population with almost all the DNA coming from the first 2 parents. 

{{< figure src="/img/post-12-ga-bridge/fitness_score.svg" title="Probability of being selected proportional to the fitness score." >}}

Having all the DNA coming only from a few parents, and thus having little variation in the population increases the chance of getting stuck on local maxima. In order to maintain a healthy population, the chance of being selected will be proportional the rank of the parent instead of the fitness score. 

The figure below shows the selection probability being equal to the rank of the parent. We can see that the next generation will have most DNA coming from the highest scoring parent, but the low scoring parents will also contribute a significant part in the gene pool.

{{< figure src="/img/post-12-ga-bridge/rank_score.svg" title="Probability of being selected proportional to the rank of the parent." >}}

## Crossover
After the selection process has succeeded (ergo you had pretty good dance moves, you didn't stutter and you also don't know how it has happened, but now you're in the other persons bed) the exchange of DNA takes place, Crossover!

Crossover can be done by combining the two DNA arrays randomly. This DNA exchange process can be done on various ways (random, the first part of the sequence, the last part of the sequence, etc.). The method I've used was letting a percentage of the selected parents exchange their DNA completely random. 

The figure below shows how a DNA exchange could look if we randomly swap DNA parts.

{{< figure src="/img/post-12-ga-bridge/Crossover.svg" title="Random crossover strategy." >}}

## Mutation
In alignment with what I've learned in high school biology, during the copying of DNA there is a chance of random mutation.
This random mutation makes it possible to explore new DNA states that aren't available the gene pool of the population.

## Eat, sleep, mate, repeat
These were all the steps needed for a genetic algorithm. The fitness, selection, crossover, and mutation steps will be repeated until it yields a satisfactory result, or the model is stuck on its (local) optimum.

