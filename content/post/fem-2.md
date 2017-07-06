+++

date = "2017-02-12T21:35:33+02:00"
draft = false
title = "Python 1D FEM Example 2"
description = "Example code for 1D FEM in Python."

+++

# Example 2: Truss framework

A while ago I wrote a FEM package for basic frames and trusses in Python.

This is a basic example that shows how to use it.

You can download it on [github](https://github.com/ritchie46/structural_engineering)

![view](/img/fem/example_2/example_2.png)

```python
# if using ipython notebook
%matplotlib inline

import StructuralEngineering.FEM.system as se

# Create a new system object.
system = se.SystemElements()

# Add beams to the system. Positive z-axis is down, positive x-axis is the right.
system.add_truss_element(location_list=[[0, 0], [0, -5]], EA=5000)
system.add_truss_element(location_list=[[0, -5], [5, -5]], EA=5000)
system.add_truss_element(location_list=[[5, -5], [5, 0]], EA=5000)
system.add_truss_element(location_list=[[0, 0], [5, -5]], EA=5000 * math.sqrt(2))

# get a visual of the element ID's and the node ID's
system.show_structure()
```

![structure_2](/img/fem/example_2/structure_2.png)

```python
# add hinged supports at node ID 1 and node ID 2
system.add_support_hinged(nodeID=1)
system.add_support_hinged(nodeID=4)

# add point load at node ID 2
system.point_load(Fx=10, nodeID=2)

# show the structure
system.show_structure()
```

![structure_2](/img/fem/example_2/structure_wi_supp_2.png)


```python
# solve
system.solve()
# show the reaction forces
system.show_reaction_force()

``` 
![](/img/fem/example_2/reaction_2.png)

```python
# show the normal force
system.show_normal_force()
``` 
![](/img/fem/example_2/normal2.png)

```python
system.show_displacement()
``` 

![](/img/fem/example_2/displacement_2.png)
