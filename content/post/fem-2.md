+++

date = "2017-02-12T21:35:33+02:00"
draft = false
title = "Python 1D FEM Example 2"
description = "Example code for 1D FEM in Python."
tags = ["fem"]
+++

# Example 2: Truss framework

Simple code example for [anaStruct](https://github.com/ritchie46/anaStruct).

![view](/img/fem/example_2/example_2.png)

```python
# if using ipython notebook
%matplotlib inline

import math
from anastruct.fem.system import SystemElements

# Create a new system object.
ss = SystemElements(EA=5000)

# Add beams to the system.
ss.add_truss_element(location=[[0, 0], [0, 5]])
ss.add_truss_element(location=[[0, 5], [5, 5]])
ss.add_truss_element(location=[[5, 5], [5, 0]])
ss.add_truss_element(location=[[0, 0], [5, 5]], EA=5000 * math.sqrt(2))

# get a visual of the element ID's and the node ID's
ss.show_structure()
```

![structure_2](/img/fem/example_2/structure_2.png)

```python
# add hinged supports at node ID 1 and node ID 2
ss.add_support_hinged(node_id=1)
ss.add_support_hinged(node_id=4)


# add point load at node ID 2
ss.point_load(Fx=10, node_id=2)

# show the structure
ss.show_structure()
```

![structure_2](/img/fem/example_2/structure_wi_supp_2.png)


```python
# solve
ss.solve()
# show the reaction forces
ss.show_reaction_force()

``` 
![](/img/fem/example_2/reaction_2.png)

```python
# show the normal force
ss.show_axial_force()
``` 
![](/img/fem/example_2/normal2.png)

```python
ss.show_displacement()
``` 

![](/img/fem/example_2/displacement_2.png)
