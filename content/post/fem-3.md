+++

date = "2017-03-12T21:35:33+02:00"
draft = false
title = "Python 1D FEM Example 3"
description = "Example code for 1D FEM in Python."

+++

# Python 1D FEM Example 3.

A while ago I wrote a FEM package for basic frames and trusses in Python.

This is a basic example that shows how to use it.

You can download it on [github](https://github.com/ritchie46/structural_engineering)

![view](/img/fem/example_3/example.png)

```python
# if using ipython notebook
%matplotlib inline

import StructuralEngineering.FEM.system as se

# Create a new system object.
system = se.SystemElements()

# Add beams to the system. Positive z-axis is down, positive x-axis is the right.
system.add_element(location_list=[[0, 0], [0, -5]], EA=15000, EI=5000)
system.add_element(location_list=[[0, -5], [5, -5]], EA=15000, EI=5000)
system.add_element(location_list=[[5, -5], [5, 0]], EA=15000, EI=5000)

# Add supports.
system.add_support_fixed(nodeID=1)
# Add a rotational spring at node 4.
system.add_support_spring(nodeID=4, translation=3, K=4000)

# Add loads.
system.point_load(Fx=30, nodeID=2)
system.q_load(q=10, elementID=2)

system.show_structure()
system.solve()
```

![structure](/img/fem/example_3/structure_1.png)

```python
system.show_reaction_force()
```

![](/img/fem/example_3/reaction_3.png)

```python
system.show_normal_force()
```

![](/img/fem/example_3/normal_3.png)

```python
system.show_shear_force()
```

![](/img/fem/example_3/shear_3.png)

```python
system.show_bending_moment()
```

![](/img/fem/example_3/moment_3.png)

```python
system.show_displacement()
```

![](/img/fem/example_3/displacement_3.png)

