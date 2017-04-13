+++

date = "2017-01-12T21:35:33+02:00"
draft = false
title = "Python 1D FEM Example 1"

+++

# Example 1: Framework

![view](/img/fem/example_1/example_1.png)

```python
# if using ipython notebook
%matplotlib inline

import StructuralEngineering.FEM.system as se

# Create a new system object.
system = se.SystemElements()

# Add beams to the system. Positive z-axis is down, positive x-axis is the right.
system.add_element(location_list=[[0, 0], [3, -4]], EA=5e9, EI=8000)
system.add_element(location_list=[[3, -4], [8, -4]], EA=5e9, EI=4000)

# get a visual of the element ID's and the node ID's
system.show_structure()
```

![](/img/fem/example_1/wosupports_1.png)

```python
# add loads to the element ID 2
system.q_load(elementID=2, q=10, direction=1)

# add hinged support to node ID 1
system.add_support_hinged(nodeID=1)

# add fixed support to node ID 2
system.add_support_fixed(nodeID=3)

# solve
system.solve()

# show the structure
system.show_structure()
```

![](/img/fem/example_1/supports_1_.png)

```python
# show the reaction forces
system.show_reaction_force()
```

![](/img/fem/example_1/reaction_1.png)

```python
# show the normal force
system.show_normal_force()
```

![](/img/fem/example_1/normal_1.png)

```python
# show the shear force
system.show_shear_force()
```

![](/img/fem/example_1/shear_1.png)

```python
# show the bending moment
system.show_bending_moment()
``` 
![](/img/fem/example_1/moment_1.png)

```python
# show the displacements
system.show_displacement()
``` 
![](/img/fem/example_1/displacement_1.png)
