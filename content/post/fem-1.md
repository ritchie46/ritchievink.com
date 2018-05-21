+++

date = "2017-01-12T21:35:33+02:00"
draft = false
title = "Python 1D FEM Example 1"
description = "Example code for 1D FEM in Python."
tags = ["fem"]

+++

# Example 1: Framework

Simple code example for [anaStruct](https://github.com/ritchie46/anaStruct).

![view](/img/fem/example_1/example_1.png)

```python
# if using ipython notebook
%matplotlib inline

from anastruct.fem.system import SystemElements

# Create a new system object.
ss = SystemElements()

# Add beams to the system.
ss.add_element(location=[[0, 0], [3, 4]], EA=5e9, EI=8000)
ss.add_element(location=[[3, 4], [8, 4]], EA=5e9, EI=4000)

# get a visual of the element IDs and the node IDs
ss.show_structure()
```

![](/img/fem/example_1/wosupports_1.png)

```python
# add loads to the element ID 2
ss.q_load(element_id=2, q=-10)

# add hinged support to node ID 1
ss.add_support_hinged(node_id=1)

# add fixed support to node ID 2
ss.add_support_fixed(node_id=3)

# solve
ss.solve()

# show the structure
ss.show_structure()
```

![](/img/fem/example_1/supports_1_.png)

```python
# show the reaction forces
ss.show_reaction_force()
```

![](/img/fem/example_1/reaction_1.png)

```python
# show the axial forces
ss.show_axial_force()
```

![](/img/fem/example_1/normal_1.png)

```python
# show the shear force
ss.show_shear_force()
```

![](/img/fem/example_1/shear_1.png)

```python
# show the bending moment
ss.show_bending_moment()
``` 
![](/img/fem/example_1/moment_1.png)

```python
# show the displacements
ss.show_displacement()
``` 
![](/img/fem/example_1/displacement_1.png)


