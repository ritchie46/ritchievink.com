+++

date = "2017-03-12T21:35:33+02:00"
draft = false
title = "Python 1D FEM Example 3"
description = "Example code for 1D FEM in Python."

+++

# Python 1D FEM Example 3.

Simple code example for [anaStruct](https://github.com/ritchie46/anaStruct).

![view](/img/fem/example_3/example.png)

```python
# if using ipython notebook
%matplotlib inline

from anastruct.fem.system import SystemElements

# Create a new system object.
ss = SystemElements(EA=15000, EI=5000)

# Add beams to the system.
ss.add_element(location=[[0, 0], [0, 5]])
ss.add_element(location=[[0, 5], [5, 5]])
ss.add_element(location=[[5, 5], [5, 0]])

# Add a fixed support at node 1.
ss.add_support_fixed(node_id=1)

# Add a rotational spring at node 4.
ss.add_support_spring(node_id=4, translation=3, k=4000)

# Add loads.
ss.point_load(Fx=30, node_id=2)
ss.q_load(q=-10, element_id=2)

ss.show_structure()
ss.solve()
```

![structure](/img/fem/example_3/structure_1.png)

```python
ss.show_reaction_force()
```

![](/img/fem/example_3/reaction_3.png)

```python
ss.show_axial_force()
```

![](/img/fem/example_3/normal_3.png)

```python
ss.show_shear_force()
```

![](/img/fem/example_3/shear_3.png)

```python
ss.show_bending_moment()
```

![](/img/fem/example_3/moment_3.png)

```python
ss.show_displacement()
```

![](/img/fem/example_3/displacement_3.png)

