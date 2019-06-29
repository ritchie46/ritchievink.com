# anaStruct: Non linear structural analysis in Python

[**download here**](https://github.com/ritchie46/anaStruct)

{{< figure src="/img/post-10-water/moment_1.png" title="Bending moment diagram" >}}


Structural analysis is often done in proprietary software. During my work as a structural engineer this often was a burden in my quest to automation. Most programs offered you no other interface than a GUI, leaving you to the task of typing and clicking data from program A to program B. 

To be able to analyze 2D frames and trusses, I wrote anaStruct. AnaStruct is a structural analysis package written in Python. This solved a few problems. Models were now represented in code, and thus can be saved in version control and have templates between projects. Furthermore you now can solve whole different problems, as you can feed the outcome of a linear calculation in another iteration of the model. Iterative mechanical problems can now easily be modeled. 

### Examples

* [Accumulating water analysis](https://www.ritchievink.com/blog/2017/08/23/a-nonlinear-water-accumulation-analysis-in-python/)
* [Simple example](https://www.ritchievink.com/blog/2017/01/12/python-1d-fem-example-1/)

### Reference guide

[reference](http://anastruct.readthedocs.io)


