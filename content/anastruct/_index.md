# anaStruct: Non linear structural analysis in Python

[**download here**](https://github.com/ritchie46/anaStruct)

{{< figure src="/img/post-10-water/moment_1.png" title="Bending moment diagram" >}}


Structural analysis is often done in proprietary software. During my works as a structural engineer this often a burden in my quest of automation. There were a lot of repetitive tasks when making models and there is most of the times no other way to interact with the program than using your mouse and keyboard. 

To be able to interact with 2D frames and trusses structural analysis, I wrote anaStruct. AnaStruct is a structural analysis package written in Python. This solved a few problems. I now could have a model as code, and thus utilize version control and have templates between projects. Furthermore you now can solve whole different problems, as you can feed the outcome of a linear calculation in another iteration of the model. Iterative mechanical problems can now easily be modelled. 

### Examples

* [Accumulating water analysis](https://www.ritchievink.com/blog/2017/08/23/a-nonlinear-water-accumulation-analysis-in-python/)
* [Simple example](https://www.ritchievink.com/blog/2017/01/12/python-1d-fem-example-1/)

### Reference guide

[reference](http://anastruct.readthedocs.io)


