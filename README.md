[![DOI](https://zenodo.org/badge/1220023342.svg)](https://doi.org/10.5281/zenodo.22726613)

# Pyloa - (Py)thon (l)ocation (o)ptimization (a)lgorithms

Pyloa is a collection of methods for solving, (i) location problems in the Euclidian plane, (ii), location problems on a network (graph) and, (iii) discrete facility location problem. The package is primarily intended for teaching purposes but might to some extend also be useful for research. See the package's [documentation](https://andreasklose.github.io/pyloa/) and the below mentioned book for further information.

A number of the package's procedures require the availability of a MIP/MILP solver. The time being, either Cplex (via docplex) or GuRoBi can the used to this end. As the size of MIP formulations of location problems quickly exceed the limits of these solvers' community versions, it is recommended to install the MIP solver's fully licensed version.

**Remark**: Log-output to the screen can mostly be suppressed by setting *problem.silent=True*, where problem is
an instance of a location problem in pyloa. GuRoBi will then nevertheless print messages regarding the license
or parameter settings to the screen. To fully suppress all output from GuRoBi, create a file named *gurobi.env* in the folder from which you are using pyloa and add the following to this file::

    OutputFlag 0
    
### References

[Klose A (2026) Optimisation Models and Methods for Location Planning -- With Implementations in Python. Graduate Texts in Operations Research. Springer Nature.](https://link.springer.com/book/9783032303912)
