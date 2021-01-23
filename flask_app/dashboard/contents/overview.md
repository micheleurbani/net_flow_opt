# Welcome to Net_Flow_Opt

This webapp is an interactive version of the model presented in the paper
*"Maintenance optimization on a network flow problem: a bi-objective
approach"*.

## Workflow

The workflow is pretty straightforward and it consists of the following three
steps:

1. Declaration of components' properties.
2. Setup the parameters of the multi-objective genetic algorithm (MOGA).
3. Analyze the solution: load the results of an experiment,
and browse through all the available maintenance plans.

### Declaration of components

Declaration of component parameters can be done using the page "Component" in
the side menu.
It is possible to declare the shape and scale factors of the Weibull
distribution that model the failure rate, and the costs of preventive and
corrective maintenance can also be declared.

### Setup the MOGA

The parameters required to run the multi-objective GA can be declared using the
page "MOGA settings" in the side menu.

### Solution analysis

Soultion analysis allows to explore the quality of the solutions found by the
MOGA by representig the maintenance plans through Gantt charts.
