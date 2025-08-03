# Dispersion Relations

<center><img src="docs/source/_static/DR_light.png" width="300"></center>

The `dispersionrelations` package bundles methods commonly used in *phenomenological particle physics*.
Included in the package and documented below are commonly occurring constants, such as masses and decay widths,
auxiliary functions used in kinematics as well as relations for the description of the dynamics of a system,
ranging from simple vertex prescriptions to classes for computing dispersion relations,
making use of the **unitarity** and the **analytic structure** of the **S-matrix**.
The package was developed in the course of doctoral studies at the [University of Bonn](https://uni-bonn.de/).

## Authors

- George Chanturia (maintainer)
- Leon Antonius Heuser
- Miriam Penners

## Content

The package is divided in the following submodules:

- The `constants` module includes a collection of physical and mathematical constants and related functions used throughout the package.
- The `utilities` module provides complex functions and statistical methods.
- The `kinematics` module accommodates functions necessary for particle kinematics such as phase space functions, etc.
- The `dynamics` module houses integration vertices, particle propagators, channel definitions, etc.
- The `integrals` module provides numerical integration routines for dispersion integrals.