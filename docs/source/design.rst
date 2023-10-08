Design
^^^^^^

Regression in C has been developed as a standalone regression package with few
dependencies on other packages. There are many good regression packages in
Python, however few of these have widespread support for complex-valued data
therefore building widespread dependencies on those packages is risky and would
potentially be prone to failure or silent errors.

The package structure of regressioninc is:

- Modules at the root package level to implement shared functions
- Subpackages for different types of regression, for example linear models

With regards to estimators, models
