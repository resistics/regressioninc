Introduction
------------

This page introduces some ideas behind |pkgnm| to help users quickly understand
its purpose and the thoughts behind its design. Let's begin by doing a very
quick overview of regression analysis and the components of it.

Regression analysis is the process of estimating the relationship between a
dependent variable and one or more indepdent variables. It is often described
mathematically as:

.. math::

   Y_i = f(X_i, \beta) + e_i \quad \textrm{with} \quad i = 1,..,n,

where,

- :math:`Y_i` is the dependent variable or the **regressand**
- :math:`X_i` are the independent variables or the **regressors**
- :math:`\beta` are the unknown **parameters** (also sometimes called the
  coefficients) of the function to be estimated
- :math:`e_i` are error terms, which are recorded in data but not directly
  observed
- :math:`i` is a single row of data with the values of the independent variables
  and the corresponding value of the dependent variable
- :math:`n` is the total number of rows or observations

The |pkgnm| package focusses on enabling regression analysis for complex-valued
variables, meaning variables that have real and imaginary parts. There are many
Python packages that perform regression for real-valued data, but few have
reliable support for complex-values. |pkgnm| tries to fill this gap.

Complex-valued regression is common in areas such as signal processing, for
example when doing analysis in the frequency domain.

For more information about the design of |pkgnm|, see below. For more background
information, please see the references and for more information about the
terminology used within |pkgnm|, refer to the terminology page.

Design
^^^^^^

|pkgnm| has been developed as a standalone regression package with few
dependencies on other packages. There are many good regression packages in
Python, however few of these have widespread support for complex-valued data
therefore building widespread dependencies on those packages is risky and would
potentially be prone to failure or silent errors.

The package structure of |pkgnm| is:

- Modules at the root package level to implement shared functions
- Subpackages for different types of regression, for example linear models


References
^^^^^^^^^^
The following references have been useful in the development of |pkgnm|.

Regression analysis
"""""""""""""""""""

- `Introduction to regression analysis on Wikipedia <https://en.wikipedia.org/wiki/Regression_analysis>`_
- `Dependent and independent variables on Wikipedia <https://en.wikipedia.org/wiki/Dependent_and_independent_variables>`_
- `Robust regression and norms in statsmodels <https://www.statsmodels.org/stable/examples/notebooks/generated/robust_models_1.html>`_


Complex-valued regression analysis
""""""""""""""""""""""""""""""""""

- `Impact of converting complex-valued problems to real-valued <https://stats.stackexchange.com/questions/66088/analysis-with-complex-data-anything-different>`_
- `Random gaussian noise for complex data <https://stackoverflow.com/questions/55700338/how-to-generate-a-complex-gaussian-white-noise-signal-in-pythonor-numpy-scipy>`_
