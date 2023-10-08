Introduction
------------

This page introduces terminology used in regression in C to help users make
sense of documentation and naming. Let's begin by doing a very quick overview
of regression analysis and the components of it.

Regression analysis is the process of estimating the relationship between a
dependent variable and one or more indepdent variable. It is often described
mathematically as:

.. math::

   Y_i = f(X_i, \beta) + e_i \quad \textrm{with} \quad i = 1,..,n,

where

- :math:`Y_i` is the dependent variable
- :math:`X_i` are the independent variables
- :math:`\beta` are the unknown parameters/coefficients of the function to be
  estimated
- :math:`e_i` are error terms, which are recorded in data but not directly
  observed
- :math:`i` is a single row of data with the values of the independent variables
  and the corresponding value of the dependent variable
- :math:`n` is the total number of rows or observations

The regression in C package focusses on enabling regression analysis for
complex-valued variables, meaning variables that have real and imaginary parts.
There are many Python packages that perform regression for real-valued numbers,
but few have reliable support for complex-values. Regression in C tries to fill
this gap.

Complex-valued regression is common in areas such as signal processing, for
example when doing analysis in the frequency domain.

For more information about complex-valued regression or the terminology and
design of regression in C, see the links below.

.. toctree::
   :maxdepth: 2

   design.rst
   terminology.rst
   references.rst
