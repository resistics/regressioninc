.. regressioninc documentation master file, created by
   sphinx-quickstart on Wed Oct 19 21:49:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to regressioninc's documentation!
=========================================

Regression in C (RegressionInC) is a package focussing on regression for complex-valued
problems. There are already multiple packages for regression in Python, however
these offer varying amounts of support for the complex-valued variables. Numpy's
least squares implementation has partial support for complex-valued variables
and other packages support multi-target regression, which can be used for
complex-valued problems, but again support is spotty.

This package is currently being developed. The plan for current and future
development includes:

- Least squares and weighted least squares implementation
- Visualisation of complex-valued linear problems
- Robust linear regression for complex-valued variables
- Uncertainty estimation

Next steps
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   introduction.rst
   examples_complex_regression/index.rst
   examples_linear_models/index.rst

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   regressioninc.rst

.. toctree::
   :maxdepth: 2
   :caption: Contributing:

   development.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
